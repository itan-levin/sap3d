# Modified by Itan Levin (2026): DMC vertices placed in the decode bounds and `mc_level` honored; MC scales by the decoded grid; extractor lookup helpers.
# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Optional, Union, Tuple, List

import numpy as np
import torch
from skimage import measure


class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class SurfaceExtractor:
    def _compute_box(self, bounds: Union[Tuple[float], List[float], float]):
        """
        Compute bounding box minimum coordinates and size based on input bounds.

        Args:
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or a single
            float representing half side length.
                If float, bounds are assumed symmetric around zero in all axes.
                Expected format if list/tuple: [xmin, ymin, zmin, xmax, ymax, zmax].

        Returns:
            bbox_min (np.ndarray): Minimum coordinates of the bounding box (xmin, ymin, zmin).
            bbox_size (np.ndarray): Size of the bounding box along each axis (xmax - xmin, etc.).
        """
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        return bbox_min, bbox_max - bbox_min

    def run(self, *args, **kwargs):
        """
        Abstract method to extract surface mesh from grid logits.

        This method should be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        return NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        """
        Process a batch of grid logits to extract surface meshes.

        Args:
            grid_logits (torch.Tensor): Batch of grid logits with shape (batch_size, ...).
            **kwargs: Additional keyword arguments passed to the `run` method.

        Returns:
            List[Optional[Latent2MeshOutput]]: List of mesh outputs for each grid in the batch.
                If extraction fails for a grid, None is appended at that position.
        """
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)

        return outputs


class MCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, **kwargs):
        """
        Extract surface mesh using the Marching Cubes algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            mc_level (float): The level (iso-value) at which to extract the surface.
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or half side length.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, scaled and translated to bounding
                  box coordinates.
                - faces (np.ndarray): Extracted mesh faces (triangles).
        """
        vertices, faces, normals, _ = measure.marching_cubes(grid_logit.cpu().numpy(),
                                                             mc_level,
                                                             method="lewiner")
        bbox_min, bbox_size = self._compute_box(bounds)
        # The query grid is endpoint-inclusive, and FlashVDM decoding rounds the resolution down
        # from the one requested, so only the decoded array gives the true cell count.
        cells = np.array(grid_logit.shape) - 1
        vertices = vertices / cells * bbox_size + bbox_min
        return vertices, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    def __init__(self):
        # Construction must fail here when diso is unusable, because callers discard an
        # extraction-time exception as a null mesh. Loading touches no device, so this succeeds
        # without CUDA; only `run` needs a CUDA tensor.
        try:
            from diso import DiffDMC
        except ImportError as exc:
            raise ImportError(
                "The 'dmc' surface extractor requires the diso package, and the installed one "
                "could not be loaded. Install or rebuild it, or select marching cubes with "
                "mc_algo='mc'."
            ) from exc
        self.dmc = DiffDMC(dtype=torch.float32)

    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using Differentiable Marching Cubes (DMC) algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            mc_level (float): The level (iso-value) at which to extract the surface.
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, scaled to bounding box coordinates.
                - faces (np.ndarray): Extracted mesh faces (triangles), with reversed vertex order.

        Raises:
            ValueError: If the field never crosses the requested level, matching the marching-cubes
                extractor rather than returning an empty mesh.
        """
        device = grid_logit.device
        # The field is an occupancy logit, positive inside, where diso expects negative inside. The
        # level must be negated and scaled exactly as the field is, or it names a different surface.
        sdf = (-grid_logit / octree_resolution).to(torch.float32).contiguous()
        verts, faces = self.dmc(sdf, deform=None, isovalue=-mc_level / octree_resolution,
                                return_quads=False, normalize=True)
        if verts.shape[0] == 0:
            raise ValueError(
                f"The field never crosses level {mc_level}, so there is no surface to extract."
            )

        # Vertices must land in the bounds frame, as the marching-cubes extractor places them.
        bbox_min, bbox_size = self._compute_box(bounds)
        bbox_min_t = torch.tensor(bbox_min, device=device, dtype=torch.float32)
        bbox_size_t = torch.tensor(bbox_size, device=device, dtype=torch.float32)
        verts = verts * bbox_size_t + bbox_min_t

        vertices = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()[:, ::-1]
        return vertices, faces


SurfaceExtractors = {
    'mc': MCSurfaceExtractor,
    'dmc': DMCSurfaceExtractor,
}


def surface_extractor_from_name(mc_algo: str) -> SurfaceExtractor:
    """Build the extractor a caller asked for by name."""
    if mc_algo not in SurfaceExtractors:
        raise ValueError(f'Unsupported mc_algo {mc_algo}, available: {list(SurfaceExtractors)}')
    return SurfaceExtractors[mc_algo]()


def default_surface_extractor(is_cuda: bool) -> SurfaceExtractor:
    """Pick an extractor for a grid on the given kind of device.

    `is_cuda` must describe the grid that will be extracted, not merely whether CUDA exists: the
    differentiable extractor rejects a CPU tensor, and a model can be placed on CPU on a CUDA host.
    """
    return DMCSurfaceExtractor() if is_cuda else MCSurfaceExtractor()
