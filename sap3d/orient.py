"""Which quarter turn about the vertical axis best reproduces the source object outside the edit region.

The candidates are the four quarter turns about +Y, the vertical axis of a glTF asset as trimesh loads it.
Each is scored by the exact distance to the source mesh with the edit region excluded; without the
exclusion the newly generated content dominates the score and no turn ever wins. A turn is applied only
when it beats leaving the mesh alone by the given ratio.
"""

import time

import numpy as np
import trimesh

from . import distance

VERTICAL_AXIS = 1
QUARTER_TURNS = (0, 90, 180, 270)


def vertical_rotation(degrees: int) -> np.ndarray:
    """The right-handed quarter turn about +Y that multiplies a column vector."""
    if degrees % 90 != 0:
        raise ValueError(f"degrees must be a multiple of 90, got {degrees}")
    radians = np.radians(float(degrees))
    cos, sin = np.rint(np.cos(radians)), np.rint(np.sin(radians))
    first, second = (axis for axis in range(3) if axis != VERTICAL_AXIS)
    matrix = np.zeros((3, 3))
    matrix[VERTICAL_AXIS, VERTICAL_AXIS] = 1.0
    matrix[first, first] = cos
    matrix[second, second] = cos
    matrix[first, second] = sin
    matrix[second, first] = -sin
    return matrix


def turn_points(points: np.ndarray, degrees: int, pivot) -> np.ndarray:
    """`points` turned about the vertical axis through `pivot`; a multiple of 360 returns the same array."""
    if degrees % 360 == 0:
        return points
    center = np.asarray(pivot, dtype=np.float64)
    return (np.asarray(points, dtype=np.float64) - center) @ vertical_rotation(degrees).T + center


def turn(mesh: trimesh.Trimesh, degrees: int, pivot) -> trimesh.Trimesh:
    if degrees % 360 == 0:
        return mesh
    return trimesh.Trimesh(vertices=turn_points(mesh.vertices, degrees, pivot), faces=mesh.faces,
                           process=False)


def inverse_turn(degrees: int) -> int:
    if degrees % 90 != 0:
        raise ValueError(f"degrees must be a multiple of 90, got {degrees}")
    return int(-degrees) % 360


def best_orientation(mesh: trimesh.Trimesh, truth: trimesh.Trimesh, sdf, pivot, *, min_improvement: float,
                     num_points: int = distance.SURFACE_POINTS,
                     seed: int = distance.SAMPLE_SEED) -> tuple[int, dict]:
    """(degrees, record): the turn carrying `mesh` onto `truth`, or 0 when none wins by `min_improvement`.

    `mesh`, `truth` and the region behind `sdf` must share the source frame, and `pivot` must be the
    source's bounding-box center. When either surface lies wholly inside the region the search declines
    and returns 0; the record says so.
    """
    started = time.time()
    try:
        scores = {degrees: float(distance.compare(truth, turn(mesh, degrees, pivot), sdf, num_points,
                                                  seed)["symmetric"])
                  for degrees in QUARTER_TURNS}
    except distance.RegionSwallowedSurface as swallowed:
        return 0, {"degrees": 0, "best_degrees": 0, "identity_ratio": None,
                   "min_improvement_ratio": min_improvement, "masked_distance": {},
                   "declined": str(swallowed), "seconds": round(time.time() - started, 2)}
    identity = scores[0]
    best = min(QUARTER_TURNS, key=lambda degrees: scores[degrees])
    if scores[best] <= 0.0:
        raise ValueError(f"the masked distance to the source mesh is {scores[best]} at {best} degrees; "
                         f"two different meshes cannot agree exactly outside the region")
    ratio = identity / scores[best]
    applied = best if ratio >= min_improvement else 0
    return applied, {"degrees": applied, "best_degrees": best, "identity_ratio": round(ratio, 6),
                     "min_improvement_ratio": min_improvement,
                     "masked_distance": {str(d): round(scores[d], 8) for d in QUARTER_TURNS},
                     "declined": None, "seconds": round(time.time() - started, 2)}
