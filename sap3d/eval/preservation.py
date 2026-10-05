"""Preservation: exact point-to-surface distance from the edited meshes to the source.

Local edits compare outside the region: both surfaces are sampled (200,000 points, fixed seed), samples
inside the region are dropped, and the two directional mean distances are averaged (`sap3d.distance`).
Global edits compare the whole surface. Distances are in the source mesh's units.
"""

from .. import distance, frames
from ..region import load_region


def preservation(source, meshes: dict, region=None) -> dict:
    """{name: distance record} for each mesh in `meshes` ({name: path}); `region` None compares everything."""
    truth = frames.load_mesh(source)
    sdf = distance.region_sdf(load_region(region).surface) if region is not None else None
    records = {name: distance.compare(truth, frames.load_mesh(path), sdf) for name, path in meshes.items()}
    for record in records.values():
        record["region_excluded"] = region is not None
    return records
