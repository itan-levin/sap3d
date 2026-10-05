"""Frames, quarter turns, the orientation search, and region placement on a turned preliminary mesh."""

import numpy as np
import trimesh

from sap3d import distance, frames, orient
from sap3d.region import from_primitives, place_components, points_inside

POINTS = 20_000


def _asymmetric_source() -> trimesh.Trimesh:
    body = trimesh.creation.box(extents=[0.6, 0.3, 0.2])
    arm = trimesh.creation.box(extents=[0.15, 0.4, 0.15])
    arm.apply_translation([0.22, 0.3, 0.0])
    nose = trimesh.creation.box(extents=[0.1, 0.1, 0.3])
    nose.apply_translation([-0.2, -0.05, 0.22])
    mesh = trimesh.util.concatenate([body, arm, nose])
    mesh.apply_translation([0.03, -0.02, 0.01])
    return mesh


def _arm_region():
    return from_primitives([{"type": "box", "center": [0.25, 0.53, 0.01], "extents": [0.3, 0.3, 0.3]}], "arm")


def test_frame_round_trip():
    source = _asymmetric_source()
    frame = frames.frame_of(source)
    model = frame.to_model(source.vertices)
    assert np.allclose(frame.to_source(model), source.vertices)
    extent = model.max(axis=0) - model.min(axis=0)
    assert abs(extent.max() - 2 * frames.MESH_NORM_SCALE) < 1e-9
    assert np.allclose((model.max(axis=0) + model.min(axis=0)) / 2, 0, atol=1e-12)


def test_quarter_turn_convention():
    assert np.allclose(orient.vertical_rotation(90) @ [1, 0, 0], [0, 0, -1])
    assert np.allclose(orient.vertical_rotation(90) @ [0, 1, 0], [0, 1, 0])
    for degrees in orient.QUARTER_TURNS:
        points = np.random.default_rng(0).normal(size=(10, 3))
        back = orient.turn_points(orient.turn_points(points, degrees, [0.1, 0, 0.2]),
                                  orient.inverse_turn(degrees), [0.1, 0, 0.2])
        assert np.allclose(back, points)


def test_search_recovers_planted_turns_and_declines_when_swallowed():
    source = _asymmetric_source()
    frame = frames.frame_of(source)
    sdf = distance.region_sdf(_arm_region().surface)
    for planted in (90, 180, 270):
        turned = orient.turn(source, planted, frame.center)
        degrees, record = orient.best_orientation(turned, source, sdf, frame.center, min_improvement=1.5,
                                                  num_points=POINTS)
        assert degrees == orient.inverse_turn(planted), (planted, record)
    degrees, _ = orient.best_orientation(source, source, sdf, frame.center, min_improvement=1.5, num_points=POINTS)
    assert degrees == 0
    whole = distance.region_sdf(from_primitives([{"type": "box", "center": list(frame.center),
                                                  "extents": [2, 2, 2]}], "all").surface)
    degrees, record = orient.best_orientation(source, source, whole, frame.center, min_improvement=1.5,
                                              num_points=POINTS)
    assert degrees == 0 and record["declined"]


def test_region_follows_a_preliminary_mesh_turned_by_90_and_270():
    source = _asymmetric_source()
    frame = frames.frame_of(source)
    region = _arm_region()
    sdf = distance.region_sdf(region.surface)
    arm_tip = np.array([0.25, 0.48, 0.01])
    for planted in (90, 270):
        preliminary = orient.turn(source, planted, frame.center)
        found, _ = orient.best_orientation(preliminary, source, sdf, frame.center, min_improvement=1.5,
                                           num_points=POINTS)
        components = place_components(region, frame, orient.inverse_turn(found))
        tip_in_preliminary = frame.to_model(orient.turn_points(arm_tip[None], planted, frame.center))
        tip_unturned = frame.to_model(arm_tip[None])
        assert points_inside(components, tip_in_preliminary).all(), planted
        assert not points_inside(components, tip_unturned).any(), planted
