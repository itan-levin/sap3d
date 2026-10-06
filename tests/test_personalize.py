"""Personalize stage logic that needs no GPU: the training sample transform, the Chamfer definitions, the legacy
surface extraction, the validation draw sequence, the learning-rate schedule, checkpoint selection, settings,
and command-line validation. Expected values are fixed reference outputs on synthetic inputs."""

import hashlib
import json
import math
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from sap3d import config
from sap3d.personalize import cli, select
from sap3d.personalize import validate as validation
from sap3d.personalize.data import SampleStream
from sap3d.personalize.train import make_optimizer

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _digest(array) -> str:
    array = array.numpy() if torch.is_tensor(array) else array
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _synthetic_render(path: Path) -> None:
    rng = np.random.RandomState(7)
    h, w = 300, 260
    image = rng.randint(0, 256, (h, w, 4)).astype(np.uint8)
    yy, xx = np.mgrid[:h, :w]
    alpha = np.clip(255 * (1.2 - np.hypot((yy - 140) / 90, (xx - 120) / 70)), 0, 255).astype(np.uint8)
    alpha[(yy - 140) ** 2 / 90 ** 2 + (xx - 120) ** 2 / 70 ** 2 > 1] = 0
    image[:, :, 3] = alpha
    cv2.imwrite(str(path), image)


def _dataset(folder: Path, views: int = 3, points: int = 1000) -> Path:
    (folder / "renders").mkdir(parents=True)
    for index in range(views):
        _synthetic_render(folder / "renders" / f"{index:03d}.png")
    np.savez_compressed(folder / "surface.npz", points=np.random.RandomState(3).randn(points, 6).astype(np.float32))
    return folder


def _raises(function, *args, kind=Exception) -> bool:
    try:
        function(*args)
    except kind:
        return True
    return False


def test_sample_transform_matches_research():
    with tempfile.TemporaryDirectory() as folder:
        stream = SampleStream(_dataset(Path(folder)), config.TrainConfig(points_per_sample=500))
        image, mask = stream.load_render(stream.paths[0])
        surface = stream.surface(np.random.default_rng(5))
    assert tuple(image.shape) == (3, 518, 518) and tuple(mask.shape) == (1, 518, 518)
    assert tuple(surface.shape) == (500, 7) and bool((surface[:, 6] == 0).all())
    assert (_digest(image), _digest(mask), _digest(surface)) == \
        ("c6b197c3aa55b1ac", "38799228111f89af", "f35b91bccec2e73b")


def test_render_choice_sequence():
    stream = SampleStream.__new__(SampleStream)
    stream.paths = [f"{i:03d}.png" for i in range(266)]
    stream.points = np.zeros((10, 6), np.float32)
    stream.cfg = config.TrainConfig(points_per_sample=5)
    stream.rng = random.Random(0)
    stream.load_render = lambda path: (torch.zeros(1), torch.zeros(1))
    drawn = [name for _ in range(3) for name in stream.batch(4)["renders"]]
    assert drawn == ["197.png", "215.png", "020.png", "132.png", "261.png", "248.png", "207.png", "155.png",
                     "244.png", "183.png", "111.png", "258.png"]


def test_validation_item_sequence():
    class Samples:
        def batch(self, size):
            return {}
    stream = validation.ValidationStream(Samples(), 4, seed=0)
    assert [stream.next()[1] for _ in range(6)] == [2, 0, 1, 0, 0, 2]


def _mesh_and_points():
    rng = np.random.RandomState(0)
    vertices, faces = rng.randn(60, 3), rng.randint(0, 60, (100, 3))
    return vertices, faces, rng.randn(3000, 3) * 0.5


def test_chamfer_definitions_match_research():
    vertices, faces, gt = _mesh_and_points()
    pred = validation.sample_mesh_surface(vertices, faces, 2000, 42)
    assert math.isclose(float(pred.sum()), 997.6533160979664, rel_tol=0, abs_tol=1e-9)
    assert validation.chamfer(pred, gt) == (0.0562807056749472, 0.1627266470925563)
    assert validation.legacy_chamfer(pred, gt) == (0.04776737214458616, 0.15550459281533277, 180)
    assert validation.sample_mesh_surface(vertices, np.zeros((0, 3), np.int64), 10, 42) is None


def test_legacy_marching_cubes_matches_research():
    resolution = 48
    axis = torch.linspace(-1.01, 1.01, resolution + 1)
    coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1)
    field = (0.6 - (coords - torch.tensor([0.1, -0.2, 0.05])).norm(dim=-1)) * torch.tensor([1.0, 1.3, 0.8]).mean()
    mesh = validation.LegacyMarchingCubes()(field.unsqueeze(0), mc_level=0.0, bounds=1.01,
                                            octree_resolution=resolution)[0]
    assert len(mesh.mesh_f) == 7648
    assert math.isclose(float(mesh.mesh_v.astype(np.float64).sum()), -419.0474993366515, rel_tol=0, abs_tol=1e-6)


def test_learning_rate_schedule():
    class Models:
        dit = torch.nn.Linear(2, 2)
    cfg = config.TrainConfig()
    optimizer, scheduler = make_optimizer(Models(), cfg)
    factors = {}
    for step in range(5001):
        factors[step] = optimizer.param_groups[0]["lr"] / cfg.learning_rate
        optimizer.step()
        scheduler.step()
    expected = {0: 0.1, 1: 0.109, 50: 0.55, 99: 0.991, 100: 1.0, 101: 0.9999999486172216, 2550: 0.75,
                4999: 0.5000000513827784, 5000: 0.5}
    assert all(math.isclose(factors[step], value, rel_tol=1e-12) for step, value in expected.items())
    group = optimizer.param_groups[0]
    assert (group["betas"], group["eps"], group["weight_decay"]) == ((0.9, 0.99), 1e-6, 1e-2)


def _validations(base=0.02):
    return [{"step": 0, "chamfer_distance": base, "legacy_chamfer_distance": base},
            {"step": 500, "chamfer_distance": 0.010, "legacy_chamfer_distance": 0.012},
            {"step": 1000, "chamfer_distance": 0.009, "legacy_chamfer_distance": 0.013},
            {"step": 1500, "chamfer_distance": 0.009, "legacy_chamfer_distance": 0.011}]


def test_selection_rules():
    assert select.choose(_validations(), "corrected")["step"] == 1000
    assert select.choose(_validations(), "legacy")["step"] == 1500
    assert select.choose(_validations(base=0.005), "corrected")["step"] == 1000
    assert _raises(select.choose, _validations()[:1], "corrected", kind=ValueError)
    assert _raises(select.choose, _validations(), "squared", kind=ValueError)


def test_select_copies_the_chosen_step():
    with tempfile.TemporaryDirectory() as folder:
        run = Path(folder)
        for step in (500, 1000, 1500):
            (run / "steps" / f"step_{step}").mkdir(parents=True)
            (run / "steps" / f"step_{step}" / "adapter_model.safetensors").write_text(str(step))
        record = {"settings": {"validation": {"selection_metric": "corrected"}}, "validations": _validations()}
        (run / "training.json").write_text(json.dumps(record))
        assert select.select(run)["step"] == 1000
        assert (run / "adapter" / "adapter_model.safetensors").read_text() == "1000"
        assert select.select(run, "legacy") == {"step": 1500, "metric": "legacy", "value": 0.011,
                                                "untrained_scored_lower": False}
        record["validations"][0]["chamfer_distance"] = 0.005
        (run / "training.json").write_text(json.dumps(record))
        assert select.select(run)["untrained_scored_lower"] is True
        assert select.select(run, "legacy")["untrained_scored_lower"] is False
        assert (run / "adapter" / "adapter_model.safetensors").read_text() == "1500"
        assert json.loads((run / "training.json").read_text())["selected"]["step"] == 1500
        (run / "steps" / "step_1500" / "adapter_model.safetensors").unlink()
        assert _raises(select.select, run, "legacy", kind=FileNotFoundError)


def test_personalize_settings():
    assert config.load_personalize_config(CONFIGS / "personalize.yaml") == config.PersonalizeConfig()
    cfg = config.load_personalize_config(overrides={"dataset": {"seed": 3}, "validation": {"selection_metric": "legacy"}})
    assert cfg.dataset.seed == 3 and cfg.validation.selection_metric == "legacy"
    for values in ({"train": {"stepz": 1}}, {"train": {"steps": 5001}}, {"dataset": {"seed": 1.5}},
                   {"validation": {"selection_metric": "squared"}}, {"train": {"points_per_sample": 0}},
                   {"dataset": {"random_views": 990}}, {"train": {"warmup_steps": 5000}}):
        assert _raises(config.load_personalize_config, None, values, kind=ValueError), values


def test_cli_validation():
    def exits(argv):
        try:
            cli.main(argv)
        except SystemExit as stop:
            return stop.code not in (0, None)
        return False
    assert exits([])
    assert exits(["train", "--data", "x"])
    assert exits(["select", "--run", "x", "--selection-metric", "squared"])
    assert exits(["dataset", "--asset", "x", "--out", "y", "--seed", "1.5"])
