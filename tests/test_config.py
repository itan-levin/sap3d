"""Settings: defaults, the shipped YAML files, and rejection of unknown or inconsistent settings."""

import tempfile
from pathlib import Path

from sap3d import config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _rejects(values: dict) -> bool:
    try:
        config.from_dict(values)
    except ValueError:
        return True
    return False


def test_defaults():
    cfg = config.load_config()
    assert cfg.mode == "local"
    assert (cfg.sampler.num_steps, cfg.sampler.guidance_scale, cfg.sampler.seed) == (30, 5.0, 43)
    assert (cfg.schedule.pre_mask_strength, cfg.schedule.outside_strength, cfg.schedule.inside_strength) == \
        (0.7064, 1.0, 0.0)
    assert (cfg.schedule.mask_from_step, cfg.schedule.drop_from_step) == (8, 16)
    assert (cfg.probe_3d.query_points, cfg.probe_3d.knee_factor) == (1000, 0.8344)
    assert cfg.probe_2d.heads == ((15, 6), (14, 0)) and cfg.probe_2d.knee_factor == 2.5394
    assert (cfg.pose.surface_points, cfg.pose.sample_seed) == (200_000, 11)


def test_shipped_files():
    assert config.load_config(CONFIGS / "edit.yaml") == config.EditConfig()
    uniform = config.load_config(CONFIGS / "edit_global_uniform.yaml")
    assert uniform.mode == "uniform" and uniform.uniform_strength == 0.5
    assert uniform.sampler == config.SamplerConfig()


def test_dump_round_trip():
    cfg = config.from_dict({"mode": "uniform", "probe_2d": {"heads": [[3, 1]]}, "sampler": {"seed": 7}})
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "c.yaml"
        path.write_text(config.dump_config(cfg))
        assert config.load_config(path) == cfg


def test_rejections():
    assert _rejects({"sampler": {"steps": 30}})
    assert _rejects({"unknown": 1})
    assert _rejects({"mode": "both"})
    assert _rejects({"schedule": {"mask_from_step": 0}})
    assert _rejects({"schedule": {"drop_from_step": 8}})
    assert _rejects({"schedule": {"drop_from_step": 30}})
    assert _rejects({"sampler": {"num_steps": "30"}})
    assert _rejects({"pose": {"enabled": 1}})
    assert _rejects({"probe_2d": {"heads": []}})
    assert _rejects({"probe_2d": {"heads": [[15.7, 6]]}})
    assert _rejects({"probe_2d": {"heads": [[15, True]]}})
    assert _rejects({"probe_2d": {"heads": [[21, 0]]}})
    assert _rejects({"probe_2d": {"heads": [[15, 6], [15, 6]]}})
    assert _rejects({"probe_2d": {"heads": ["15"]}})
    assert _rejects({"knee": {"smoothing_width": 50}})
    assert _rejects({"pose": {"surface_points": 0}})
    assert _rejects({"sampler": {"octree_resolution": 0}})
    assert not _rejects({"sampler": {"guidance_scale": 7}})
