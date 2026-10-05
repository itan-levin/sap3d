"""Edit-stage settings: frozen dataclasses holding the default values, and a YAML loader.

A YAML file names any subset of the settings by section; every key it names must exist, so a misspelled
setting fails instead of silently leaving the default in force.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODES = ("local", "uniform")

# Cross-attention layers and heads of the Hunyuan3D-2.1 DiT.
DIT_LAYERS, DIT_HEADS = 21, 16


@dataclass(frozen=True)
class SamplerConfig:
    num_steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 43
    octree_resolution: int = 384
    num_chunks: int = 20000


@dataclass(frozen=True)
class ScheduleConfig:
    pre_mask_strength: float = 0.7064
    outside_strength: float = 1.0
    inside_strength: float = 0.0
    mask_from_step: int = 8
    drop_from_step: int = 16


@dataclass(frozen=True)
class Probe3DConfig:
    query_points: int = 1000
    query_seed: int = 0
    knee_factor: float = 0.8344


@dataclass(frozen=True)
class Probe2DConfig:
    heads: tuple = ((15, 6), (14, 0))
    coverage: float = 0.5
    knee_factor: float = 2.5394


@dataclass(frozen=True)
class KneeConfig:
    smoothing_width: int = 51
    sensitivity: float = 1.0


@dataclass(frozen=True)
class PoseConfig:
    enabled: bool = True
    region_min_improvement: float = 1.5
    output_min_improvement: float = 1.5
    surface_points: int = 200_000
    sample_seed: int = 11


@dataclass(frozen=True)
class ConditioningConfig:
    crop_alpha_min: int = 127
    near_white_luma: float = 240.0
    min_component_px: int = 64
    connectivity: int = 4


@dataclass(frozen=True)
class EditConfig:
    mode: str = "local"
    uniform_strength: float = 0.5
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    probe_3d: Probe3DConfig = field(default_factory=Probe3DConfig)
    probe_2d: Probe2DConfig = field(default_factory=Probe2DConfig)
    knee: KneeConfig = field(default_factory=KneeConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)


def _coerce(name: str, value, default):
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false, got {value!r}")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string, got {value!r}")
        return value
    if name == "probe_2d.heads":
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{name} must be a non-empty list of [layer, head] pairs, got {value!r}")
        pairs = []
        for pair in value:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in pair)
                    or not (0 <= pair[0] < DIT_LAYERS and 0 <= pair[1] < DIT_HEADS)):
                raise ValueError(f"{name} entries must be [layer, head] integers with 0 <= layer < {DIT_LAYERS} "
                                 f"and 0 <= head < {DIT_HEADS}, got {pair!r}")
            pairs.append((pair[0], pair[1]))
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"{name} lists a pair twice: {value!r}")
        return tuple(pairs)
    raise TypeError(f"no coercion for {name}")


def _build(cls, values: dict, prefix: str = ""):
    if not isinstance(values, dict):
        raise ValueError(f"{prefix.rstrip('.') or 'the configuration'} must be a mapping")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"unknown setting(s) {[prefix + k for k in unknown]}; known here: {sorted(known)}")
    defaults = cls()
    kwargs = {}
    for name, value in values.items():
        default = getattr(defaults, name)
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{prefix}{name}.")
        else:
            kwargs[name] = _coerce(prefix + name, value, default)
    return cls(**kwargs)


def validate(cfg: EditConfig) -> EditConfig:
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {cfg.mode!r}")
    sampler, schedule = cfg.sampler, cfg.schedule
    for name, value, least in (("sampler.num_steps", sampler.num_steps, 2),
                               ("sampler.octree_resolution", sampler.octree_resolution, 1),
                               ("sampler.num_chunks", sampler.num_chunks, 1),
                               ("probe_3d.query_points", cfg.probe_3d.query_points, 1),
                               ("pose.surface_points", cfg.pose.surface_points, 1)):
        if value < least:
            raise ValueError(f"{name} must be at least {least}, got {value}")
    if not 1 <= schedule.mask_from_step < schedule.drop_from_step < sampler.num_steps:
        raise ValueError(
            f"the schedule needs 1 <= mask_from_step < drop_from_step < num_steps, got "
            f"{schedule.mask_from_step}, {schedule.drop_from_step}, {sampler.num_steps}: the preliminary "
            f"pass that places the region shares the edit's steps before mask_from_step, and the region "
            f"only acts between mask_from_step and drop_from_step")
    for name, factor in (("probe_3d.knee_factor", cfg.probe_3d.knee_factor),
                         ("probe_2d.knee_factor", cfg.probe_2d.knee_factor)):
        if not factor > 0:
            raise ValueError(f"{name} must be positive, got {factor}")
    if not 0 < cfg.probe_2d.coverage <= 1:
        raise ValueError(f"probe_2d.coverage must lie in (0, 1], got {cfg.probe_2d.coverage}")
    if cfg.knee.smoothing_width < 1 or cfg.knee.smoothing_width % 2 == 0:
        raise ValueError(f"knee.smoothing_width must be a positive odd number, got {cfg.knee.smoothing_width}")
    if cfg.conditioning.connectivity not in (4, 8):
        raise ValueError("conditioning.connectivity must be 4 or 8")
    return cfg


def from_dict(values: dict) -> EditConfig:
    return validate(_build(EditConfig, values or {}))


def load_config(path=None) -> EditConfig:
    """The default configuration, overridden by the YAML file at `path` when one is given."""
    if path is None:
        return validate(EditConfig())
    return from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def to_dict(cfg: EditConfig) -> dict:
    values = dataclasses.asdict(cfg)
    values["probe_2d"]["heads"] = [list(pair) for pair in cfg.probe_2d.heads]
    return values


def dump_config(cfg: EditConfig) -> str:
    return yaml.safe_dump(to_dict(cfg), sort_keys=False, default_flow_style=False)
