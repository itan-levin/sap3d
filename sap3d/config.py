"""Edit-, texture- and personalize-stage settings: frozen dataclasses holding the default values, and YAML loaders.

A YAML file names any subset of the settings by section; every key it names must exist, so a misspelled
setting fails instead of silently leaving the default in force.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


@dataclass(frozen=True)
class TextureConfig:
    atlas_size: int = 2048
    target_faces: int = 40_000
    normal_gate: bool = True
    boundary_margin: float = 0.02
    feather_width: float = 0.01
    color_fit: bool = True
    realesrgan_ckpt: str = ""


@dataclass(frozen=True)
class DatasetConfig:
    random_views: int = 250
    render_size: int = 512
    render_samples: int = 128
    camera_fov: float = 49.1
    elevation_min: float = -5.0
    elevation_max: float = 40.0
    azimuth_min: float = -180.0
    azimuth_max: float = 180.0
    distance_scale_min: float = 1.0
    distance_scale_max: float = 1.5
    surface_points: int = 200_000
    seed: Optional[int] = None


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 5000
    batch_size: int = 4
    learning_rate: float = 7e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_eps: float = 1e-6
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    lr_factor_start: float = 0.1
    lr_factor_max: float = 1.0
    lr_factor_min: float = 0.5
    grad_clip_norm: float = 1.0
    seed: int = 0
    lora_rank: int = 16
    lora_alpha: int = 16
    points_per_sample: int = 81_920
    image_size: int = 518
    crop_margin: int = 5
    pad_ratio: float = 1.15


SELECTION_METRICS = ("corrected", "legacy")


@dataclass(frozen=True)
class ValidationConfig:
    every: int = 500
    num_steps: int = 50
    guidance_scale: float = 5.0
    octree_resolution: int = 384
    num_chunks: int = 8000
    chamfer_points: int = 50_000
    selection_metric: str = "corrected"


@dataclass(frozen=True)
class PersonalizeConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def _coerce(name: str, value, default):
    if name == "dataset.seed":
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name} must be an integer or null, got {value!r}")
        return value
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


# hy3dpaint decimates anything above this many faces itself, so a larger budget would paint a different
# surface from the one the atlas is baked on.
PAINT_FACE_BUDGET = 40_000


def validate_texture(cfg: TextureConfig) -> TextureConfig:
    if not 1 <= cfg.target_faces <= PAINT_FACE_BUDGET:
        raise ValueError(f"target_faces must lie in [1, {PAINT_FACE_BUDGET}], got {cfg.target_faces}")
    if cfg.atlas_size < 16:
        raise ValueError(f"atlas_size must be at least 16, got {cfg.atlas_size}")
    if not cfg.feather_width > 0 or cfg.boundary_margin < 0:
        raise ValueError(f"feather_width must be positive and boundary_margin non-negative, got "
                         f"{cfg.feather_width} and {cfg.boundary_margin}")
    return cfg


def load_texture_config(path=None) -> TextureConfig:
    """The default texture settings, overridden by the YAML file at `path` when one is given."""
    if path is None:
        return validate_texture(TextureConfig())
    return validate_texture(_build(TextureConfig, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}))


def validate_personalize(cfg: PersonalizeConfig) -> PersonalizeConfig:
    data, train, val = cfg.dataset, cfg.train, cfg.validation
    for name, value, least in (("dataset.random_views", data.random_views, 0),
                               ("dataset.render_size", data.render_size, 16),
                               ("dataset.render_samples", data.render_samples, 1),
                               ("dataset.surface_points", data.surface_points, 1),
                               ("train.steps", train.steps, 1),
                               ("train.batch_size", train.batch_size, 1),
                               ("train.warmup_steps", train.warmup_steps, 0),
                               ("train.lora_rank", train.lora_rank, 1),
                               ("train.lora_alpha", train.lora_alpha, 1),
                               ("train.points_per_sample", train.points_per_sample, 1),
                               ("train.image_size", train.image_size, 16),
                               ("train.crop_margin", train.crop_margin, 0),
                               ("validation.every", val.every, 1),
                               ("validation.num_steps", val.num_steps, 2),
                               ("validation.octree_resolution", val.octree_resolution, 1),
                               ("validation.num_chunks", val.num_chunks, 1),
                               ("validation.chamfer_points", val.chamfer_points, 1)):
        if value < least:
            raise ValueError(f"{name} must be at least {least}, got {value}")
    # 16 benchmark-style views plus the random ones are stored as three-digit file names.
    if data.random_views + 16 > 1000:
        raise ValueError(f"dataset.random_views must be at most 984, got {data.random_views}")
    if not data.elevation_min <= data.elevation_max or not data.azimuth_min <= data.azimuth_max:
        raise ValueError("dataset elevation and azimuth ranges must have min <= max")
    if not 0 < data.distance_scale_min <= data.distance_scale_max:
        raise ValueError("dataset.distance_scale_min must be positive and at most distance_scale_max")
    if not 0 < data.camera_fov < 180:
        raise ValueError(f"dataset.camera_fov must lie in (0, 180) degrees, got {data.camera_fov}")
    if train.warmup_steps >= train.steps:
        raise ValueError(f"train.warmup_steps must be below train.steps, got {train.warmup_steps} and {train.steps}")
    if not train.learning_rate > 0 or not train.adam_eps > 0 or train.weight_decay < 0 or not train.grad_clip_norm > 0:
        raise ValueError("train.learning_rate, adam_eps and grad_clip_norm must be positive and weight_decay "
                         "non-negative")
    if not (0 <= train.adam_beta1 < 1 and 0 <= train.adam_beta2 < 1):
        raise ValueError("train.adam_beta1 and adam_beta2 must lie in [0, 1)")
    if not 0 < train.lr_factor_min <= train.lr_factor_max or not 0 < train.lr_factor_start:
        raise ValueError("train learning-rate factors must be positive with lr_factor_min <= lr_factor_max")
    if not train.pad_ratio >= 1:
        raise ValueError(f"train.pad_ratio must be at least 1, got {train.pad_ratio}")
    if train.steps % val.every:
        raise ValueError(f"train.steps ({train.steps}) must be a multiple of validation.every ({val.every}), so "
                         f"that the last step is validated")
    if val.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"validation.selection_metric must be one of {list(SELECTION_METRICS)}, "
                         f"got {val.selection_metric!r}")
    return cfg


def load_personalize_config(path=None, overrides: Optional[dict] = None) -> PersonalizeConfig:
    """The default personalize settings, overridden by the YAML file at `path` and then by `overrides`
    (a mapping in the same sectioned form)."""
    values = {} if path is None else (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if not isinstance(values, dict):
        raise ValueError(f"{path} must hold a mapping of sections")
    for section, entries in (overrides or {}).items():
        current = values.get(section) or {}
        if not isinstance(current, dict):
            raise ValueError(f"{section} must be a mapping")
        values[section] = {**current, **entries}
    return validate_personalize(_build(PersonalizeConfig, values))


def personalize_to_dict(cfg: PersonalizeConfig) -> dict:
    return dataclasses.asdict(cfg)
