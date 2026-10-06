"""The shape pipeline with a per-object adapter whose strength can be set per token."""

import json
from pathlib import Path

import torch

ADAPTED_LAYERS = 138

# Settings of adapter_config.json that do not change the adapted layers.
NEUTRAL_CONFIG_KEYS = ("base_model_name_or_path", "revision", "inference_mode")
# PEFT keeps target_modules as a set, so its saved order is arbitrary.
UNORDERED_CONFIG_KEYS = ("target_modules",)


def read_adapter_config(adapter_dir) -> dict:
    path = Path(adapter_dir) / "adapter_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"no adapter_config.json in {adapter_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_settings(config: dict) -> dict:
    return {key: sorted(value) if key in UNORDERED_CONFIG_KEYS and isinstance(value, list) else value
            for key, value in config.items() if key not in NEUTRAL_CONFIG_KEYS}


def attach_adapter(model, adapter_dir, device="cuda") -> tuple:
    """(`model` carrying the adapter at `adapter_dir` as per-token-scalable layers, the adapter's config)."""
    from hy3dshape.models.per_sample_lora import convert_to_per_sample_lora
    from peft import PeftModel

    config = read_adapter_config(adapter_dir)
    adapted = convert_to_per_sample_lora(PeftModel.from_pretrained(model, str(adapter_dir), torch_device=str(device)))
    adapted.eval()
    return adapted, config


def load_edit_pipeline(adapter_dir, model_path: str = "tencent/Hunyuan3D-2.1", device="cuda"):
    """An fp16 pipeline on `device` whose DiT carries the adapter at `adapter_dir` as per-token-scalable layers.

    The adapter is never merged into the weights: a merged adapter ignores every per-token strength
    without raising, so the layer count is checked. `pipeline.adapter_config` keeps the adapter's config,
    against which `load_adapter_weights` checks a replacement.
    """
    from hy3dshape.models.per_sample_lora import PerSampleLoRALinear
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    adapter_dir = Path(adapter_dir)
    read_adapter_config(adapter_dir)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, device=device, dtype=torch.float16)
    pipeline.model, pipeline.adapter_config = attach_adapter(pipeline.model, adapter_dir, device)

    layers = [m for m in pipeline.model.modules() if isinstance(m, PerSampleLoRALinear)]
    if len(layers) != ADAPTED_LAYERS:
        raise ValueError(f"{adapter_dir} adapts {len(layers)} layers; the edit expects {ADAPTED_LAYERS} "
                         f"(fc1/fc2 of the block MLPs, net.0.proj/net.2 of the MoE experts)")
    for layer in layers:
        rate = getattr(layer.dropout, "p", 0.0)
        if rate != 0.0:
            raise ValueError(f"adapter dropout is {rate}; the edit needs 0")
    if pipeline.vae.surface_extractor is not None:
        raise ValueError(f"the VAE's surface extractor is pinned to "
                         f"{type(pipeline.vae.surface_extractor).__name__}; it must resolve per decode")
    return pipeline


def load_adapter_weights(pipeline, adapter_dir) -> None:
    """Overwrite, in place, the adapter weights of a pipeline from `load_edit_pipeline` with another adapter's.

    The replacement's config must equal the one the pipeline was built with in every setting outside
    NEUTRAL_CONFIG_KEYS (target_modules in any order), and its weights must name exactly the pipeline's adapted layers with the same
    shapes. Each weight is cast to its layer's dtype, as building the pipeline with this adapter would.
    """
    from hy3dshape.models.per_sample_lora import PerSampleLoRALinear
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    config, built = _layer_settings(read_adapter_config(adapter_dir)), _layer_settings(pipeline.adapter_config)
    differing = sorted(key for key in set(config) | set(built) if config.get(key) != built.get(key))
    if differing:
        raise ValueError(f"{adapter_dir} differs from the adapter the pipeline was built with in {differing}")
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"no adapter_model.safetensors in {adapter_dir}")
    stored = load_file(str(weights_path))

    targets = {}
    for name, module in pipeline.model.named_modules():
        if isinstance(module, PerSampleLoRALinear):
            targets[f"base_model.model.{name}.lora_A.weight"] = module.lora_A.weight
            targets[f"base_model.model.{name}.lora_B.weight"] = module.lora_B.weight
    missing, extra = sorted(set(targets) - set(stored)), sorted(set(stored) - set(targets))
    if missing or extra:
        raise ValueError(f"{weights_path} does not match the pipeline's adapted layers: {len(missing)} missing "
                         f"(first {missing[:3]}), {len(extra)} unexpected (first {extra[:3]})")
    for key, parameter in targets.items():
        if stored[key].shape != parameter.shape:
            raise ValueError(f"{key} is {tuple(stored[key].shape)} in {weights_path}, "
                             f"{tuple(parameter.shape)} in the pipeline")
    with torch.no_grad():
        for key, parameter in targets.items():
            parameter.copy_(stored[key])
