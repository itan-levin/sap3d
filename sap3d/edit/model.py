"""The shape pipeline with a per-object adapter whose strength can be set per token."""

from pathlib import Path

import torch

ADAPTED_LAYERS = 138


def load_edit_pipeline(adapter_dir, model_path: str = "tencent/Hunyuan3D-2.1"):
    """An fp16 CUDA pipeline whose DiT carries the adapter at `adapter_dir` as per-token-scalable layers.

    The adapter is never merged into the weights: a merged adapter ignores every per-token strength
    without raising, so the layer count is checked.
    """
    from hy3dshape.models.per_sample_lora import PerSampleLoRALinear, convert_to_per_sample_lora
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"no adapter_config.json in {adapter_dir}")

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, device="cuda", dtype=torch.float16)
    pipeline.model = convert_to_per_sample_lora(PeftModel.from_pretrained(pipeline.model, str(adapter_dir)))
    pipeline.model.eval()

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
