# Added by Itan Levin (2026); not part of Tencent's Hunyuan3D-2.1 release.
"""Adapter strength per batch element or per token for PEFT LoRA layers.

Each wrapped layer reads the strength from LoRAScaleContext on every forward:

    None             the adapter at its trained strength
    [B] tensor       one strength per batch element
    [B, L] tensor    one strength per token of an L-token sequence

MoE routed experts receive flattened token rows rather than [B, L] sequences, so the MoE block publishes
each row's batch index and flat token index through the two index contexts below while an expert runs.
"""

from contextvars import ContextVar
from typing import Optional

import torch
import torch.nn as nn

_lora_scale_var = ContextVar('lora_scale', default=None)


class LoRAScaleContext:
    @classmethod
    def set(cls, scales: Optional[torch.Tensor]):
        _lora_scale_var.set(scales)

    @classmethod
    def get(cls) -> Optional[torch.Tensor]:
        return _lora_scale_var.get()

    @classmethod
    def clear(cls):
        _lora_scale_var.set(None)


class LoRASampleIndexContext:
    """Batch index of each row an MoE routed expert is processing, values in [0, B)."""
    _current_indices: Optional[torch.Tensor] = None

    @classmethod
    def set(cls, indices: Optional[torch.Tensor]):
        cls._current_indices = indices

    @classmethod
    def get(cls) -> Optional[torch.Tensor]:
        return cls._current_indices

    @classmethod
    def clear(cls):
        cls._current_indices = None


class LoRATokenIndexContext:
    """Flat token index of each row an MoE routed expert is processing, values in [0, B * L)."""
    _current_indices: Optional[torch.Tensor] = None

    @classmethod
    def set(cls, indices: Optional[torch.Tensor]):
        cls._current_indices = indices

    @classmethod
    def get(cls) -> Optional[torch.Tensor]:
        return cls._current_indices

    @classmethod
    def clear(cls):
        cls._current_indices = None


class PerSampleLoRALinear(nn.Module):
    """A LoRA linear layer whose adapter term is multiplied by a strength read from LoRAScaleContext.

    The adapter term is computed and multiplied even at strength 0, so a zero strength yields exactly
    `base + lora * 0` rather than skipping the adapter.
    """

    def __init__(self, base_layer: nn.Linear, lora_A: nn.Linear, lora_B: nn.Linear,
                 base_scaling: float, dropout: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.base_scaling = base_scaling
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))

        scales = LoRAScaleContext.get()
        if scales is None:
            return result + lora_out * self.base_scaling

        sample_indices = LoRASampleIndexContext.get()
        if scales.dim() == 2:
            if sample_indices is not None:
                token_indices = LoRATokenIndexContext.get()
                if token_indices is None:
                    raise RuntimeError("an MoE expert published batch indices without token indices, "
                                       "so a per-token strength cannot be looked up")
                token_scales = scales.reshape(-1)[token_indices]
                if lora_out.dim() == 2:
                    token_scales = token_scales.unsqueeze(-1)
                return result + lora_out * token_scales * self.base_scaling
            if lora_out.dim() == 3 and lora_out.shape[1] == scales.shape[1]:
                return result + lora_out * scales.unsqueeze(-1) * self.base_scaling
            raise ValueError(f"a per-token strength of shape {tuple(scales.shape)} cannot scale an "
                             f"adapter output of shape {tuple(lora_out.shape)}")

        if sample_indices is not None:
            token_scales = scales[sample_indices]
            if lora_out.dim() == 2:
                token_scales = token_scales.view(-1, 1)
            return result + lora_out * token_scales * self.base_scaling
        return result + lora_out * scales.view(-1, *([1] * (lora_out.dim() - 1))) * self.base_scaling


def convert_to_per_sample_lora(peft_model) -> nn.Module:
    """Replace every PEFT LoRA layer of `peft_model` by a PerSampleLoRALinear; return the base model.

    Each wrapper is cast to its base layer's device and dtype, so an adapter stored in float32 runs in
    the model's precision.
    """
    from peft import PeftModel

    if not isinstance(peft_model, PeftModel):
        raise TypeError(f"expected a PeftModel, got {type(peft_model).__name__}")

    targets = [(name, module) for name, module in peft_model.named_modules()
               if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'base_layer')]
    for name, module in targets:
        adapters = list(module.lora_A.keys())
        if len(adapters) != 1:
            raise ValueError(f"{name} carries {len(adapters)} adapters; exactly one is supported")
        adapter = adapters[0]
        if module.use_dora[adapter]:
            raise ValueError(f"{name} is a DoRA layer, whose magnitude term this conversion would drop")
        wrapper = PerSampleLoRALinear(
            base_layer=module.base_layer,
            lora_A=module.lora_A[adapter],
            lora_B=module.lora_B[adapter],
            base_scaling=module.scaling[adapter],
            dropout=module.lora_dropout[adapter],
        ).to(device=module.base_layer.weight.device, dtype=module.base_layer.weight.dtype)
        parent = peft_model
        *path, leaf = name.split('.')
        for part in path:
            parent = getattr(parent, part)
        setattr(parent, leaf, wrapper)
    return peft_model.get_base_model()
