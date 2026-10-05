# Added by Itan Levin (2026); not part of Tencent's Hunyuan3D-2.1 release.
"""Capture of the DiT's image cross-attention weights.

`CrossAttention.forward` calls `compute_and_store_attention_weights` whenever the installed storage is
enabled. The weights handed to the storage are the fp16 softmax of q kᵀ times the attention scale, over
every head and batch element, one call per cross-attention layer in block order. A storage needs an
`enabled` attribute and a `store(weights)` method.
"""

from contextvars import ContextVar

import torch
import torch.nn.functional as F


class _NoStorage:
    enabled = False


_NO_STORAGE = _NoStorage()
_attention_storage_var: ContextVar = ContextVar('attention_storage', default=_NO_STORAGE)


def get_attention_storage():
    return _attention_storage_var.get()


def install_attention_storage(storage):
    """Install `storage` for the current context; returns the token `remove_attention_storage` needs."""
    return _attention_storage_var.set(storage)


def remove_attention_storage(token):
    _attention_storage_var.reset(token)


def compute_and_store_attention_weights(q: torch.Tensor, k: torch.Tensor, scale: float):
    """q [B, H, L, D] latent queries, k [B, H, T, D] image keys."""
    weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
    _attention_storage_var.get().store(weights)
