"""Per-sample and per-token adapter strength, and the step schedule of a local edit."""

import torch
import torch.nn as nn

from hy3dshape.models.per_sample_lora import (LoRAScaleContext, LoRASampleIndexContext, LoRATokenIndexContext,
                                              PerSampleLoRALinear)
from sap3d.edit.sampler import build_latent_mask, step_regime


def _layer(seed=0):
    torch.manual_seed(seed)
    return PerSampleLoRALinear(nn.Linear(6, 5), nn.Linear(6, 2, bias=False), nn.Linear(2, 5, bias=False),
                               base_scaling=1.0, dropout=nn.Identity()).eval()


def _parts(layer, x):
    return layer.base_layer(x), layer.lora_B(layer.lora_A(x))


def _run(layer, x, scales, sample_indices=None, token_indices=None):
    LoRAScaleContext.set(scales)
    LoRASampleIndexContext.set(sample_indices)
    LoRATokenIndexContext.set(token_indices)
    try:
        return layer(x)
    finally:
        LoRAScaleContext.clear()
        LoRASampleIndexContext.clear()
        LoRATokenIndexContext.clear()


def test_no_context_is_full_strength():
    layer, x = _layer(), torch.randn(2, 7, 6)
    base, lora = _parts(layer, x)
    assert torch.equal(layer(x), base + lora)


def test_per_sample_and_per_token():
    layer, x = _layer(), torch.randn(2, 7, 6)
    base, lora = _parts(layer, x)
    out = _run(layer, x, torch.tensor([0.25, 2.0]))
    assert torch.allclose(out, base + lora * torch.tensor([0.25, 2.0]).view(2, 1, 1))
    tokens = torch.rand(2, 7)
    out = _run(layer, x, tokens)
    assert torch.allclose(out, base + lora * tokens.unsqueeze(-1))


def test_equal_strengths_give_identical_bits_on_both_paths():
    layer, x = _layer(), torch.randn(2, 7, 6).half()
    layer = layer.half()
    per_sample = _run(layer, x, torch.tensor([0.5, 0.5], dtype=torch.float16))
    per_token = _run(layer, x, torch.full((2, 7), 0.5, dtype=torch.float16))
    assert torch.equal(per_sample, per_token)


def test_zero_strength_still_evaluates_the_adapter_term():
    layer, x = _layer(), torch.randn(2, 7, 6)
    with torch.no_grad():
        layer.lora_B.weight.fill_(float("inf"))
    assert torch.isnan(_run(layer, x, torch.zeros(2))).all()
    assert torch.isnan(_run(layer, x, torch.zeros(2, 7))).all()


def test_moe_rows_read_their_own_strengths():
    layer = _layer()
    rows = torch.randn(5, 6)
    base, lora = _parts(layer, rows)
    flat_tokens = torch.tensor([0, 3, 7, 8, 13])
    batch = flat_tokens // 7
    tokens = torch.rand(2, 7)
    out = _run(layer, rows, tokens, batch, flat_tokens)
    assert torch.allclose(out, base + lora * tokens.reshape(-1)[flat_tokens].unsqueeze(-1))
    out = _run(layer, rows, torch.tensor([0.1, 0.9]), batch, flat_tokens)
    assert torch.allclose(out, base + lora * torch.tensor([0.1, 0.9])[batch].view(-1, 1))


def test_mismatches_raise():
    layer = _layer()
    for args in ((torch.randn(2, 7, 6), torch.rand(2, 9)),
                 (torch.randn(5, 6), torch.rand(2, 7), torch.zeros(5, dtype=torch.long), None)):
        try:
            _run(layer, *args)
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"no error for {[tuple(a.shape) if a is not None else None for a in args]}")


def test_schedule():
    regimes = [step_regime(step, 8, 16) for step in range(30)]
    assert regimes == ["pre_mask"] * 8 + ["per_token"] * 8 + ["dropped"] * 14


def test_latent_mask():
    mask = build_latent_mask({0, 5, 4095}, 1.0, 0.0, "cpu", torch.float16)
    assert mask.shape == (1, 4097) and mask.dtype == torch.float16
    assert mask[0, 0] == 1.0 and mask[0, 1] == 0.0 and mask[0, 6] == 0.0 and mask[0, 4096] == 0.0
    assert int((mask == 0).sum()) == 3
    assert torch.equal(build_latent_mask(set(), 1.0, 0.0, "cpu", torch.float16), torch.ones(1, 4097).half())
