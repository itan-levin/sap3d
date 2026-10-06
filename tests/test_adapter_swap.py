"""Replacing the adapter weights of a built pipeline equals building it with the replacement adapter."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from sap3d.edit.model import attach_adapter, load_adapter_weights

TARGETS = ["fc1", "fc2", "net.0.proj", "net.2"]


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 12)

    def forward(self, x):
        return torch.relu(self.proj(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(8, 16), nn.Linear(16, 8)
        self.net = nn.Sequential(_Projection(), nn.Identity(), nn.Linear(12, 8))

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))) + self.net(x)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _base():
    torch.manual_seed(0)
    return _Toy().half()


def _save_adapter(folder: Path, seed: int, alpha: int = 4) -> Path:
    from peft import LoraConfig, get_peft_model

    base = _base()
    torch.manual_seed(seed)
    peft_model = get_peft_model(base, LoraConfig(r=4, lora_alpha=alpha, target_modules=TARGETS,
                                                lora_dropout=0.0, init_lora_weights=False),
                                autocast_adapter_dtype=False)
    peft_model.save_pretrained(str(folder))
    return folder


def _pipeline(adapter_dir: Path):
    model, config = attach_adapter(_base(), adapter_dir, device="cpu")
    return SimpleNamespace(model=model, adapter_config=config)


def _states_equal(a, b) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]) for k in sa)


def test_swapped_weights_equal_a_fresh_build_bit_for_bit():
    with tempfile.TemporaryDirectory() as tmp:
        first, second = _save_adapter(Path(tmp) / "a", 1), _save_adapter(Path(tmp) / "b", 2)
        assert all(t.dtype == torch.float16 for t in load_file(str(second / "adapter_model.safetensors")).values())
        swapped, fresh = _pipeline(first), _pipeline(second)
        assert not _states_equal(swapped.model, fresh.model)
        load_adapter_weights(swapped, second)
        assert _states_equal(swapped.model, fresh.model)
        x = torch.randn(3, 8).half()
        with torch.no_grad():
            assert torch.equal(swapped.model(x), fresh.model(x))
        load_adapter_weights(swapped, first)
        assert _states_equal(swapped.model, _pipeline(first).model)


def test_mismatched_adapters_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        first, second = _save_adapter(Path(tmp) / "a", 1), _save_adapter(Path(tmp) / "b", 2)
        pipeline = _pipeline(first)
        before = {k: v.clone() for k, v in pipeline.model.state_dict().items()}

        other_alpha = _save_adapter(Path(tmp) / "alpha", 3, alpha=8)
        neutral = json.loads((second / "adapter_config.json").read_text())
        neutral.update(base_model_name_or_path="elsewhere", revision="r1", inference_mode=False,
                       target_modules=list(reversed(neutral["target_modules"])))
        (second / "adapter_config.json").write_text(json.dumps(neutral))
        load_adapter_weights(pipeline, second)

        weights = load_file(str(second / "adapter_model.safetensors"))
        dropped = Path(tmp) / "dropped"
        dropped.mkdir()
        (dropped / "adapter_config.json").write_text(json.dumps(neutral))
        save_file({k: v for k, v in weights.items() if k != sorted(weights)[0]},
                  str(dropped / "adapter_model.safetensors"))
        reshaped = Path(tmp) / "reshaped"
        reshaped.mkdir()
        (reshaped / "adapter_config.json").write_text(json.dumps(neutral))
        key = sorted(weights)[0]
        save_file({**weights, key: weights[key][:, :-1].contiguous()}, str(reshaped / "adapter_model.safetensors"))

        load_adapter_weights(pipeline, first)
        for folder, message in ((other_alpha, "differs"), (dropped, "missing"), (reshaped, "in the pipeline")):
            try:
                load_adapter_weights(pipeline, folder)
            except ValueError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError(f"{folder.name} was accepted")
        after = pipeline.model.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
