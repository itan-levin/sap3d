"""The adapter of a training run: the saved step with the lowest validation Chamfer distance.

Ties go to the earlier step. The untrained model's score (step 0) is never chosen; the record states whether
it was lower than the chosen step's.
"""

import json
import shutil
from pathlib import Path

KEYS = {"corrected": "chamfer_distance", "legacy": "legacy_chamfer_distance"}


def choose(validations: list, metric: str) -> dict:
    if metric not in KEYS:
        raise ValueError(f"the selection metric is one of {list(KEYS)}, got {metric!r}")
    key = KEYS[metric]
    trained = [entry for entry in validations if entry["step"] > 0]
    if not trained:
        raise ValueError("no validated checkpoint to choose from")
    return min(trained, key=lambda entry: (entry[key], entry["step"]))


def write_record(path: Path, record: dict) -> None:
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(record, indent=1), encoding="utf-8")
    partial.replace(path)


def select(run_dir, metric=None) -> dict:
    """Copy the chosen step of `run_dir` to `run_dir/adapter` and record the choice in its training.json."""
    run_dir = Path(run_dir)
    path = run_dir / "training.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    metric = metric or record["settings"]["validation"]["selection_metric"]
    best = choose(record["validations"], metric)
    step_dir = run_dir / "steps" / f"step_{best['step']}"
    if not (step_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"the chosen step's adapter is missing from {step_dir}")
    target = run_dir / "adapter"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(step_dir, target)
    base = [entry[KEYS[metric]] for entry in record["validations"] if entry["step"] == 0]
    record["selected"] = {"step": best["step"], "metric": metric, "value": best[KEYS[metric]],
                          "untrained_scored_lower": bool(base) and base[0] <= best[KEYS[metric]]}
    write_record(path, record)
    return record["selected"]
