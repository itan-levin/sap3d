"""The Edit3D-Bench scorer, run unmodified in its own process on a one-case prediction tree.

The tree is `<tree>/<dataset>/<object>/<prompt_key>/edit.glb` with the sixteen benchmark views rendered
into `images/`; the scorer pairs them with the benchmark's own renders, masks and source mesh. With one
case in the tree, each metric's mean is that case's value.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from .. import third_party

METRICS = ("chamfer", "psnr", "ssim", "lpips", "dino_i", "clip_t")
IMAGE_SIZE = 512
DEVICE = "cuda:0"

# Scores each metric contributes per case: one per view, CLIP-T on its six near-frontal views, Chamfer once.
SCORES_PER_CASE = {"psnr": 16, "ssim": 16, "lpips": 16, "dino_i": 16, "clip_t": 6, "chamfer": 1}

# The scorer drops a view whose error inside the scored region is exactly zero, which happens when the
# region covers the whole silhouette from that camera; only PSNR has that second reason to drop a view.
ZERO_ERROR_METRIC = "psnr"


def stage_tree(tree: Path, case, mesh: Path) -> Path:
    """Copy `mesh` into a fresh one-case tree and return the folder its sixteen views are rendered into."""
    if tree.exists():
        shutil.rmtree(tree)
    folder = tree / case.dataset / case.object_name / case.prompt_key
    folder.mkdir(parents=True)
    shutil.copyfile(mesh, folder / "edit.glb")
    return folder


def check_counts(results: dict) -> dict:
    """Raise unless every metric has its expected count; returns the PSNR views dropped for zero error."""
    problems, shortfall = [], {}
    for metric in METRICS:
        value = results.get(metric)
        if not isinstance(value, dict) or value.get("mean") is None:
            problems.append(f"{metric}: no value produced")
            continue
        expected, actual = SCORES_PER_CASE[metric], value.get("count", 0)
        if actual == expected:
            continue
        if metric == ZERO_ERROR_METRIC and actual < expected:
            shortfall[metric] = expected - actual
            continue
        problems.append(f"{metric}: {actual} scores, expected {expected}")
    if problems:
        if shortfall:
            problems.append(f"{ZERO_ERROR_METRIC} is short as well, so views were lost rather than dropped for "
                            f"zero error")
        raise RuntimeError("the scorer produced incomplete results:\n  " + "\n  ".join(problems))
    return shortfall


def score(bench_root: Path, tree: Path, out_dir: Path) -> dict:
    """Run the scorer on `tree` against `bench_root`; returns {metric: {mean, std, count, all_scores}}."""
    scorer_dir = third_party.require(third_party.SCORER_DIR, "eval_main.py")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    env = third_party.child_env()
    env.pop("PYTHONPATH")
    command = [sys.executable, "eval_main.py", "--gt_root", str(Path(bench_root).resolve()),
               "--pred_root", str(tree.resolve()), "--metrics", *METRICS, "--device", DEVICE,
               "--image_size", str(IMAGE_SIZE), str(IMAGE_SIZE), "--batch_size", "32", "--num_workers", "4",
               "--output_dir", str(out_dir.resolve())]
    subprocess.run(command, cwd=scorer_dir, env=env, check=True)
    results = json.loads((out_dir / "detailed_results.json").read_text(encoding="utf-8"))
    dropped = check_counts(results)
    values = {metric: results[metric] for metric in METRICS}
    values["views_dropped_for_zero_error"] = dropped
    return values
