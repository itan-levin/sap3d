"""sap3d-bench: edit, texture and evaluate a list of Edit3D-Bench cases, resumably.

    sap3d-bench --bench-root DATA --adapters ADAPTERS --out OUT --cases cases.txt
    sap3d-bench --bench-root DATA --adapters ADAPTERS --out OUT --case GSO/SORTING_BUS/prompt_2 ...

One invocation covers one benchmark root and one edit configuration. Local edits are textured in composite
mode with the color fit and scored with preservation outside the region, edit fidelity and the benchmark's
six metrics. `--global` marks whole-object edits (a whole-object region from the case folder, or
`--config configs/edit_global_uniform.yaml`): generated texture, whole-surface preservation, edit fidelity,
no benchmark metrics.

The stages run one after another over all cases: edits in this process (the shape model is reloaded only
when the adapter changes), then texturing in one child process (the paint model loads once), then
evaluation in another. Each case writes `OUT/cases/<case id>/{edit,texture,eval}/` and `case.json`. A
finished stage is skipped when its recorded settings and input digests match, redone when only its inputs
changed (an upstream stage was redone), and refused when its settings differ; `--overwrite` redoes every
stage of every listed case. A failing case stops the run; `--skip-cases` leaves cases out.
`OUT/summary.{json,csv,md}` cover every finished case under OUT, with means taken separately for local and
global edits; keep one OUT per configuration.
"""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import bench, third_party
from .. import config as config_module
from .. import region as regions
from ..texture import cli as texture_cli

SUMMARY_COLUMNS = (
    ("faces", "faces of the edited mesh", None),
    ("final_region_tokens", "region tokens at the last probe step", None),
    ("preservation_composite", "preservation, composite (outside region) ↓", "lower"),
    ("preservation_generated", "preservation, generated (whole surface) ↓", "lower"),
    ("preservation_raw", "preservation, raw geometry ↓", "lower"),
    ("dino_uncorrected", "DINOv2 edit distance, uncorrected composite ↓", "lower"),
    ("dino_color_fitted", "DINOv2 edit distance, color-fitted ↓", "lower"),
    ("dino_generated", "DINOv2 edit distance, generated ↓", "lower"),
    ("clip_uncorrected", "CLIP edit distance, uncorrected composite ↓", "lower"),
    ("clip_color_fitted", "CLIP edit distance, color-fitted ↓", "lower"),
    ("clip_generated", "CLIP edit distance, generated ↓", "lower"),
    ("bench_chamfer", "benchmark Chamfer ↓", "lower"),
    ("bench_psnr", "benchmark PSNR, dB ↑", "higher"),
    ("bench_ssim", "benchmark SSIM ↑", "higher"),
    ("bench_lpips", "benchmark LPIPS ↓", "lower"),
    ("bench_dino_i", "benchmark DINO-I ↑", "higher"),
    ("bench_clip_t", "benchmark CLIP-T ↑", "higher"),
)


def _digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cases(args) -> list:
    if args.cases is not None:
        ids = [line.split("#", 1)[0].strip() for line in args.cases.read_text(encoding="utf-8").splitlines()]
        ids = [case for case in ids if case]
    else:
        ids = list(args.case or [])
    if not ids:
        raise SystemExit("no cases given (--cases FILE or --case ID ...)")
    if len(set(ids)) != len(ids):
        raise SystemExit("a case is listed twice")
    skip = list(args.skip_cases or [])
    unknown = sorted(set(skip) - set(ids))
    if unknown:
        raise SystemExit(f"--skip-cases names cases that are not listed: {unknown}")
    return [case for case in ids if case not in skip]


def case_dir(out: Path, case: bench.BenchCase) -> Path:
    return out / "cases" / case.dataset / case.object_name / case.prompt_key


def _must_run(stage: str, case_id: str, recorded, wanted: dict, overwrite: bool) -> bool:
    """Whether a finished stage reruns: always with `overwrite`, otherwise when its inputs changed.

    A finished stage made with other settings is refused rather than silently redone or kept.
    """
    if overwrite:
        return True
    if not isinstance(recorded, dict) or set(recorded) != {"settings", "inputs"}:
        raise SystemExit(f"{case_id}: the finished {stage} stage carries no record of how it was made; pass "
                         f"--overwrite to redo it")
    if recorded["settings"] != wanted["settings"]:
        differing = sorted(key for key in set(recorded["settings"]) | set(wanted["settings"])
                           if recorded["settings"].get(key) != wanted["settings"].get(key))
        raise SystemExit(f"{case_id}: the finished {stage} stage was made with different settings ({differing}); "
                         f"pass --overwrite to redo it")
    return recorded["inputs"] != wanted["inputs"]


def edit_provenance(case, cfg, adapter: Path, model: str) -> dict:
    inputs = {"source": _digest(case.source_glb), "edited_view": _digest(case.edited_view),
              "adapter_weights": _digest(adapter / "adapter_model.safetensors"),
              "adapter_config": _digest(adapter / "adapter_config.json")}
    if cfg.mode == "local":
        inputs.update(region=_digest(case.region_glb), mask=_digest(case.mask))
    return {"settings": {"edit": config_module.to_dict(cfg), "model": model}, "inputs": inputs}


def run_edits(cases, out, cfg, adapters, model, overwrite) -> None:
    pending = []
    for case in cases:
        folder = case_dir(out, case) / "edit"
        adapter = bench.adapter_dir(adapters, case)
        wanted = edit_provenance(case, cfg, adapter, model)
        record_path = folder / "edit.json"
        if record_path.is_file() and not _must_run(
                "edit", case.case_id, json.loads(record_path.read_text(encoding="utf-8")).get("provenance"),
                wanted, overwrite):
            print(f"edit     skip  {case.case_id}", flush=True)
            continue
        pending.append((case, adapter, wanted))
    if not pending:
        return

    import torch

    from ..edit.edit import clear_outputs, run_edit, save_outputs
    from ..edit.model import load_edit_pipeline

    pipeline, loaded = None, None
    for position, (case, adapter, wanted) in enumerate(pending, start=1):
        if adapter != loaded:
            pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            pipeline, loaded = load_edit_pipeline(adapter, model), adapter
        folder = case_dir(out, case) / "edit"
        clear_outputs(folder)
        region = regions.load_region(case.region_glb) if cfg.mode == "local" else None
        mask = case.mask if cfg.mode == "local" else None
        mesh, record, steps = run_edit(pipeline, cfg, case.source_glb, case.edited_view, region, mask)
        record["provenance"] = wanted
        save_outputs(folder, mesh, record, steps, cfg, source=case.source_glb, edited_view=case.edited_view,
                     adapter=adapter)
        print(f"edit     [{position}/{len(pending)}] {case.case_id}  faces={record['faces']}  "
              f"({record['seconds']} s)", flush=True)
    pipeline = None
    gc.collect()
    torch.cuda.empty_cache()


def run_textures(cases, out, tcfg, texture_config, global_edit, model, overwrite) -> None:
    from ..texture.texture import FITTED_NAME, job_inputs, settings

    pending = []
    for case in cases:
        folder = case_dir(out, case)
        job = texture_cli.bench_job(case, folder / "edit" / "edit.glb", folder / "texture",
                                    "generated" if global_edit else "composite", tcfg.color_fit)
        wanted = {"inputs": job_inputs(job), "settings": settings(tcfg, model)}
        record_path = folder / "texture" / "texture.json"
        finished = record_path.is_file() and (not job["color_fit"] or (
            (folder / "texture" / "color_fit" / "fit.json").is_file() and (folder / "texture" / FITTED_NAME).is_file()))
        if finished:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if not _must_run("texture", case.case_id,
                             {"inputs": record["inputs"], "settings": record["settings"]}, wanted, overwrite):
                print(f"texture  skip  {case.case_id}", flush=True)
                continue
        pending.append(job)
    if not pending:
        return
    jobs_path = out / "work" / "texture_jobs.json"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps(pending, indent=1), encoding="utf-8")
    command = ["--jobs", str(jobs_path.resolve()), "--model", model]
    if texture_config is not None:
        command += ["--config", str(Path(texture_config).resolve())]
    _run_child("sap3d.texture.cli", command)


def run_evals(cases, out, bench_root, fidelity_device, overwrite) -> None:
    from .evaluate import _plan

    pending = []
    for case in cases:
        folder = case_dir(out, case)
        job = {"texture_dir": str((folder / "texture").resolve()), "out_dir": str((folder / "eval").resolve()),
               "fidelity_device": fidelity_device,
               "bench": {"root": str(Path(bench_root).resolve()), "case": case.case_id}}
        record_path = folder / "eval" / "eval.json"
        if record_path.is_file():
            plan = _plan(job)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if not _must_run("eval", case.case_id, {"inputs": record["inputs"], "settings": record["settings"]},
                             {"inputs": plan["inputs"], "settings": plan["settings"]}, overwrite):
                print(f"eval     skip  {case.case_id}", flush=True)
                continue
        pending.append(job)
    if not pending:
        return
    jobs_path = out / "work" / "eval_jobs.json"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps(pending, indent=1), encoding="utf-8")
    _run_child("sap3d.eval.cli", ["--jobs", str(jobs_path.resolve())])


def _run_child(module: str, arguments: list) -> None:
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", module, *arguments], env=third_party.child_env(), check=True)


def case_row(folder: Path, case_id: str) -> dict:
    """One case's headline numbers from its stage records."""
    edit = json.loads((folder / "edit" / "edit.json").read_text(encoding="utf-8"))
    texture = json.loads((folder / "texture" / "texture.json").read_text(encoding="utf-8"))
    evaluation = json.loads((folder / "eval" / "eval.json").read_text(encoding="utf-8"))
    row = {"case": case_id, "mode": evaluation["mode"], "faces": edit["faces"],
           "final_region_tokens": edit.get("final_region_tokens"),
           "output_turn_degrees": edit.get("output_turn_degrees"),
           "edit_seconds": edit["seconds"], "texture_seconds": texture["seconds"]}
    for name, record in evaluation["preservation"].items():
        row[f"preservation_{name}"] = record["symmetric"]
    names = {"composite": "uncorrected", "color_fitted": "color_fitted", "generated": "generated"}
    for name, record in (evaluation.get("fidelity") or {}).items():
        row[f"dino_{names[name]}"] = record["dino_embedding_distance"]
        row[f"clip_{names[name]}"] = record["clip_embedding_distance"]
    for metric, value in (evaluation.get("benchmark") or {}).items():
        if isinstance(value, dict) and "mean" in value:
            row[f"bench_{metric}"] = value["mean"]
    fit_path = folder / "texture" / "color_fit" / "fit.json"
    if fit_path.is_file():
        row["color_fit_round"] = json.loads(fit_path.read_text(encoding="utf-8"))["published_iteration"]
    return row


def _bench_scores(folder: Path) -> dict:
    evaluation = json.loads((folder / "eval" / "eval.json").read_text(encoding="utf-8"))
    return {metric: value["all_scores"] for metric, value in (evaluation.get("benchmark") or {}).items()
            if isinstance(value, dict) and "all_scores" in value}


def write_summary(out: Path) -> dict:
    """Per-case rows and, separately for local and global edits, the mean of every headline number.

    A benchmark metric's mean pools its scores over all views of all cases, as the scorer does when it scores
    many cases at once; every other mean is over cases.
    """
    rows, scores = [], {}
    for record in sorted((out / "cases").glob("*/*/*/eval/eval.json")):
        folder = record.parent.parent
        case_id = "/".join(folder.relative_to(out / "cases").parts)
        row = case_row(folder, case_id)
        (folder / "case.json").write_text(json.dumps(row, indent=1, sort_keys=True), encoding="utf-8")
        rows.append(row)
        scores[case_id] = _bench_scores(folder)
    if not rows:
        raise SystemExit(f"no finished case under {out / 'cases'}")
    means = {}
    for mode in sorted({row["mode"] for row in rows}):
        group = [row for row in rows if row["mode"] == mode]
        entries = {}
        for key, label, direction in SUMMARY_COLUMNS:
            if key.startswith("bench_"):
                pooled = [value for row in group for value in scores[row["case"]].get(key[len("bench_"):], [])]
                cases = sum(1 for row in group if row.get(key) is not None)
                if pooled:
                    entries[key] = {"label": label, "mean": float(np.mean(pooled)), "std": float(np.std(pooled)),
                                    "count": cases, "of": len(group), "scores": len(pooled), "better": direction}
                continue
            values = [row[key] for row in group if row.get(key) is not None]
            if values:
                entries[key] = {"label": label, "mean": float(np.mean(values)), "std": float(np.std(values)),
                                "count": len(values), "of": len(group), "better": direction}
        means[mode] = entries
    summary = {"cases": {row["case"]: row for row in rows}, "means": means}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")

    columns = ["case", "mode"] + sorted({key for row in rows for key in row} - {"case", "mode"})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    titles = {"composite": "local edits (composite texture)", "generated": "global edits (generated texture)"}
    lines = [f"# Edit3D-Bench results: {len(rows)} finished cases", ""]
    for mode, entries in means.items():
        lines += [f"## {titles[mode]}", "", "| metric | mean | std | cases with a value |", "|---|---|---|---|"]
        for entry in entries.values():
            spread = "std over all views" if "scores" in entry else "std over cases"
            lines.append(f"| {entry['label']} | {entry['mean']:.6g} | {entry['std']:.6g} ({spread}) | "
                         f"{entry['count']} of {entry['of']} |")
        lines.append("")
    lines += ["Preservation is the exact point-to-surface distance to the source in the source's units (objects "
              "span about 1): outside the edit region for local edits, over the whole surface for global ones. Edit "
              "distances compare a render through the edited view's camera with the edited view inside the 2D "
              "mask. Benchmark metrics are the Edit3D-Bench scorer's, on the composite before the color fit, "
              "pooled over all views of all cases.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sap3d-bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-root", type=Path, required=True, help="Edit3D-Bench data folder")
    parser.add_argument("--adapters", type=Path, required=True, help="folder of adapters with manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--cases", type=Path, help="text file with one case id per line")
    parser.add_argument("--case", nargs="+", help="case ids")
    parser.add_argument("--skip-cases", nargs="+", help="listed case ids to leave out")
    parser.add_argument("--config", type=Path, help="edit configuration YAML")
    parser.add_argument("--texture-config", type=Path, help="texture configuration YAML")
    parser.add_argument("--global", dest="global_edit", action="store_true",
                        help="whole-object edits: generated texture, whole-surface preservation, no "
                             "benchmark metrics")
    parser.add_argument("--fidelity-device", help="device of the fidelity encoders (default: cpu for local, "
                                                  "cuda:0 for global edits)")
    parser.add_argument("--model", default="tencent/Hunyuan3D-2.1", help="shape and paint model repository")
    parser.add_argument("--overwrite", action="store_true", help="redo every stage of every listed case")
    args = parser.parse_args(argv)

    cfg = config_module.load_config(args.config)
    tcfg = config_module.load_texture_config(args.texture_config)
    if cfg.mode == "uniform" and not args.global_edit:
        raise SystemExit("a uniform-strength edit is a whole-object edit; pass --global")
    cases = [bench.parse_case(args.bench_root, case_id) for case_id in read_cases(args)]
    for case in cases:
        bench.adapter_dir(args.adapters, case)
    args.out.mkdir(parents=True, exist_ok=True)

    run_edits(cases, args.out, cfg, args.adapters, args.model, args.overwrite)
    run_textures(cases, args.out, tcfg, args.texture_config, args.global_edit, args.model, args.overwrite)
    run_evals(cases, args.out, args.bench_root, args.fidelity_device, args.overwrite)
    summary = write_summary(args.out)
    print(f"summary over {len(summary['cases'])} cases: {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
