"""Evaluate textured edits; renders run in one child process, the encoders load once.

An eval job names `texture_dir` (a texture stage's output folder, whose `texture.json` names every input),
`out_dir`, `fidelity_device` (None for the default of the texture mode) and `bench` (`{"root", "case"}` of
an Edit3D-Bench case, or None).

- Composite mode (local edits): preservation outside the region on the composite and on the raw edited
  mesh; edit fidelity on the composite and on the color-fitted mesh; for a benchmark case, the scorer's six
  metrics on the composite.
- Generated mode (global edits): preservation over the whole surface on the generated mesh and the raw
  mesh; edit fidelity on the generated mesh; no benchmark metrics.

Edit fidelity is scored on the CPU for composite mode and on cuda:0 for generated mode by default. When the
color fit kept the uncorrected atlas, the color-fitted mesh is the composite and takes the composite's
fidelity record.
"""

import hashlib
import json
import time
from pathlib import Path

from .. import bench, third_party
from ..render import frame_name
from . import fidelity as fidelity_module
from . import preservation as preservation_module
from . import scorer

DEFAULT_FIDELITY_DEVICE = {"composite": "cpu", "generated": "cuda:0"}
FITTED_NAME = "composite_color_fit.glb"


def _digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plan(job: dict) -> dict:
    """Everything one eval job reads, resolved from its texture record."""
    texture_dir = Path(job["texture_dir"]).resolve()
    record = json.loads((texture_dir / "texture.json").read_text(encoding="utf-8"))
    tjob = record["job"]
    mode = tjob["mode"]
    meshes = {"raw": Path(tjob["mesh"])}
    renders = {}
    if mode == "composite":
        meshes["composite"] = texture_dir / "composite.glb"
        renders["composite"] = meshes["composite"]
        if tjob["color_fit"]:
            fit = json.loads((texture_dir / "color_fit" / "fit.json").read_text(encoding="utf-8"))
            meshes["color_fitted"] = texture_dir / FITTED_NAME
            if fit["published_iteration"] != 0:
                renders["color_fitted"] = meshes["color_fitted"]
    else:
        meshes["generated"] = texture_dir / "generated.glb"
        renders["generated"] = meshes["generated"]
    for path in meshes.values():
        if not path.is_file():
            raise FileNotFoundError(f"no mesh at {path}")
    if not tjob.get("view"):
        renders = {}
    out_dir = Path(job["out_dir"]).resolve()
    case = None
    if job.get("bench") and mode == "composite":
        case = bench.parse_case(job["bench"]["root"], job["bench"]["case"])
        if case.source_glb.resolve() != Path(tjob["source"]).resolve():
            raise ValueError(f"benchmark case {case.case_id} has source {case.source_glb}, but the texture was made "
                             f"from {tjob['source']}")
    device = job.get("fidelity_device") or DEFAULT_FIDELITY_DEVICE[mode]
    inputs = {name: _digest(path) for name, path in meshes.items()}
    inputs.update({key: _digest(tjob[key]) for key in ("source", "region", "edited_view") if tjob.get(key)})
    if tjob.get("view"):
        inputs["mask"] = _digest(tjob["view"]["mask"])
        inputs["camera"] = tjob["view"]["camera"]
    return {"job": job, "tjob": tjob, "mode": mode, "meshes": meshes, "renders": renders, "out_dir": out_dir,
            "case": case, "device": device, "inputs": inputs,
            "settings": {"fidelity_device": device, "benchmark": case.case_id if case else None,
                         "scorer_metrics": list(scorer.METRICS), "scorer_image_size": scorer.IMAGE_SIZE}}


def run(jobs: list, work_dir: Path) -> list:
    """Evaluate every job and write each `eval.json`; returns the records."""
    plans = [_plan(job) for job in jobs]
    started = time.time()
    render_jobs = []
    for plan in plans:
        out_dir = plan["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval.json").unlink(missing_ok=True)
        for name, glb in plan["renders"].items():
            render_jobs.append({"glb": str(glb.resolve()), "out_dir": str(out_dir / "renders" / name),
                                "camera": plan["tjob"]["view"]["camera"]})
        if plan["case"] is not None:
            images = scorer.stage_tree(out_dir / "bench_tree", plan["case"], plan["meshes"]["composite"]) / "images"
            render_jobs.append({"glb": str(images.parent / "edit.glb"), "out_dir": str(images), "camera": "all"})

    records = []
    for plan in plans:
        tjob = plan["tjob"]
        region = tjob["region"] if plan["mode"] == "composite" else None
        rows = {name: path for name, path in plan["meshes"].items() if name != "color_fitted"}
        records.append({"mode": plan["mode"], "preservation": preservation_module.preservation(
            tjob["source"], rows, region)})

    if render_jobs:
        third_party.run_module("sap3d.render", render_jobs, Path(work_dir) / "render_jobs.json")

    scorers = {}
    for plan, record in zip(plans, records):
        view = plan["tjob"].get("view")
        if not view:
            record["fidelity"] = None
            continue
        device = plan["device"]
        scorers.setdefault(device, fidelity_module.FidelityScorer(device))
        record["fidelity"] = {}
        for name in plan["renders"]:
            png = plan["out_dir"] / "renders" / name / frame_name(view["camera"])
            record["fidelity"][name] = scorers[device].score(png, plan["tjob"]["edited_view"], view["mask"])
            record["fidelity"][name]["render"] = str(png)
        if "color_fitted" in plan["meshes"] and "color_fitted" not in plan["renders"]:
            record["fidelity"]["color_fitted"] = dict(record["fidelity"]["composite"],
                                                      note="the color fit kept the uncorrected atlas")

    for plan, record in zip(plans, records):
        if plan["case"] is not None:
            record["benchmark"] = scorer.score(plan["case"].root, plan["out_dir"] / "bench_tree",
                                               plan["out_dir"] / "scorer")
        else:
            record["benchmark"] = None
        record.update(inputs=plan["inputs"], settings=plan["settings"],
                      texture_dir=str(Path(plan["job"]["texture_dir"]).resolve()))
        (plan["out_dir"] / "eval.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    print(f"evaluated {len(plans)} job(s) in {time.time() - started:.0f} s", flush=True)
    return records
