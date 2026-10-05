"""Locations of the vendored hy3dpaint package and the Edit3D-Bench scorer.

Both are used from the source tree by path rather than installed: hy3dpaint and the scorer each ship a
top-level `utils` package, and the scorer's renderer reads its environment map relative to its own
directory. The texture and eval stages therefore need a source checkout (or an editable install).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "third_party"
HY3DPAINT_DIR = ROOT / "hy3dpaint"
SCORER_DIR = ROOT / "edit3d_bench"


def require(directory: Path, marker: str) -> Path:
    if not (directory / marker).is_file():
        raise FileNotFoundError(f"{directory / marker} is missing; the texture and eval stages run from a "
                                f"source checkout of this repository (or `pip install -e .`)")
    return directory


def child_env() -> dict:
    """This process's environment with the repository importable and LD_LIBRARY_PATH emptied.

    A non-empty LD_LIBRARY_PATH lets Blender's bundled libraries and the environment's CUDA runtime clash.
    """
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ""
    paths = [str(REPO_ROOT), str(ROOT / "hy3dshape")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_module(module: str, jobs: list, jobs_path: Path) -> None:
    """Run `python -m module jobs_path` in a child process after writing `jobs` there; raise on failure."""
    jobs_path = Path(jobs_path)
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    subprocess.run([sys.executable, "-m", module, str(jobs_path.resolve())], env=child_env(), check=True)
