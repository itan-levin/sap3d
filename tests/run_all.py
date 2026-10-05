"""Run every test_* function of every tests/test_*.py file. CPU only; run from the repository root.

    python tests/run_all.py
"""

import importlib.util
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "hy3dshape")]


def main() -> int:
    failures, count = [], 0
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(n for n in vars(module) if n.startswith("test_") and callable(getattr(module, n))):
            count += 1
            started = time.time()
            try:
                getattr(module, name)()
            except Exception:
                failures.append(f"{path.stem}.{name}")
                print(f"FAIL  {path.stem}.{name}")
                traceback.print_exc()
                continue
            print(f"ok    {path.stem}.{name}  ({time.time() - started:.1f} s)")
    print(f"\n{count - len(failures)} of {count} tests passed")
    for name in failures:
        print(f"  failed: {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
