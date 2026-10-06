"""Download the RealESRGAN_x4plus.pth upscaler the paint model loads (the texture stage), from the Real-ESRGAN
v0.1.0 release, to third_party/hy3dpaint/ckpt/RealESRGAN_x4plus.pth.

    python scripts/download_realesrgan.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
TARGET = Path(__file__).resolve().parent.parent / "third_party" / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if TARGET.is_file() and sha256(TARGET) == SHA256:
        print(f"{TARGET} is already in place")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    partial = TARGET.with_suffix(".partial")
    urllib.request.urlretrieve(URL, partial)
    found = sha256(partial)
    if found != SHA256:
        partial.unlink()
        raise SystemExit(f"{URL} has sha256 {found}, expected {SHA256}")
    partial.replace(TARGET)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    sys.exit(main())
