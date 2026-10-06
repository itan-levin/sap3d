"""Training and validation samples of one dataset folder (see `dataset`).

A sample is one render, chosen by the stream's own `random.Random`, composited on white, cropped to its
alpha box with a margin, padded to a square, resized and normalized, together with a fresh subsample of the
stored surface points. The subsample is drawn from operating-system entropy on every call.
"""

import glob
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from ..config import TrainConfig

MEAN = STD = (0.5, 0.5, 0.5)


def render_paths(dataset_dir) -> list:
    paths = sorted(glob.glob(str(Path(dataset_dir) / "renders" / "[0-9][0-9][0-9].png")))
    if not paths:
        raise FileNotFoundError(f"no renders/NNN.png in {dataset_dir}")
    return paths


def load_points(dataset_dir) -> np.ndarray:
    points = np.load(Path(dataset_dir) / "surface.npz")["points"]
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError(f"surface.npz holds points of shape {points.shape}; expected (N, 6)")
    return points


def pad_square(image: np.ndarray, mask: np.ndarray, ratio: float):
    """Center the crop on a white (image) and black (mask) square of side int(longest side x ratio)."""
    h, w = image.shape[:2]
    side = int(max(h, w) * ratio)
    top, left = (side - h) // 2, (side - w) // 2
    canvas = np.ones((side, side, 3), dtype=np.uint8) * 255
    canvas_mask = np.zeros((side, side), dtype=np.uint8)
    canvas[top:top + h, left:left + w] = image
    canvas_mask[top:top + h, left:left + w] = mask
    return canvas, canvas_mask


class SampleStream:
    def __init__(self, dataset_dir, cfg: TrainConfig, seed: int = 0):
        self.paths = render_paths(dataset_dir)
        self.points = load_points(dataset_dir)
        if len(self.points) < cfg.points_per_sample:
            raise ValueError(f"{dataset_dir} stores {len(self.points)} surface points; each sample takes "
                             f"{cfg.points_per_sample}")
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(cfg.image_size),
                                             transforms.Normalize(mean=MEAN, std=STD)])

    def load_render(self, path):
        """(image (3, S, S), mask (1, S, S)) of one RGBA render."""
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"{path} is not an RGBA image")
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        foreground = image[:, :, :3]
        image = (foreground * alpha + np.ones_like(foreground) * 255 * (1 - alpha)).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (alpha[:, :, 0] * 255).astype(np.uint8)
        h, w = image.shape[:2]
        rows_cols = np.argwhere(mask > 0.3)
        if len(rows_cols) == 0:
            raise ValueError(f"{path} has no opaque pixel")
        (r0, c0), (r1, c1) = rows_cols.min(axis=0), rows_cols.max(axis=0)
        m = self.cfg.crop_margin
        crop = (slice(max(r0 - m, 0), min(r1 + m, h)), slice(max(c0 - m, 0), min(c1 + m, w)))
        image, mask = pad_square(image[crop], mask[crop], self.cfg.pad_ratio)
        image = self.transform(image)
        mask = self.transform(np.stack((mask, mask, mask), axis=-1))[:1]
        return image, mask

    def surface(self, rng) -> torch.Tensor:
        """(points_per_sample, 7): position, unit normal, and a zero sharp-edge label."""
        chosen = self.points[rng.choice(len(self.points), self.cfg.points_per_sample, replace=False)]
        chosen = np.concatenate((chosen, np.zeros((len(chosen), 1))), axis=1)
        chosen = torch.FloatTensor(chosen)
        normal = torch.nn.functional.normalize(chosen[:, 3:6], p=2, dim=1)
        return torch.cat([chosen[:, 0:3], normal, chosen[:, -1:]], dim=-1)

    def sample(self) -> dict:
        path = self.rng.sample(self.paths, 1)[0]
        image, mask = self.load_render(path)
        return {"image": image, "mask": mask, "surface": self.surface(np.random.default_rng()),
                "render": Path(path).name}

    def batch(self, size: int) -> dict:
        samples = [self.sample() for _ in range(size)]
        out = {key: torch.stack([s[key] for s in samples]) for key in ("image", "mask", "surface")}
        out["renders"] = [s["render"] for s in samples]
        return out
