"""Edit fidelity: how close a render of the edit is to the edited view, inside the 2D mask.

The mask's bounding box, squared about its center, is cut from the render and from the edited view (both
composited onto white) at full resolution; every pixel outside the mask is set to gray 176 in both, so only
masked content can move a distance. Each crop is resampled to 224 px (bilinear) and encoded:
`dino_embedding_distance` is 1 - cosine of the two DINOv2-base pooled class tokens, `clip_embedding_distance`
1 - cosine of the two CLIP ViT-B/32 image embeddings.
"""

import numpy as np
import torch
from PIL import Image

from ..edit.patches import load_mask

ENCODERS = {"clip": "openai/clip-vit-base-patch32", "dino": "facebook/dinov2-base"}
FEATURE_INPUT = 224
UNSCORED_GRAY = 176.0
MIN_CROP_SIDE = 11
RESAMPLE = Image.Resampling.BILINEAR


def on_white(image) -> Image.Image:
    """An image with every transparent pixel replaced by white, as RGB."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode == "RGB":
        return image
    flat = Image.new("RGB", image.size, (255, 255, 255))
    flat.paste(image, mask=image.convert("RGBA").getchannel("A"))
    return flat


def square_box(region: np.ndarray) -> tuple:
    """(top, left, side): the mask's bounding box squared about its center, inside the frame."""
    rows, cols = np.nonzero(region)
    if rows.size == 0:
        raise ValueError("the 2D mask is empty, so there is no edit region to score")
    height, width = region.shape
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    side = min(max(bottom - top, right - left), height, width)
    if side < MIN_CROP_SIDE:
        raise ValueError(f"the mask spans {bottom - top}x{right - left} px, under the {MIN_CROP_SIDE} px the "
                         f"comparison needs")
    box_top = min(max((top + bottom - side) // 2, 0), height - side)
    box_left = min(max((left + right - side) // 2, 0), width - side)
    if box_top > top or box_left > left or box_top + side < bottom or box_left + side < right:
        raise ValueError(f"a {side} px square cannot hold a mask spanning {bottom - top}x{right - left} px")
    return box_top, box_left, side


def masked_crops(render, target, mask) -> tuple:
    """The two (side, side, 3) float64 crops with everything outside the mask set to gray, and the box."""
    region = load_mask(mask)
    height, width = region.shape
    rendered, wanted = on_white(render), on_white(target)
    if rendered.size != (width, height) or wanted.size != (width, height):
        raise ValueError(f"the mask is {height}x{width} but the render is {rendered.size[1]}x{rendered.size[0]} "
                         f"and the edited view {wanted.size[1]}x{wanted.size[0]}")
    top, left, side = square_box(region)
    annotated = region[top:top + side, left:left + side]
    crops = []
    for flat in (rendered, wanted):
        patch = np.asarray(flat.crop((left, top, left + side, top + side)), dtype=np.float64)
        patch[~annotated] = UNSCORED_GRAY
        crops.append(patch)
    if np.array_equal(crops[0], crops[1]):
        raise ValueError("the render and the edited view hold the same pixels inside the mask")
    return crops, {"box_top": top, "box_left": left, "crop_side": side, "mask_pixels": int(region.sum())}


def resampled(patch: np.ndarray, side: int) -> np.ndarray:
    picture = Image.fromarray(np.round(patch).clip(0, 255).astype(np.uint8))
    if picture.size != (side, side):
        picture = picture.resize((side, side), RESAMPLE)
    return np.asarray(picture, dtype=np.float64)


class FidelityScorer:
    """Both encoders, loaded on first use and kept for every later score."""

    def __init__(self, device: str):
        self.device = torch.device(device)
        self.models = {}

    def encoder(self, name: str):
        if name not in self.models:
            from transformers import AutoImageProcessor, CLIPVisionModelWithProjection, Dinov2Model

            repo = ENCODERS[name]
            builder = CLIPVisionModelWithProjection if name == "clip" else Dinov2Model
            model = builder.from_pretrained(repo).to(self.device).eval()
            processor = AutoImageProcessor.from_pretrained(repo)
            mean = torch.tensor(processor.image_mean, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(processor.image_std, device=self.device).view(1, 3, 1, 1)
            self.models[name] = (model, mean, std)
        return self.models[name]

    def embedding_distance(self, name: str, first: np.ndarray, second: np.ndarray) -> float:
        model, mean, std = self.encoder(name)
        pixels = torch.from_numpy(np.stack([resampled(patch, FEATURE_INPUT).transpose(2, 0, 1) / 255.0
                                            for patch in (first, second)]).astype(np.float32)).to(self.device)
        with torch.no_grad():
            output = model(pixel_values=(pixels - mean) / std)
        descriptors = output.image_embeds if name == "clip" else output.pooler_output
        return float(1.0 - torch.cosine_similarity(descriptors[0:1], descriptors[1:2], dim=1).item())

    def score(self, render, target, mask) -> dict:
        (first, second), record = masked_crops(render, target, mask)
        record["dino_embedding_distance"] = self.embedding_distance("dino", first, second)
        record["clip_embedding_distance"] = self.embedding_distance("clip", first, second)
        record["device"] = str(self.device)
        if not all(np.isfinite(record[key]) for key in ("dino_embedding_distance", "clip_embedding_distance")):
            raise ValueError(f"unusable fidelity score {record}")
        return record
