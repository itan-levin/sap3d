"""
Text Alignment Evaluation Module
Contains text alignment metrics such as CLIP-T
"""

import logging
import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class ImageTextPairDataset(Dataset):
    """Image-text pair dataset"""

    def __init__(self, image_text_pairs, image_size=(512, 512)):
        self.image_text_pairs = image_text_pairs
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_text_pairs)

    def __getitem__(self, idx):
        pair = self.image_text_pairs[idx]

        try:
            # Load image
            img = Image.open(pair.image_path)
            if img.mode != "RGB":
                background = Image.new("RGBA", img.size, (127, 127, 127, 255))
                background.paste(img, (0, 0), img)
                img = background.convert("RGB")
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

            # Get text
            text = pair.text

            return {"id": pair.id, "image": img, "text": text}

        except Exception as e:
            logger.warning(f"Failed to load image-text pair {pair.id}: {e}")
            return None


class TextAlignmentMetricsEvaluator:
    """Text alignment evaluator"""

    _supported_metrics = ["clip_t"]

    def __init__(
        self,
        device: str = "cuda:0",
        image_size: tuple = (512, 512),
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize CLIP model
        self.clip_model = (
            CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Image preprocessing
        self.to_tensor = transforms.ToTensor()

    def _collate_fn(self, batch):
        """Custom collate function to handle None values"""
        # Filter out None values
        valid_batch = [item for item in batch if item is not None]

        if not valid_batch:
            return None

        # Reorganize data
        collated = {}
        for key in valid_batch[0].keys():
            if key == "id":
                collated[key] = [item[key] for item in valid_batch]
            elif key == "text":
                collated[key] = [item[key] for item in valid_batch]
            elif key == "image":
                collated[key] = [item[key] for item in valid_batch]

        return collated

    def _compute_clip_similarity(self, image: Image.Image, text: str) -> float:
        """Calculate CLIP image-text similarity"""
        try:
            with torch.no_grad():
                # Process image and text
                image_inputs = self.clip_processor(
                    images=image, return_tensors="pt"
                ).to(self.device)
                text_inputs = self.clip_processor(
                    text=text, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                # Get image and text features
                image_features = self.clip_model.get_image_features(**image_inputs)
                text_features = self.clip_model.get_text_features(**text_inputs)

                # Normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Calculate cosine similarity
                similarity = torch.cosine_similarity(
                    image_features, text_features, dim=-1
                )

                return similarity.item()

        except Exception as e:
            logger.warning(f"CLIP similarity calculation failed: {e}")
            return None

    def _compute_clip_t_with_dataloader(self, image_text_pairs) -> dict:
        """Calculate CLIP-T using DataLoader"""
        logger.info(
            f"Computing CLIP-T (batch_size={self.batch_size}, num_workers={self.num_workers})..."
        )

        # Create dataset
        dataset = ImageTextPairDataset(image_text_pairs, self.image_size)

        # Create data loader
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.startswith("cuda") else False,
        )

        scores = []
        processed = 0
        skipped = 0

        for batch in tqdm(dataloader, desc="Computing CLIP-T"):
            if batch is None:
                skipped += self.batch_size
                continue

            batch_size = len(batch["id"])

            for i in range(batch_size):
                try:
                    # Calculate CLIP similarity
                    similarity = self._compute_clip_similarity(
                        batch["image"][i], batch["text"][i]
                    )

                    if similarity is not None:
                        scores.append(similarity)
                        processed += 1
                    else:
                        skipped += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to process image-text pair {batch['id'][i]}: {e}"
                    )
                    skipped += 1
                    continue

        return scores, processed, skipped

    def compute_clip_t(self, image_text_pairs) -> dict:
        """Calculate CLIP-T metric"""
        scores, processed, skipped = self._compute_clip_t_with_dataloader(
            image_text_pairs
        )

        logger.info(
            f"Successfully processed {processed} image-text pairs, skipped {skipped}"
        )

        if len(scores) == 0:
            return {"mean": None, "std": None, "count": 0}

        scores = np.array(scores)
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "count": len(scores),
            "all_scores": scores.tolist(),
        }

    def compute(
        self,
        image_text_pairs: list,
        metrics_to_compute: List[str] = _supported_metrics,
    ) -> dict:
        """Calculate CLIP-T metric"""
        result = {}
        metrics_to_compute = list(
            set(metrics_to_compute) & set(self._supported_metrics)
        )
        for metric in metrics_to_compute:
            if metric == "clip_t":
                result[metric] = self.compute_clip_t(image_text_pairs)
        return result
