"""
Distribution Evaluation Module
Contains distribution metrics such as FID, DINO-I
"""

import logging
from typing import List

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Image dataset (for distribution metrics)"""

    def __init__(self, image_paths, image_size=(512, 512)):
        self.image_paths = image_paths
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            # Load image
            img = Image.open(img_path)
            if img.mode != "RGB":
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                background.paste(img, (0, 0), img)
                img = background.convert("RGB")
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

            # Convert to tensor
            img_tensor = self.to_tensor(img)  # [0,1]

            # Convert to uint8 tensor (required by FID)
            img_uint8 = (img_tensor * 255).type(torch.uint8)

            # Convert to PIL image (required by DINO)
            img_pil = img

            return {"path": img_path, "img_tensor": img_tensor, "img_uint8": img_uint8}

        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            return None


class DistributionMetricsEvaluator:
    """Distribution evaluator"""

    _supported_metrics = ["fid"]

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

        # Initialize FID
        self.fid = FrechetInceptionDistance(feature=2048).to(device)

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
            if key == "path":
                collated[key] = [item[key] for item in valid_batch]
            else:
                collated[key] = torch.stack([item[key] for item in valid_batch])

        return collated

    def _compute_fid_with_dataloader(self, gt_images: list, pred_images: list) -> float:
        """Calculate FID using DataLoader"""

        # Reset FID
        self.fid.reset()

        # Process real images
        logger.info("Processing real images...")
        gt_dataset = ImageDataset(gt_images, self.image_size)
        gt_dataloader = TorchDataLoader(
            gt_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.startswith("cuda") else False,
        )

        processed_gt = 0
        for batch in tqdm(gt_dataloader, desc="Real images"):
            if batch is not None:
                batch_tensor = batch["img_uint8"].to(self.device)
                self.fid.update(batch_tensor, real=True)
                processed_gt += len(batch["path"])

        # Process generated images
        logger.info("Processing generated images...")
        pred_dataset = ImageDataset(pred_images, self.image_size)
        pred_dataloader = TorchDataLoader(
            pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.startswith("cuda") else False,
        )

        processed_pred = 0
        for batch in tqdm(pred_dataloader, desc="Generated images"):
            if batch is not None:
                batch_tensor = batch["img_uint8"].to(self.device)
                self.fid.update(batch_tensor, real=False)
                processed_pred += len(batch["path"])

        logger.info(
            f"Successfully processed {processed_gt} real images and {processed_pred} generated images"
        )

        # Calculate FID score
        fid_score = self.fid.compute().item()
        return fid_score

    def compute(
        self,
        gt_images: list,
        pred_images: list,
        metrics_to_compute: List[str] = _supported_metrics,
    ) -> dict:
        """Calculate FID metric"""
        results = {}
        metrics_to_compute = list(
            set(metrics_to_compute) & set(self._supported_metrics)
        )
        logger.info(
            f"Computing {metrics_to_compute} metrics (batch_size={self.batch_size}, num_workers={self.num_workers})..."
        )
        for metric in metrics_to_compute:
            if metric == "fid":
                results[metric] = self._compute_fid_with_dataloader(
                    gt_images, pred_images
                )

        return results
