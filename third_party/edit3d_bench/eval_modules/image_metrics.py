"""
Image Quality Evaluation Module
Contains image quality metrics such as PSNR, SSIM, LPIPS, etc.
"""

import logging
import os
from typing import List

import lpips
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

logger = logging.getLogger(__name__)


class ImagePairDataset(Dataset):
    """Image pair dataset"""

    def __init__(self, image_pairs, image_size=(512, 512), ignore_mask=False):
        self.image_pairs = image_pairs
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        self.ignore_mask = ignore_mask

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]

        try:
            # Load images
            img1 = Image.open(pair.gt_path)
            if img1.mode != "RGB":
                background = Image.new("RGBA", img1.size, (127, 127, 127, 255))
                background.paste(img1, (0, 0), img1)
                img1 = background.convert("RGB")
            img1 = img1.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

            img2 = Image.open(pair.pred_path)
            if img2.mode != "RGB":
                background = Image.new("RGBA", img2.size, (127, 127, 127, 255))
                background.paste(img2, (0, 0), img2)
                img2 = background.convert("RGB")
            img2 = img2.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

            # Load mask
            mask = Image.open(pair.mask_path).convert("L")
            mask = mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
            mask_np = np.array(mask, dtype=np.uint8)
            include_mask = (
                mask_np <= 0
            )  # Inverted logic: only non-edited regions participate in comparison

            if self.ignore_mask:
                include_mask = np.ones_like(mask_np)

            # Convert to tensor
            img1_tensor = self.to_tensor(img1)  # [0,1]
            img2_tensor = self.to_tensor(img2)  # [0,1]

            # Convert to mask tensor
            include_mask_tensor = torch.from_numpy(include_mask.astype(np.float32))

            return {
                "id": pair.id,
                "img1": img1_tensor,  # [0,1] for PSNR and SSIM
                "img2": img2_tensor,  # [0,1] for PSNR and SSIM
                "img1_pil": img1,
                "img2_pil": img2,
                "include_mask": include_mask_tensor,  # [H,W] for all metrics
            }

        except Exception as e:
            logger.warning(f"Failed to load image pair {pair.id}: {e}")
            return None


class ImageMetricsEvaluator:
    """Image quality evaluator"""

    _supported_metrics = ["psnr", "ssim", "lpips", "dino_i"]

    def __init__(
        self,
        device: str = "cuda:0",
        image_size: tuple = (512, 512),
        batch_size: int = 32,
        num_workers: int = 4,
        ignore_mask: bool = False,
    ):
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_mask = ignore_mask

        # Initialize LPIPS
        self.lpips_fn = (
            lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()
        )

        # Initialize DINO
        self.dino_model = (
            Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
        )
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

        # Image preprocessing
        self.to_tensor = transforms.ToTensor()

        # SSIM window cache
        self._ssim_window = None

    def _get_ssim_window(self, window_size: int = 11, channel: int = 3):
        """Get SSIM window with caching"""
        if self._ssim_window is None or self._ssim_window.size(2) != window_size:

            def gaussian(window_size, sigma):
                gauss = torch.Tensor(
                    [
                        torch.exp(
                            torch.tensor(
                                -((x - window_size // 2) ** 2) / float(2 * sigma**2)
                            )
                        )
                        for x in range(window_size)
                    ]
                )
                return gauss / gauss.sum()

            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(
                channel, 1, window_size, window_size
            ).contiguous()
            self._ssim_window = window.to(self.device)

        return self._ssim_window

    @torch.no_grad()
    def _calculate_psnr_torch(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor,
        max_val: float = 255.0,
    ) -> float:
        """Calculate PSNR with mask using PyTorch"""
        # Ensure inputs are on the correct device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        mask = mask.to(self.device)

        # Convert [0,1] range to [0,255] range to match skimage
        img1 = img1 * 255.0
        img2 = img2 * 255.0

        mse = torch.mean((img1 - img2) ** 2, dim=0)  # [C,H,W]

        # Calculate average MSE in valid regions
        valid_pixels = mask.bool()  # Ensure mask is [H,W] shape
        if valid_pixels.sum() == 0:
            return None

        mse_valid = mse[valid_pixels].mean()

        if mse_valid == 0:
            return None

        # Calculate PSNR using standard formula
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse_valid))

        return psnr.item()

    @torch.no_grad()
    def _calculate_ssim_torch(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        mask: torch.Tensor,
        window_size: int = 11,
    ) -> float:
        """Calculate SSIM with mask using PyTorch"""
        # Ensure inputs are on the correct device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        mask = mask.to(self.device)

        # Get cached SSIM window
        window = self._get_ssim_window(window_size, img1.size(1))

        # Calculate SSIM
        mu1 = torch.nn.functional.conv2d(
            img1, window, padding=window_size // 2, groups=img1.size(1)
        )
        mu2 = torch.nn.functional.conv2d(
            img2, window, padding=window_size // 2, groups=img2.size(1)
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            torch.nn.functional.conv2d(
                img1 * img1, window, padding=window_size // 2, groups=img1.size(1)
            )
            - mu1_sq
        )
        sigma2_sq = (
            torch.nn.functional.conv2d(
                img2 * img2, window, padding=window_size // 2, groups=img2.size(1)
            )
            - mu2_sq
        )
        sigma12 = (
            torch.nn.functional.conv2d(
                img1 * img2, window, padding=window_size // 2, groups=img1.size(1)
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        # Calculate average SSIM in valid regions
        valid_pixels = mask.squeeze().bool()  # Ensure mask is [H,W] shape
        if valid_pixels.sum() == 0:
            return None

        # Ensure ssim_map is [H,W] shape
        ssim_map = ssim_map.squeeze()  # Remove batch dimension
        ssim_valid = ssim_map.mean(dim=0)[
            valid_pixels
        ].mean()  # First average over channels, then average over valid pixels
        return ssim_valid.item()

    @torch.no_grad()
    def _calculate_lpips_masked(
        self, img1_t: torch.Tensor, img2_t: torch.Tensor, include_mask_t: torch.Tensor
    ) -> float:
        """Calculate LPIPS with mask"""
        # Calculate LPIPS spatial map
        lpips_map = self.lpips_fn(img1_t, img2_t)  # [B,1,H,W]

        # Calculate average LPIPS in valid regions
        valid_lpips = lpips_map[include_mask_t.bool()]
        if len(valid_lpips) == 0:
            return None

        return valid_lpips.mean().cpu().numpy().tolist()

    @torch.no_grad()
    def _calculate_dino_i(
        self, img1_pil: List[Image.Image], img2_pil: List[Image.Image]
    ) -> float:
        img1_features = self.dino_processor(images=img1_pil, return_tensors="pt").to(
            self.device
        )
        img1_features = self.dino_model(**img1_features).last_hidden_state

        img2_features = self.dino_processor(images=img2_pil, return_tensors="pt").to(
            self.device
        )
        img2_features = self.dino_model(**img2_features).last_hidden_state

        img1_features = img1_features.mean(dim=1)
        img2_features = img2_features.mean(dim=1)

        img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)
        img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)

        similarity_score = torch.nn.functional.cosine_similarity(
            img1_features, img2_features, dim=1
        )

        return similarity_score.cpu().numpy().tolist()

    def _compute_metrics_with_dataloader(self, image_pairs, metrics_to_compute):
        """Calculate metrics using DataLoader"""
        # Create dataset
        dataset = ImagePairDataset(image_pairs, self.image_size, self.ignore_mask)

        # Create data loader
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if self.device.startswith("cuda") else False,
        )

        scores = {metric: [] for metric in metrics_to_compute}
        processed = 0
        skipped = 0

        for batch in tqdm(
            dataloader, desc=f"Computing {', '.join(metrics_to_compute)}"
        ):
            if batch is None:
                skipped += self.batch_size
                continue

            batch_size = len(batch["id"])

            # Calculate PSNR
            if "psnr" in metrics_to_compute:
                for i in range(batch_size):
                    psnr = self._calculate_psnr_torch(
                        batch["img1"][i], batch["img2"][i], batch["include_mask"][i]
                    )
                    if psnr is not None:
                        scores["psnr"].append(psnr)
                        processed += 1
                    else:
                        skipped += 1

            # Calculate SSIM
            if "ssim" in metrics_to_compute:
                for i in range(batch_size):
                    ssim = self._calculate_ssim_torch(
                        batch["img1"][i].unsqueeze(0),
                        batch["img2"][i].unsqueeze(0),
                        batch["include_mask"][i],
                    )
                    if ssim is not None:
                        scores["ssim"].append(ssim)

            # Calculate LPIPS
            if "lpips" in metrics_to_compute:
                for i in range(batch_size):
                    img1_t = batch["img1"][i].unsqueeze(0).to(self.device)
                    img2_t = batch["img2"][i].unsqueeze(0).to(self.device)
                    mask_t = (
                        batch["include_mask"][i]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.device)
                    )  # [1,1,H,W]

                    lpips_score = self._calculate_lpips_masked(img1_t, img2_t, mask_t)
                    if lpips_score is not None:
                        scores["lpips"].append(lpips_score)

            # Calculate DINO-I
            if "dino_i" in metrics_to_compute:
                dino_i_score = self._calculate_dino_i(
                    batch["img1_pil"], batch["img2_pil"]
                )
                if dino_i_score is not None:
                    scores["dino_i"].extend(dino_i_score)

        return scores, processed, skipped

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
            elif key == "img1_pil" or key == "img2_pil":
                collated[key] = [item[key] for item in valid_batch]
            else:
                collated[key] = torch.stack([item[key] for item in valid_batch])

        return collated

    def compute(
        self,
        image_pairs,
        metrics_to_compute: List[str] = _supported_metrics,
    ) -> dict:
        metrics_to_compute = list(
            set(metrics_to_compute) & set(self._supported_metrics)
        )
        logger.info(
            f"Computing {metrics_to_compute} metrics (batch_size={self.batch_size}, num_workers={self.num_workers})..."
        )

        scores, processed, skipped = self._compute_metrics_with_dataloader(
            image_pairs, metrics_to_compute
        )

        results = {}
        for metric in metrics_to_compute:
            if len(scores[metric]) == 0:
                results[metric] = {"mean": None, "std": None, "count": 0}
            else:
                scores_array = np.array(scores[metric])
                results[metric] = {
                    "mean": float(np.mean(scores_array)),
                    "std": float(np.std(scores_array)),
                    "count": len(scores_array),
                    "all_scores": scores_array.tolist(),
                }

        return results
