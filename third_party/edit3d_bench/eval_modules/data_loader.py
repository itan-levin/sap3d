"""
Data Loader Module
Responsible for loading and organizing various types of data for evaluation
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ImagePair:
    """Image pair data"""

    gt_path: str
    pred_path: str
    mask_path: str
    id: str


@dataclass
class VideoPair:
    """Video pair data"""

    gt_path: str
    pred_path: str
    id: str


@dataclass
class ModelPair:
    """3D model pair data"""

    gt_path: str
    pred_path: str
    mask_path: str
    id: str


@dataclass
class ImageTextPair:
    """Image-text pair data"""

    image_path: str
    text: str
    id: str


class DataLoader:
    """Data loader"""

    def __init__(self, config):
        self.config = config
        self.gt_root = Path(config.gt_root)
        self.pred_root = Path(config.pred_root)

        # Load metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        """Load metadata file"""
        metadata_path = self.gt_root / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.warning(f"Metadata file does not exist: {metadata_path}")
            return []

    def _extract_id_by_prefix(self, filename: str, prefix: str) -> Optional[str]:
        """Extract ID from filename"""
        m = re.search(rf"{re.escape(prefix)}_(\d+)", filename)
        return m.group(1) if m else None

    def _collect_image_pairs(self) -> List[ImagePair]:
        """Collect image pair data"""
        pairs = []

        # Iterate through each sample in metadata
        for item in self.metadata:
            dataset = item["dataset"]
            source_model = item["source_model"]

            # Iterate through each prompt
            for prompt_idx in range(1, 4):
                prompt_key = f"prompt_{prompt_idx}"
                if prompt_key not in item:
                    continue

                # Build paths
                gt_render_dir = (
                    self.gt_root / dataset / source_model / "source_model" / "render"
                )
                gt_mask_dir = (
                    self.gt_root / dataset / source_model / prompt_key / "render"
                )
                pred_render_dir = (
                    self.pred_root / dataset / source_model / prompt_key / "images"
                )

                if (
                    not gt_render_dir.exists()
                    or not gt_mask_dir.exists()
                    or not pred_render_dir.exists()
                ):
                    continue

                # Collect GT images and masks
                gt_images = {}
                gt_masks = {}

                for file in gt_render_dir.glob("*.png"):
                    if file.name.startswith("render_"):
                        img_id = self._extract_id_by_prefix(file.name, "render")
                        if img_id:
                            gt_images[img_id] = str(file)

                for file in gt_mask_dir.glob("*.png"):
                    if file.name.startswith("mask_"):
                        mask_id = self._extract_id_by_prefix(file.name, "mask")
                        if mask_id:
                            gt_masks[mask_id] = str(file)

                # Collect prediction images
                pred_images = {}
                for file in pred_render_dir.glob("*.png"):
                    if file.name.startswith("render_"):
                        img_id = self._extract_id_by_prefix(file.name, "render")
                        if img_id:
                            pred_images[img_id] = str(file)

                # Pair matching
                common_ids = (
                    set(gt_images.keys())
                    & set(pred_images.keys())
                    & set(gt_masks.keys())
                )
                for img_id in sorted(common_ids, key=lambda x: int(x)):
                    pairs.append(
                        ImagePair(
                            gt_path=gt_images[img_id],
                            pred_path=pred_images[img_id],
                            mask_path=gt_masks[img_id],
                            id=f"{dataset}_{source_model}_{prompt_key}_{img_id}",
                        )
                    )

        logger.info(f"Collected {len(pairs)} image pairs")
        return pairs

    def _collect_video_pairs(self) -> List[VideoPair]:
        """Collect video pair data"""
        pairs = []

        for item in self.metadata:
            dataset = item["dataset"]
            source_model = item["source_model"]

            for prompt_idx in range(1, 4):
                prompt_key = f"prompt_{prompt_idx}"
                if prompt_key not in item:
                    continue

                # Build paths
                gt_video_path = (
                    self.gt_root
                    / dataset
                    / source_model
                    / "source_model"
                    / "video_rgb.mp4"
                )
                pred_video_path = (
                    self.pred_root
                    / dataset
                    / source_model
                    / prompt_key
                    / "videos"
                    / "video_rgb.mp4"
                )

                if gt_video_path.exists() and pred_video_path.exists():
                    pairs.append(
                        VideoPair(
                            gt_path=str(gt_video_path),
                            pred_path=str(pred_video_path),
                            id=f"{dataset}_{source_model}_{prompt_key}",
                        )
                    )

        logger.info(f"Collected {len(pairs)} video pairs")
        return pairs

    def _collect_model_pairs(self) -> List[ModelPair]:
        """Collect 3D model pair data"""
        pairs = []

        for item in self.metadata:
            dataset = item["dataset"]
            source_model = item["source_model"]

            for prompt_idx in range(1, 4):
                prompt_key = f"prompt_{prompt_idx}"
                if prompt_key not in item:
                    continue

                # Build paths
                gt_model_path = (
                    self.gt_root / dataset / source_model / "source_model" / "model.glb"
                )
                pred_model_path = (
                    self.pred_root / dataset / source_model / prompt_key / "edit.glb"
                )
                mask_path = (
                    self.gt_root
                    / dataset
                    / source_model
                    / prompt_key
                    / "3d_edit_region.glb"
                )

                if (
                    gt_model_path.exists()
                    and pred_model_path.exists()
                    and mask_path.exists()
                ):
                    pairs.append(
                        ModelPair(
                            gt_path=str(gt_model_path),
                            pred_path=str(pred_model_path),
                            mask_path=str(mask_path),
                            id=f"{dataset}_{source_model}_{prompt_key}",
                        )
                    )

        logger.info(f"Collected {len(pairs)} model pairs")
        return pairs

    def _collect_image_text_pairs(self) -> List[ImageTextPair]:
        """Collect image-text pair data"""
        pairs = []

        for item in self.metadata:
            dataset = item["dataset"]
            source_model = item["source_model"]

            for prompt_idx in range(1, 4):
                prompt_key = f"prompt_{prompt_idx}"
                if prompt_key not in item:
                    continue

                # Build paths
                pred_render_dir = (
                    self.pred_root / dataset / source_model / prompt_key / "images"
                )
                text_path = (
                    self.gt_root / dataset / source_model / prompt_key / "prompt.txt"
                )

                if not pred_render_dir.exists():
                    continue

                # If no dedicated text file, use prompt from metadata
                if text_path.exists():
                    text_content = text_path.read_text(encoding="utf-8").strip()
                else:
                    text_content = item[prompt_key]

                # Create image-text pairs for each rendered image
                for file in pred_render_dir.glob("*.png"):
                    if file.name.startswith("render_"):
                        img_id = self._extract_id_by_prefix(file.name, "render")
                        if img_id and img_id in [
                            "0000",
                            "0001",
                            "0007",
                            "0008",
                            "0009",
                            "0015",
                        ]:  # select the front -45~45 degree
                            pairs.append(
                                ImageTextPair(
                                    image_path=str(file),
                                    text=text_content,
                                    id=f"{dataset}_{source_model}_{prompt_key}_{img_id}",
                                )
                            )

        logger.info(f"Collected {len(pairs)} image-text pairs")
        return pairs

    def _get_all_image_paths(
        self, pairs: List[ImagePair]
    ) -> Tuple[List[str], List[str]]:
        """Extract all image paths from image pairs"""
        gt_paths = [pair.gt_path for pair in pairs]
        pred_paths = [pair.pred_path for pair in pairs]
        return gt_paths, pred_paths

    def load_all_data(self) -> Dict[str, Any]:
        """Load all data"""
        data = {}

        # Load image pairs
        try:
            image_pairs = self._collect_image_pairs()
            if self.config.max_images and len(image_pairs) > self.config.max_images:
                image_pairs = image_pairs[: self.config.max_images]
            data["image_pairs"] = image_pairs
        except Exception as e:
            logger.warning(f"Failed to load image pairs: {e}")
            data["image_pairs"] = []

        # Extract all image paths (for distribution metrics)
        try:
            gt_images, pred_images = self._get_all_image_paths(image_pairs)
            data["gt_images"] = gt_images
            data["pred_images"] = pred_images
        except Exception as e:
            logger.warning(f"Failed to load image paths: {e}")
            data["gt_images"] = []
            data["pred_images"] = []

        # Load video pairs
        try:
            video_pairs = self._collect_video_pairs()
            if self.config.max_videos and len(video_pairs) > self.config.max_videos:
                video_pairs = video_pairs[: self.config.max_videos]
            data["video_pairs"] = video_pairs
        except Exception as e:
            logger.warning(f"Failed to load video pairs: {e}")
            data["video_pairs"] = []

        # Extract video paths
        try:
            data["gt_videos"] = [pair.gt_path for pair in video_pairs]
            data["pred_videos"] = [pair.pred_path for pair in video_pairs]
        except Exception as e:
            logger.warning(f"Failed to extract video paths: {e}")
            data["gt_videos"] = []
            data["pred_videos"] = []

        # Load model pairs
        try:
            model_pairs = self._collect_model_pairs()
            data["model_pairs"] = model_pairs
        except Exception as e:
            logger.warning(f"Failed to load model pairs: {e}")
            data["model_pairs"] = []

        # Extract model paths
        try:
            data["gt_models"] = [pair.gt_path for pair in model_pairs]
            data["pred_models"] = [pair.pred_path for pair in model_pairs]
            data["masks"] = [pair.mask_path for pair in model_pairs]
        except Exception as e:
            logger.warning(f"Failed to extract model paths: {e}")
            data["gt_models"] = []
            data["pred_models"] = []
            data["masks"] = []

        # Load image-text pairs
        try:
            image_text_pairs = self._collect_image_text_pairs()
            data["image_text_pairs"] = image_text_pairs
        except Exception as e:
            logger.warning(f"Failed to load image-text pairs: {e}")
            data["image_text_pairs"] = []

        return data
