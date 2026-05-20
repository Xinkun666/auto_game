from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.paths import (
    VENDORED_SAM3_DIR,
    require_real_model_file,
    sam3_bpe_path,
    sam3_checkpoint_path,
)


def _ensure_vendored_sam3_path() -> None:
    vendored = str(VENDORED_SAM3_DIR)
    if vendored not in sys.path:
        sys.path.insert(0, vendored)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class EmbeddedSam3Perception:
    def __init__(
        self,
        *,
        checkpoint_path: Optional[Path] = None,
        bpe_path: Optional[Path] = None,
        device: str = "auto",
        load_from_hf: Optional[bool] = None,
    ):
        self.checkpoint_path = checkpoint_path or sam3_checkpoint_path()
        self.bpe_path = bpe_path if bpe_path is not None else sam3_bpe_path()
        self.device = os.environ.get("AUTOGAME_PUBG_SAM3_DEVICE", device)
        self.load_from_hf = (
            _env_bool("AUTOGAME_PUBG_SAM3_LOAD_FROM_HF", False)
            if load_from_hf is None
            else load_from_hf
        )
        self._segmenter = None
        self._load_lock = threading.Lock()

    def load(self):
        with self._load_lock:
            if self._segmenter is not None:
                return self._segmenter
            _ensure_vendored_sam3_path()
            from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.sam3.segmenter import (
                Sam3Segmenter,
            )

            checkpoint = None
            if self.checkpoint_path.exists() or not self.load_from_hf:
                checkpoint = require_real_model_file(
                    self.checkpoint_path, "PUBG SAM3 checkpoint"
                )
            self._segmenter = Sam3Segmenter(
                backend=Sam3Segmenter.BACKEND_LOCAL,
                checkpoint_path=checkpoint,
                bpe_path=self.bpe_path,
                load_from_hf=self.load_from_hf,
                device=self.device,
            )
            self._segmenter.load_model()
            return self._segmenter

    def segment_house(self, image_bgr: np.ndarray, min_mask_area_ratio=None):
        return self.load().segment_house(
            image_bgr, min_mask_area_ratio=min_mask_area_ratio
        )

    def segment_door(self, image_bgr: np.ndarray, min_mask_area_ratio=None):
        return self.load().segment_door(image_bgr, min_mask_area_ratio=min_mask_area_ratio)

    def segment_door_all(
        self,
        image_bgr: np.ndarray,
        *,
        max_masks: int = 8,
        min_mask_area_ratio: float = 0.001,
    ):
        return self.load().segment_door_all(
            image_bgr,
            max_masks=max_masks,
            min_mask_area_ratio=min_mask_area_ratio,
        )


_SAM3_SINGLETON: Optional[EmbeddedSam3Perception] = None
_SAM3_SINGLETON_LOCK = threading.Lock()


def get_sam3_perception() -> EmbeddedSam3Perception:
    global _SAM3_SINGLETON
    with _SAM3_SINGLETON_LOCK:
        if _SAM3_SINGLETON is None:
            _SAM3_SINGLETON = EmbeddedSam3Perception()
        return _SAM3_SINGLETON


def segmentation_to_dict(result) -> dict:
    if result is None:
        return {"ok": False, "result": None}
    return {
        "ok": True,
        "result": {
            "bbox_xyxy": [int(v) for v in result.bbox_xyxy],
            "score": float(result.score),
            "mask_shape": list(result.mask.shape),
            "mask_area": int((result.mask > 0).sum()),
            "sam3_inference_ms": result.sam3_inference_ms,
        },
    }

