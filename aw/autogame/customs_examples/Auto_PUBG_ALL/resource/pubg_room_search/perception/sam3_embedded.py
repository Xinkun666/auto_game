from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.sam3_http import (
    Sam3RemoteResult,
    result_to_dict,
    segment_door,
    segment_door_all,
    segment_house,
)


class RemoteSam3Perception:
    """Compatibility facade for callers that used the old embedded SAM3 API."""

    def segment_house(
        self,
        image_bgr: np.ndarray,
        min_mask_area_ratio: Optional[float] = None,
    ) -> Optional[Sam3RemoteResult]:
        return segment_house(image_bgr, min_mask_area_ratio=min_mask_area_ratio)

    def segment_door(
        self,
        image_bgr: np.ndarray,
        min_mask_area_ratio: Optional[float] = None,
    ) -> Optional[Sam3RemoteResult]:
        return segment_door(image_bgr, min_mask_area_ratio=min_mask_area_ratio)

    def segment_door_all(
        self,
        image_bgr: np.ndarray,
        *,
        max_masks: int = 8,
        min_mask_area_ratio: float = 0.001,
    ) -> list[Sam3RemoteResult]:
        return segment_door_all(
            image_bgr,
            max_masks=max_masks,
            min_mask_area_ratio=min_mask_area_ratio,
        )


_SAM3_SINGLETON: Optional[RemoteSam3Perception] = None
_SAM3_SINGLETON_LOCK = threading.Lock()


def get_sam3_perception() -> RemoteSam3Perception:
    global _SAM3_SINGLETON
    with _SAM3_SINGLETON_LOCK:
        if _SAM3_SINGLETON is None:
            _SAM3_SINGLETON = RemoteSam3Perception()
        return _SAM3_SINGLETON


def segmentation_to_dict(result: Sam3RemoteResult | None) -> dict:
    return result_to_dict(result)
