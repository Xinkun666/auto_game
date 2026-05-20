from __future__ import annotations

import cv2
import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.sam3_embedded import (
    get_sam3_perception,
    segmentation_to_dict,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.yolo_embedded import (
    detect_as_forward_scene,
    get_yolo_perception,
)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def pubg_yolo_detect(img: np.ndarray):
    return detect_as_forward_scene(_to_bgr(img))


def pubg_yolo_detect_detail(img: np.ndarray):
    return get_yolo_perception().detect(_to_bgr(img)).to_dict()


def pubg_yolo_classify(img: np.ndarray):
    return get_yolo_perception().classify(_to_bgr(img)).to_dict()


def pubg_yolo_detect_and_classify(img: np.ndarray):
    return get_yolo_perception().classify_and_detect(_to_bgr(img)).to_dict()


def pubg_yolo_reset_tracker(_img: np.ndarray = None):
    get_yolo_perception().reset_tracker()
    return {"ok": True, "status": "reset_requested"}


def pubg_sam3_segment_house(img: np.ndarray):
    result = get_sam3_perception().segment_house(_to_bgr(img))
    return segmentation_to_dict(result)


def pubg_sam3_segment_door(img: np.ndarray):
    result = get_sam3_perception().segment_door(_to_bgr(img))
    return segmentation_to_dict(result)


def pubg_sam3_segment_door_all(img: np.ndarray):
    results = get_sam3_perception().segment_door_all(_to_bgr(img))
    return {
        "ok": True,
        "results": [segmentation_to_dict(result)["result"] for result in results],
    }

