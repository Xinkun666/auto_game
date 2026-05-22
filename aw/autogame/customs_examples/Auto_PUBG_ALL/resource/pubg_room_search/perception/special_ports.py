from __future__ import annotations

import logging

import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.sam3_http import (
    result_to_dict,
    segment_door,
    segment_door_all,
    segment_house,
    to_bgr,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.yolo_embedded import (
    detect_as_forward_scene,
    get_yolo_perception,
)

logger = logging.getLogger(__name__)

def _to_bgr(img: np.ndarray) -> np.ndarray:
    return to_bgr(img)


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
    try:
        result = segment_house(img)
    except Exception as exc:
        logger.warning("SAM3 HTTP house segmentation failed: %s", exc)
        return {"ok": False, "result": None, "error": str(exc)}
    return result_to_dict(result, label="sam3 house")


def pubg_sam3_segment_door(img: np.ndarray):
    try:
        result = segment_door(img)
    except Exception as exc:
        logger.warning("SAM3 HTTP door segmentation failed: %s", exc)
        return {"ok": False, "result": None, "error": str(exc)}
    return result_to_dict(result, label="sam3 door")


def pubg_sam3_segment_door_all(img: np.ndarray):
    try:
        results = segment_door_all(img)
    except Exception as exc:
        logger.warning("SAM3 HTTP door_all segmentation failed: %s", exc)
        return {"ok": False, "results": [], "error": str(exc)}
    return {
        "ok": True,
        "results": [result.to_summary_dict() for result in results],
        "__visualizations__": [
            result.to_visualization_dict(f"sam3 door {index + 1}")
            for index, result in enumerate(results)
        ],
    }
