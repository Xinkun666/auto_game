from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.perception.paths import (
    require_real_model_file,
    yolo_classify_model_path,
    yolo_detect_model_path,
)


@dataclass
class Bbox:
    x1_norm: float
    y1_norm: float
    x2_norm: float
    y2_norm: float


@dataclass
class DetectionItem:
    class_id: int
    class_name: str
    track_id: Optional[int]
    confidence: float
    bbox: Bbox

    def bbox_area(self) -> float:
        return max(0.0, self.bbox.x2_norm - self.bbox.x1_norm) * max(
            0.0, self.bbox.y2_norm - self.bbox.y1_norm
        )

    def get_center_x(self) -> float:
        return (self.bbox.x1_norm + self.bbox.x2_norm) / 2.0

    def get_center_y(self) -> float:
        return (self.bbox.y1_norm + self.bbox.y2_norm) / 2.0

    def to_list_xyxy(self, image_width: int, image_height: int) -> list[float]:
        return [
            self.bbox.x1_norm * image_width,
            self.bbox.y1_norm * image_height,
            self.bbox.x2_norm * image_width,
            self.bbox.y2_norm * image_height,
            self.confidence,
            self.class_id,
        ]

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "confidence": self.confidence,
            "bbox": {
                "x1_norm": self.bbox.x1_norm,
                "y1_norm": self.bbox.y1_norm,
                "x2_norm": self.bbox.x2_norm,
                "y2_norm": self.bbox.y2_norm,
            },
            "center": {
                "x_norm": self.get_center_x(),
                "y_norm": self.get_center_y(),
            },
            "area": self.bbox_area(),
        }


@dataclass
class DetectionResult:
    detections: list[DetectionItem]

    def is_empty(self) -> bool:
        return not self.detections

    def to_dict(self) -> dict:
        return {"detections": [item.to_dict() for item in self.detections]}


@dataclass
class ClassificationItem:
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
        }


@dataclass
class ClassificationResult:
    top1: Optional[ClassificationItem]
    top5_list: list[ClassificationItem]

    def to_dict(self) -> dict:
        return {
            "top1": self.top1.to_dict() if self.top1 is not None else None,
            "top5_list": [item.to_dict() for item in self.top5_list],
        }


@dataclass
class ClassificationDetectionResult:
    detection: DetectionResult
    classification: ClassificationResult

    def to_dict(self) -> dict:
        return {
            "detection": self.detection.to_dict(),
            "classification": self.classification.to_dict(),
        }


class FrameType(Enum):
    WALL = "wall"
    OUTDOOR = "outdoors"
    OTHER = "other"
    UNKNOWN = "unknown"


class ObjectType(Enum):
    INTERIOR_DOOR = "interior_door"
    EXTERIOR_DOOR = "exterior_door"
    OBJECT = "object"
    UPSTAIRS = "upstairs"
    DOWNSTAIRS = "downstairs"
    PICK_MENU = "pick_menu"


class EmbeddedYoloPerception:
    def __init__(
        self,
        *,
        detection_model_path=None,
        classification_model_path=None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.detection_model_path = detection_model_path or yolo_detect_model_path()
        self.classification_model_path = (
            classification_model_path or yolo_classify_model_path()
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._detect_model = None
        self._classify_model = None
        self._load_lock = threading.Lock()
        self._track_need_reset = False

    def load(self) -> None:
        with self._load_lock:
            if self._detect_model is not None and self._classify_model is not None:
                return
            from ultralytics import YOLO

            detect_path = require_real_model_file(
                self.detection_model_path, "PUBG YOLO 检测"
            )
            classify_path = require_real_model_file(
                self.classification_model_path, "PUBG YOLO 分类"
            )
            self._detect_model = YOLO(str(detect_path))
            self._classify_model = YOLO(str(classify_path))
            if self.device:
                self._detect_model.to(self.device)
                self._classify_model.to(self.device)

    def reset_tracker(self) -> None:
        self._track_need_reset = True

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        self.load()
        img_h, img_w = image_bgr.shape[:2]
        results = self._detect_model.track(
            source=image_bgr,
            persist=not self._track_need_reset,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        self._track_need_reset = False
        detections: list[DetectionItem] = []
        result = results[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1 = float(box.xyxy[0][0].item())
                y1 = float(box.xyxy[0][1].item())
                x2 = float(box.xyxy[0][2].item())
                y2 = float(box.xyxy[0][3].item())
                class_id = int(box.cls.item())
                detections.append(
                    DetectionItem(
                        class_id=class_id,
                        class_name=result.names.get(class_id, str(class_id)),
                        track_id=(
                            int(box.id) if box.is_track and box.id is not None else None
                        ),
                        confidence=round(float(box.conf.item()), 4),
                        bbox=Bbox(
                            x1_norm=round(x1 / img_w, 6),
                            y1_norm=round(y1 / img_h, 6),
                            x2_norm=round(x2 / img_w, 6),
                            y2_norm=round(y2 / img_h, 6),
                        ),
                    )
                )
        return DetectionResult(detections=detections)

    def classify(self, image_bgr: np.ndarray) -> ClassificationResult:
        self.load()
        results = self._classify_model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            verbose=False,
        )
        result = results[0]
        probs = result.probs
        names = result.names
        if probs is None:
            return ClassificationResult(top1=None, top5_list=[])
        top1_id = int(probs.top1)
        top1 = ClassificationItem(
            class_id=top1_id,
            class_name=names.get(top1_id, str(top1_id)),
            confidence=round(float(probs.top1conf), 4),
        )
        top5 = [
            ClassificationItem(
                class_id=int(class_id),
                class_name=names.get(int(class_id), str(class_id)),
                confidence=round(float(probs.data[int(class_id)]), 4),
            )
            for class_id in probs.top5
        ]
        return ClassificationResult(top1=top1, top5_list=top5)

    def classify_and_detect(self, image_bgr: np.ndarray) -> ClassificationDetectionResult:
        return ClassificationDetectionResult(
            detection=self.detect(image_bgr),
            classification=self.classify(image_bgr),
        )


_YOLO_SINGLETON: Optional[EmbeddedYoloPerception] = None
_YOLO_SINGLETON_LOCK = threading.Lock()


def get_yolo_perception() -> EmbeddedYoloPerception:
    global _YOLO_SINGLETON
    with _YOLO_SINGLETON_LOCK:
        if _YOLO_SINGLETON is None:
            _YOLO_SINGLETON = EmbeddedYoloPerception()
        return _YOLO_SINGLETON


def detect_as_forward_scene(image_bgr: np.ndarray) -> list[list[float]]:
    h, w = image_bgr.shape[:2]
    result = get_yolo_perception().detect(image_bgr)
    return [item.to_list_xyxy(w, h) for item in result.detections]

