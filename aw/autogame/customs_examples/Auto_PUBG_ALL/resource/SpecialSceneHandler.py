# -*- coding: utf-8 -*-
from aw.autogame.tools.Utils import *

import time
from functools import wraps
from pathlib import Path

dire_tool_ctc = loc_tool = yolo_detector = tracker = speed_cls = scene_cls = None
house_yolo_detector = None
WEIGHTS_DIR = Path(__file__).resolve().parent / 'weights'
DETECTION_CLASSES = {0: 'house', 1: 'door', 2: 'open_door', 3: 'window', 4: 'car', 5: 'wrecked_car'}
DETECTION_BUSINESS_IDS = {0: 8, 1: 0, 2: 4, 3: 2, 4: 7, 5: 17}


def _load_detection_model(filename):
    from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.yolo_detector import YOLO26Detector
    detector = YOLO26Detector(model_path=str(WEIGHTS_DIR / filename))
    if detector.names != DETECTION_CLASSES:
        raise ValueError(f'{filename} 权重类别不匹配: {detector.names}; 预期 {DETECTION_CLASSES}')
    return detector


def _business_detections(detector, img):
    """六类训练编号转成控制器编号，车辆与废车分别为 7、17。"""
    return [
        [int(x1), int(y1), int(x2), int(y2), conf, DETECTION_BUSINESS_IDS[int(cls)]]
        for x1, y1, x2, y2, conf, cls in detector.infer(img)
    ]


def _direction_tool():
    global dire_tool_ctc
    if dire_tool_ctc is None:
        from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.direction_ctc_service import Get_Direction
        dire_tool_ctc = Get_Direction(model_weight=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/direction_ctc.pt')
    return dire_tool_ctc


def _location_tool():
    global loc_tool
    if loc_tool is None:
        from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.location_service import LocatePoints
        loc_tool = LocatePoints()
    return loc_tool


def _yolo_detector():
    global yolo_detector
    if yolo_detector is None:
        yolo_detector = _load_detection_model('driving.pt')
    return yolo_detector


def house_forward_scene(img):
    """搜房入口，输出控制器通用类别编号。"""
    global house_yolo_detector
    if house_yolo_detector is None:
        house_yolo_detector = _load_detection_model('house_search.pt')
    return _business_detections(house_yolo_detector, img)


def _tracker():
    global tracker
    if tracker is None:
        from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.angle_tracker import AngleTracker
        tracker = AngleTracker(window_size=30)
    return tracker


def _speed_classifier():
    global speed_cls
    if speed_cls is None:
        from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.speed_classifier import SpeedClassifier
        speed_cls = SpeedClassifier(weight_path=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/speed_classifier.pt')
    return speed_cls


def _scene_classifier():
    global scene_cls
    if scene_cls is None:
        from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.scene_predictor import GameSceneClassifier
        scene_cls = GameSceneClassifier(checkpoint_path=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/scene_best_model.pth')
    return scene_cls


def special_timing(func):
    """
    Special 方法耗时统计装饰器。
    使用方式：在需要统计的 special 方法上方增加 @special_timing。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return result, elapsed_ms

    wrapper.__special_timing_enabled__ = True
    return wrapper


def direction(img):
    return _direction_tool().get_direction(img)


def location(img):
    return _location_tool().get_location(img)


def reset_location_tracking():
    """人物落地后丢弃跳伞阶段的 SIFT/卡尔曼状态。"""
    return _location_tool().reset_tracking()


def forward_scene(img):
    """开车、跑图入口；输入画面的像素坐标取整，类别转为业务编号。"""
    return _business_detections(_yolo_detector(), img)


def white_angle(img):
    return _tracker().get_angle(img)


def speed(img):
    return _speed_classifier().infer(img)


def house_scene(img):
    return _scene_classifier().predict(img)


@special_timing
def sam3(img, seg_name=None, version=0):
    """按 version 选择本地 EfficientSAM3 实现。"""
    from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.sam3_tiny import segment_sam3
    return segment_sam3(img, seg_name=seg_name, version=version)


def passable(img):
    """PIDNet-S 可通行性分割；输入为 HOS RGB 画面。"""
    from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.passability import passable as segment_passability
    return segment_passability(img)
