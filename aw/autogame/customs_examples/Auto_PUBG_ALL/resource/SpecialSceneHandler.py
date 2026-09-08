# -*- coding: utf-8 -*-
from aw.autogame.tools.Utils import *

import time
from functools import wraps

dire_tool_ctc = loc_tool = yolo_detector = tracker = speed_cls = scene_cls = None


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
        from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.yolo_detector import YOLO26Detector
        yolo_detector = YOLO26Detector(model_path=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/best.pt')
    return yolo_detector


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
    """
    这是之前 forward_scene 的原始逻辑，已单独迁移到这个函数中保留。
    当前你先不用它，但逻辑没有删除，后续需要时可以继续调用这个函数。

    功能：
    对 ROI 局部图像进行推理，并将结果坐标转换回原始全图绝对坐标（取整）
    """

    YOLO26_CLASSES = {
        0: 'door',
        1: 'object',
        2: 'window',
        3: 'pick_menu',
        4: 'open_door',
        5: 'stair',
        6: 'down_stair',
        7: 'car',
        8: 'house',
        9: 'stone_wall',
        10: 'stump',
        11: 'rock',
        12: 'grass_tuft',
        13: 'fence',
        14: 'water',
        15: 'ditch',
        16: 'unique_construction',
        17: 'wrecked_car',
        18: 'box',
        19: 'sandband_wall',
    }
    # 1. 执行检测
    res = _yolo_detector().infer(img)
    if not res:
        return []

    # 2. 获取相对比例配置
    roi_rect = [0, 0, 1, 1]

    if roi_rect is None:
        print("警告: 未能找到 forward_scene 的配置，返回原始结果")
        return res

    h, w = img.shape[:2]
    x_offset = w * roi_rect[0]
    y_offset = h * roi_rect[1]

    # 3. 遍历结果并转换坐标系
    global_res = []
    for x1, y1, x2, y2, conf, cls_id in res:
        # 局部像素 + 全局偏移 = 全图像素坐标
        gx1 = int(x1 + x_offset)
        gy1 = int(y1 + y_offset)
        gx2 = int(x2 + x_offset)
        gy2 = int(y2 + y_offset)

        global_res.append([gx1, gy1, gx2, gy2, conf, cls_id])

    return global_res


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


@special_timing
def passble(img):
    """标注工具中的 special 区域名；保留该拼写以匹配项目配置。"""
    return passable(img)
