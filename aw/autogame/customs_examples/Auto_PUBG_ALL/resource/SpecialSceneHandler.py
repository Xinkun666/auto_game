# -*- coding: utf-8 -*-
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.get_direction import Get_Direction
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.get_location2 import LocatePoints
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.yolo26 import YOLO26Detector
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.yolov5 import YOLOv5TorchScript
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.angle_tracker import AngleTracker
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.speed_classifier import SpeedClassifier

from aw.autogame.tools.Utils import *

dire_tool = Get_Direction(model_weight=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/direction.pt')
loc_tool = LocatePoints()
yolo26_detector = YOLO26Detector(model_path=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/yolo26.pt')
yolo5_detector = YOLOv5TorchScript()
tracker = AngleTracker(window_size=30)
speed_cls = SpeedClassifier(weight_path=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/speed_classifier.pt')

h, w = get_wh()

def direction(img):
    return dire_tool.get_direction(img)

def location(img):
    return loc_tool.get_location(img)

def forward_scene(img):
    """
    对 ROI 局部图像进行推理，并将结果坐标转换回原始全图绝对坐标（取整）
    """
    # 1. 执行检测
    res = yolo26_detector.infer(img)
    if not res:
        return []

    # 2. 获取相对比例配置 (解决 Key 前缀动态变化问题)
    roi_rect =  [0,0,1,1]

    if roi_rect is None:
        print("警告: 未能找到 forward_scene 的配置，返回原始结果")
        return res

    x_offset = w * roi_rect[0]
    y_offset = h * roi_rect[1]

    # 4. 遍历结果并转换坐标系
    global_res = []
    for x1, y1, x2, y2, conf, cls_id in res:
        # 局部像素 + 全局偏移 = 全图像素坐标
        # 使用 int() 确保输出为整数像素
        gx1 = int(x1 + x_offset)
        gy1 = int(y1 + y_offset)
        gx2 = int(x2 + x_offset)
        gy2 = int(y2 + y_offset)

        global_res.append([gx1, gy1, gx2, gy2, conf, cls_id])

    return global_res

def white_angle(img):
    return tracker.get_angle(img)

def speed(img):
    return speed_cls.infer(img)

def forward_scene2(img):
    res = yolo5_detector.infer(img)
    if not res:
        return []

    # 2. 获取相对比例配置 (解决 Key 前缀动态变化问题)
    roi_rect = [0.279700,0.282600,0.720300,0.617500]

    if roi_rect is None:
        print("警告: 未能找到 forward_scene 的配置，返回原始结果")
        return res

    x_offset = w * roi_rect[0]
    y_offset = h * roi_rect[1]

    # 4. 遍历结果并转换坐标系
    global_res = []
    for x1, y1, x2, y2, conf, cls_id in res:
        # 局部像素 + 全局偏移 = 全图像素坐标
        # 使用 int() 确保输出为整数像素
        gx1 = int(x1 + x_offset)
        gy1 = int(y1 + y_offset)
        gx2 = int(x2 + x_offset)
        gy2 = int(y2 + y_offset)

        global_res.append([gx1, gy1, gx2, gy2, conf, cls_id])

    return global_res

def house_scene(img):
    print('process house_scene function')

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
    }

    # 1. 执行 YOLO26 检测
    res = yolo26_detector.infer(img)
    if not res:
        print("未检测到目标")
        return []

    # 2. 获取 ROI 配置
    roi_rect = [0, 0, 1, 1]
    x_offset = w * roi_rect[0]
    y_offset = h * roi_rect[1]

    # 3. 坐标系转换并添加类别名称
    global_res = []
    for x1, y1, x2, y2, conf, cls_id in res:
        gx1 = int(x1 + x_offset)
        gy1 = int(y1 + y_offset)
        gx2 = int(x2 + x_offset)
        gy2 = int(y2 + y_offset)
        cls_name = YOLO26_CLASSES.get(int(cls_id), f'unknown_{cls_id}')
        global_res.append([gx1, gy1, gx2, gy2, conf, cls_id, cls_name])

    print(f"YOLO26 检测到 {len(global_res)} 个目标:")
    for item in global_res:
        print(f"  - {item[6]}: {item[:4]} conf={item[4]:.2f}")

    return global_res
