# -*- coding: utf-8 -*-
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.direction_service import Get_Direction
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.direction_ctc_service import Get_Direction as Get_Direction_CTC
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.location_service import LocatePoints
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.yolo_detector import YOLO26Detector
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.angle_tracker import AngleTracker
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.speed_classifier import SpeedClassifier
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.perception.scene_predictor import GameSceneClassifier

from aw.autogame.tools.Utils import *

import os
import cv2
from datetime import datetime

dire_tool = Get_Direction(model_weight=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/direction.pt')
dire_tool_ctc = Get_Direction_CTC(model_weight=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/direction_ctc.pt')
loc_tool = LocatePoints()
yolo_detector = YOLO26Detector(model_path=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/best.pt')
tracker = AngleTracker(window_size=30)
speed_cls = SpeedClassifier(weight_path=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/speed_classifier.pt')
scene_cls = GameSceneClassifier(checkpoint_path=r'aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/scene_best_model.pth')

h, w = get_wh()


def direction(img):
    return dire_tool_ctc.get_direction(img)


def location(img):
    return loc_tool.get_location(img)


def save_img_png(img):
    """
    保存输入图像到中文路径：
    D:\\Resource\\数据集\\和平精英yolo数据集汇总\\totals

    保存要求：
    1. 格式为 png
    2. 文件名格式：年月日时分秒-毫秒
       例如：20260421153045-123.png
    3. 使用 cv2.imencode + tofile 方式，确保中文路径可正常保存
    4. 注意通道格式：
       - 如果输入是 3 通道图像，默认按 RGB -> BGR 转换后保存
       - 如果本身是单通道或 4 通道，则直接保存
    """
    save_dir = r'D:\Resource\数据集\和平精英yolo数据集汇总\totals'
    os.makedirs(save_dir, exist_ok=True)

    now = datetime.now()
    filename = now.strftime('%Y%m%d%H%M%S') + '-{:03d}.png'.format(now.microsecond // 1000)
    save_path = os.path.join(save_dir, filename)

    # 处理通道格式
    # 如果是3通道，默认认为输入是RGB，转成BGR再用opencv保存
    if len(img.shape) == 3 and img.shape[2] == 3:
        save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        save_img = img.copy()

    # 使用 imencode + tofile，兼容中文路径
    success, buffer = cv2.imencode('.png', save_img)
    if success:
        buffer.tofile(save_path)
    else:
        print("图片保存失败:", save_path)


def forward_scene_save_image(img):
    """
    当前仅保留保存图像逻辑，不再执行原来的检测与坐标映射。
    调用该函数时，会将输入 img 保存到指定目录。
    """
    save_img_png(img)
    return []


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
    res = yolo_detector.infer(img)
    if not res:
        return []

    # 2. 获取相对比例配置
    roi_rect = [0, 0, 1, 1]

    if roi_rect is None:
        print("警告: 未能找到 forward_scene 的配置，返回原始结果")
        return res

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
    return tracker.get_angle(img)


def speed(img):
    return speed_cls.infer(img)


def house_scene(img):
    return scene_cls.predict(img)
