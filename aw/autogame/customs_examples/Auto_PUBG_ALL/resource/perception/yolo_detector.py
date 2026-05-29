import cv2
import time
import numpy as np
from ultralytics import YOLO


class YOLO26Detector:
    def __init__(self, model_path, conf_thres=0.7, iou_thres=0.5, device=None):
        """
        初始化检测器，接口对齐 YOLOv5TorchScript
        """
        print(f"正在加载 YOLO26 模型: {model_path} ...")
        # 如果 device 为 'cpu'，则强制使用 CPU，否则自动使用 GPU
        self.model = YOLO(model_path)

        # 预热模型
        self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        print("模型预热结束。")

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

        # 为了保持与旧代码一致，这里可以手动定义 names 或者直接从模型获取
        # 如果你之前进行了类别过滤，这里的 self.names 会自动对应新训练的 0-4 索引
        self.names = self.model.names

    def infer(self, frame):
        """
        对单帧图像执行推理，返回格式: [[x1, y1, x2, y2, conf, cls_id], ...]
        对齐旧版 YOLOv5TorchScript 的接口
        """
        if frame is None:
            return []

        # 执行推理
        # Ultralytics 内部会自动处理处理 preprocess (resize, letterbox, BGR2RGB, 归一化)
        results = self.model(frame,
                             conf=self.conf_thres,
                             iou=self.iou_thres,
                             device=self.device,
                             verbose=False)

        result = results[0]
        output = []

        # 如果有检测结果
        if result.boxes:
            # .xyxy 是坐标, .conf 是置信度, .cls 是类别索引
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box
                # 构造与旧版一致的列表格式
                output.append([
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(conf),
                    int(cls)
                ])

        return output

    def detect_and_plot(self, frame):
        """
        调试用：同时返回结果和画好框的图片
        """
        results = self.model(frame, conf=self.conf_thres, verbose=False)
        return self.infer(frame), results[0].plot()

    def set_conf(self, new_conf):
        self.conf_thres = new_conf