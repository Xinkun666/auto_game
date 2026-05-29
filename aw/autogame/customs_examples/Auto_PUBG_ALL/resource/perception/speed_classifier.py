# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np


# =========================
# 模型结构（必须和训练一致）
# =========================
class TinyDigitCNNv2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================
# 推理类
# =========================
class SpeedClassifier:

    def __init__(self, weight_path, device=None):
        """
        weight_path: best.pt 路径
        """
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 加载 checkpoint
        ckpt = torch.load(weight_path, map_location=self.device)

        # 获取参数
        self.img_size = ckpt.get("img_size", (32, 32))
        self.class_names = ckpt.get("class_names", ["0", "1", "2", "3"])

        # 初始化模型
        self.model = TinyDigitCNNv2(num_classes=len(self.class_names))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # 推理 transform（必须和验证一致）
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def preprocess(self, img):
        """
        img: numpy (cv2 BGR 或 RGB)
        """
        # 如果是 BGR（cv2），转 RGB
        if isinstance(img, np.ndarray):
            arr = np.array(img, copy=True)
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1]  # BGR -> RGB
            img = Image.fromarray(arr)

        img = self.transform(img)
        img = img.unsqueeze(0)  # [1, C, H, W]
        return img.to(self.device)

    @torch.no_grad()
    def infer(self, img):
        """
        输入: img (numpy / PIL)
        输出: int (0~3)
        """
        tensor = self.preprocess(img)

        logits = self.model(tensor)
        pred = logits.argmax(dim=1).item()

        return pred
