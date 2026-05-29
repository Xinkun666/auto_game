from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps


DEFAULT_CHAR2ID = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "东": 10,
    "南": 11,
    "西": 12,
    "北": 13,
    "None": 14,
}

CHINESE_TO_ANGLE = {
    "北": 0,
    "东": 90,
    "南": 180,
    "西": 270,
    "东北": 45,
    "东南": 135,
    "西南": 225,
    "西北": 315,
}


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 1, feat_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
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
            nn.MaxPool2d(2),
            nn.Conv2d(128, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class DirectionModel(nn.Module):
    def __init__(self, in_ch: int = 1, feat_dim: int = 128) -> None:
        super().__init__()
        self.whole_encoder = SmallCNN(in_ch=in_ch, feat_dim=feat_dim)
        self.crop_encoder = SmallCNN(in_ch=in_ch, feat_dim=feat_dim)
        self.len_head = nn.Linear(feat_dim, 3)
        self.pos_head = nn.Linear(feat_dim, 15)

    def forward(self, whole: torch.Tensor, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, c, h, w = crops.shape
        whole_feat = self.whole_encoder(whole)
        len_logits = self.len_head(whole_feat)
        crop_feat = self.crop_encoder(crops.view(b * n, c, h, w))
        pos_logits = self.pos_head(crop_feat).view(b, n, 15)
        return len_logits, pos_logits


class Get_Direction:
    def __init__(self, model_weight: str | Path, device: str | None = None) -> None:
        self.model_weight = Path(model_weight)
        if not self.model_weight.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_weight}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(self.model_weight, map_location="cpu")
        ckpt_args = ckpt.get("args", {})

        self.whole_w = int(ckpt_args.get("whole_w", 160))
        self.whole_h = int(ckpt_args.get("whole_h", 48))
        self.crop_w = int(ckpt_args.get("crop_w", 64))
        self.crop_h = int(ckpt_args.get("crop_h", 64))
        self.rgb = bool(ckpt_args.get("rgb", False))
        feat_dim = int(ckpt_args.get("feat_dim", 128))

        char2id = ckpt.get("char2id")
        if isinstance(char2id, dict) and "None" in char2id:
            self.char2id = {str(k): int(v) for k, v in char2id.items()}
        else:
            self.char2id = dict(DEFAULT_CHAR2ID)
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.none_id = self.char2id.get("None", 14)

        in_ch = 3 if self.rgb else 1
        self.model = DirectionModel(in_ch=in_ch, feat_dim=feat_dim)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _to_pil(self, img: Any) -> Image.Image:
        if isinstance(img, Image.Image):
            # 强制拷贝到底层 numpy，再重建纯净 PIL.Image，避免继承原图 info/buffer
            arr = np.array(ImageOps.exif_transpose(img).convert("RGB"), copy=True)
            return Image.fromarray(arr, mode="RGB")
        if isinstance(img, (str, Path)):
            with Image.open(img) as im:
                arr = np.array(ImageOps.exif_transpose(im).convert("RGB"), copy=True)
                return Image.fromarray(arr, mode="RGB")
        if isinstance(img, np.ndarray):
            arr = np.array(img, copy=True)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        raise TypeError("img 必须是 PIL.Image / numpy.ndarray / 路径字符串")

    def _preprocess(self, img: Image.Image, size: tuple[int, int]) -> torch.Tensor:
        if self.rgb:
            x = img.convert("RGB")
        else:
            x = img.convert("L")
        x = x.resize(size, Image.BILINEAR)
        arr = np.array(x, dtype=np.float32) / 255.0
        if self.rgb:
            arr = arr.transpose(2, 0, 1)
        else:
            arr = arr[None, :, :]
        return torch.from_numpy(arr)

    def _crop_six_regions(self, img: Image.Image) -> list[Image.Image]:
        w, h = img.size
        p1 = img.crop((int(w * 0.25), 0, int(w * 0.75), h))
        mid = w // 2
        p2 = img.crop((0, 0, mid, h))
        p3 = img.crop((mid, 0, w, h))
        span = int(w * 0.4)
        p4 = img.crop((0, 0, min(w, span), h))
        center_start = max(0, int(w * 0.3))
        center_end = min(w, center_start + span)
        p5 = img.crop((center_start, 0, center_end, h))
        p6 = img.crop((max(0, w - span), 0, w, h))
        return [p1, p2, p3, p4, p5, p6]

    def _decode_label(self, pred_len_id: int, pred_pos_ids: list[int]) -> str:
        if pred_len_id == 0:
            ids = [pred_pos_ids[0]]
        elif pred_len_id == 1:
            ids = [pred_pos_ids[1], pred_pos_ids[2]]
        else:
            ids = [pred_pos_ids[3], pred_pos_ids[4], pred_pos_ids[5]]

        chars: list[str] = []
        for idx in ids:
            if idx == self.none_id:
                continue
            ch = self.id2char.get(idx, "")
            if ch and ch != "None":
                chars.append(ch)
        return "".join(chars)

    def _label_to_angle(self, label: str) -> int:
        if not label:
            return -1

        has_digit = any(ch.isdigit() for ch in label)
        has_cn = any(ch in "东南西北" for ch in label)

        # 混合非法
        if has_digit and has_cn:
            return -1

        # 中文方向
        if has_cn:
            return int(CHINESE_TO_ANGLE.get(label, -1))

        # 数字方向
        if not label.isdigit():
            return -1

        return int(label)

    def get_direction(self, img: Any) -> int:
        pil_img = self._to_pil(img)
        whole = self._preprocess(pil_img, (self.whole_w, self.whole_h))
        crops = [self._preprocess(c, (self.crop_w, self.crop_h)) for c in self._crop_six_regions(pil_img)]
        crops_tensor = torch.stack(crops, dim=0)

        whole = whole.unsqueeze(0).to(self.device)
        crops_tensor = crops_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            len_logits, pos_logits = self.model(whole, crops_tensor)
            pred_len = int(len_logits.argmax(dim=1).item())
            pred_pos = pos_logits.argmax(dim=2)[0].tolist()

        label = self._decode_label(pred_len, pred_pos)
        return self._label_to_angle(label)
