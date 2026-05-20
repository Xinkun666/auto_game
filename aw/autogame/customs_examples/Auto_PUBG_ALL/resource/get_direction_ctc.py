from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps


# ============================================================
# CNN + CTC 方向/角度推理文件
#
# 用法示例：
#
# from get_direction_ctc import Get_Direction
#
# direction_model = Get_Direction(
#     model_weight="./ctc_checkpoints/best.pt",
#     device=None,
#     enable_multi_offset=True,
#     confidence_threshold=0.0,
# )
#
# angle = direction_model.get_direction(img)
# detail = direction_model.get_direction_detail(img)
#
# 支持 img 类型：
# - PIL.Image
# - numpy.ndarray
# - 图片路径 str / Path
#
# 返回：
# - get_direction(img): int，合法角度返回 0~359，异常返回 -1
# - get_direction_detail(img): dict，包含 label/confidence/angle/all_candidates
# ============================================================


DEFAULT_CHARS = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9",
    "东", "南", "西", "北",
]

DEFAULT_BLANK_ID = len(DEFAULT_CHARS)

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


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=k,
                padding=p,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DirectionCTCModel(nn.Module):
    """
    与 direction_ctc_main.py 中训练模型保持一致。

    默认输入:
        [B, 1, 48, 160]

    CNN 输出:
        [B, C, H', W']

    展开:
        [B, W', C*H']

    BiLSTM:
        [B, W', hidden*2]

    输出:
        log_probs: [T, B, vocab]
    """

    def __init__(
        self,
        in_ch: int = 1,
        cnn_channels: int = 256,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.15,
        vocab_size: int = 15,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBNReLU(in_ch, 32),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            ConvBNReLU(128, cnn_channels),
            ConvBNReLU(cnn_channels, cnn_channels),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # LazyLinear 用于兼容不同 image_h/image_w 下的 CNN 高度变化。
        # 加载 checkpoint 时，只要先 forward 一次完成初始化，再 load_state_dict 即可。
        self.proj = nn.LazyLinear(rnn_hidden)

        self.rnn = nn.LSTM(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(rnn_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)

        b, c, h, w = feat.shape

        # [B, C, H, W] -> [B, W, C*H]
        seq = feat.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)

        seq = self.proj(seq)
        seq, _ = self.rnn(seq)
        seq = self.dropout(seq)

        logits = self.head(seq)

        # CTC 标准形状: [T, B, V]
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2).contiguous()
        return log_probs


class Get_Direction:
    def __init__(
        self,
        model_weight: str | Path,
        device: str | None = None,
        *,
        enable_multi_offset: bool = True,
        offset_x: int = 4,
        offset_y: int = 3,
        confidence_threshold: float = 0.0,
        strict_load: bool = True,
    ) -> None:
        """
        CNN + CTC 方向/角度识别器。

        Args:
            model_weight:
                训练得到的 best.pt / last.pt。

            device:
                "cuda" / "cpu"。默认自动选择。

            enable_multi_offset:
                是否启用多候选偏移推理。
                开启后会对同一张 ROI 图生成多个轻微平移版本，选择合法且置信度最高的结果。
                因为 CNN+CTC 已经比固定裁剪更稳，这个默认开启只是作为兜底。

            offset_x:
                多候选推理横向偏移像素。

            offset_y:
                多候选推理纵向偏移像素。

            confidence_threshold:
                最终结果最低置信度。
                默认 0.0 表示不额外拦截。
                如果你发现误识别较多，可以尝试设置为 0.35 / 0.5。

            strict_load:
                是否严格加载权重。默认 True。
        """
        self.model_weight = Path(model_weight)

        if not self.model_weight.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_weight}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(self.model_weight, map_location="cpu")
        ckpt_args = ckpt.get("args", {})

        # 兼容训练脚本保存的参数
        self.image_w = int(ckpt_args.get("image_w", 160))
        self.image_h = int(ckpt_args.get("image_h", 48))
        self.rgb = bool(ckpt_args.get("rgb", False))

        self.cnn_channels = int(ckpt_args.get("cnn_channels", 256))
        self.rnn_hidden = int(ckpt_args.get("rnn_hidden", 256))
        self.rnn_layers = int(ckpt_args.get("rnn_layers", 2))
        self.dropout = float(ckpt_args.get("dropout", 0.15))

        chars = ckpt.get("chars", DEFAULT_CHARS)
        if not isinstance(chars, list) or not chars:
            chars = DEFAULT_CHARS

        self.chars = [str(ch) for ch in chars]
        self.char2id = {ch: i for i, ch in enumerate(self.chars)}
        self.id2char = {i: ch for i, ch in enumerate(self.chars)}

        self.blank_id = int(ckpt.get("blank_id", len(self.chars)))
        self.vocab_size = len(self.chars) + 1

        in_ch = 3 if self.rgb else 1

        self.model = DirectionCTCModel(
            in_ch=in_ch,
            cnn_channels=self.cnn_channels,
            rnn_hidden=self.rnn_hidden,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            vocab_size=self.vocab_size,
        )

        # LazyLinear 必须先用 dummy input 跑一次，让 proj 初始化出真实权重形状。
        dummy = torch.zeros(
            1,
            in_ch,
            self.image_h,
            self.image_w,
            dtype=torch.float32,
        )
        with torch.no_grad():
            _ = self.model(dummy)

        self.model.load_state_dict(
            ckpt["model_state_dict"],
            strict=strict_load,
        )

        self.model.to(self.device)
        self.model.eval()

        self.enable_multi_offset = bool(enable_multi_offset)
        self.offset_x = int(offset_x)
        self.offset_y = int(offset_y)
        self.confidence_threshold = float(confidence_threshold)

    def _to_pil(self, img: Any) -> Image.Image:
        if isinstance(img, Image.Image):
            arr = np.array(
                ImageOps.exif_transpose(img).convert("RGB"),
                copy=True,
            )
            return Image.fromarray(arr, mode="RGB")

        if isinstance(img, (str, Path)):
            with Image.open(img) as im:
                arr = np.array(
                    ImageOps.exif_transpose(im).convert("RGB"),
                    copy=True,
                )
            return Image.fromarray(arr, mode="RGB")

        if isinstance(img, np.ndarray):
            arr = np.array(img, copy=True)

            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)

            if arr.ndim != 3:
                raise ValueError(f"numpy 图像维度不合法: {arr.shape}")

            # 如果是 CHW，转成 HWC
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))

            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]

            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)

            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            return Image.fromarray(arr, mode="RGB")

        raise TypeError("img 必须是 PIL.Image / numpy.ndarray / 路径字符串 / Path")

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        if self.rgb:
            x = img.convert("RGB")
        else:
            x = img.convert("L")

        x = x.resize((self.image_w, self.image_h), Image.BILINEAR)

        arr = np.array(x, dtype=np.float32) / 255.0

        if self.rgb:
            arr = arr.transpose(2, 0, 1)
        else:
            arr = arr[None, :, :]

        return torch.from_numpy(arr)

    def _shift_image(self, img: Image.Image, dx: int, dy: int) -> Image.Image:
        """
        对已经裁好的 direction ROI 小图做轻微平移，作为跨设备偏移兜底。

        dx > 0: 内容向右移动
        dx < 0: 内容向左移动
        dy > 0: 内容向下移动
        dy < 0: 内容向上移动

        空出来的区域用四角中位数颜色填充，避免固定黑色不符合真实背景。
        """
        if dx == 0 and dy == 0:
            return img

        img = img.convert("RGB")
        w, h = img.size

        points = [
            img.getpixel((0, 0)),
            img.getpixel((max(0, w - 1), 0)),
            img.getpixel((0, max(0, h - 1))),
            img.getpixel((max(0, w - 1), max(0, h - 1))),
        ]

        fill_arr = np.median(np.array(points, dtype=np.uint8), axis=0).astype(np.uint8)
        fill = tuple(int(v) for v in fill_arr)

        canvas = Image.new("RGB", (w, h), fill)
        canvas.paste(img, (dx, dy))
        return canvas

    def _candidate_offsets(self) -> list[tuple[int, int]]:
        if not self.enable_multi_offset:
            return [(0, 0)]

        ox = max(0, self.offset_x)
        oy = max(0, self.offset_y)

        offsets = [
            (0, 0),
            (-ox, 0),
            (ox, 0),
            (0, -oy),
            (0, oy),
            (-ox, -oy),
            (ox, -oy),
            (-ox, oy),
            (ox, oy),
        ]

        seen: set[tuple[int, int]] = set()
        unique_offsets: list[tuple[int, int]] = []

        for offset in offsets:
            if offset not in seen:
                seen.add(offset)
                unique_offsets.append(offset)

        return unique_offsets

    def _ctc_greedy_decode_with_confidence(
        self,
        log_probs: torch.Tensor,
    ) -> tuple[str, float, list[dict[str, Any]]]:
        """
        Args:
            log_probs: [T, 1, V]

        Returns:
            label:
                CTC 解码后的字符串。

            confidence:
                简单置信度：最终保留字符的概率均值。
                如果没有字符，返回 0.0。

            steps:
                保留字符对应的时间步信息，方便调试。
        """
        probs = log_probs.exp()[:, 0, :]  # [T, V]
        max_probs, max_ids = probs.max(dim=1)

        ids = max_ids.detach().cpu().tolist()
        confs = max_probs.detach().cpu().tolist()

        chars: list[str] = []
        kept_confs: list[float] = []
        steps: list[dict[str, Any]] = []

        last_id: int | None = None

        for t, (idx, conf) in enumerate(zip(ids, confs)):
            idx = int(idx)
            conf = float(conf)

            # CTC 规则：先合并连续重复，再去 blank
            if idx != last_id:
                if idx != self.blank_id and idx in self.id2char:
                    ch = self.id2char[idx]
                    chars.append(ch)
                    kept_confs.append(conf)
                    steps.append({
                        "t": t,
                        "id": idx,
                        "char": ch,
                        "confidence": conf,
                    })

            last_id = idx

        label = "".join(chars)

        if kept_confs:
            confidence = float(np.mean(kept_confs))
        else:
            confidence = 0.0

        return label, confidence, steps

    def _label_to_angle(self, label: str) -> int:
        if not label:
            return -1

        has_digit = any(ch.isdigit() for ch in label)
        has_cn = any(ch in "东南西北" for ch in label)

        # 数字和中文混合，直接判非法。
        if has_digit and has_cn:
            return -1

        # 中文方向。
        if has_cn:
            return int(CHINESE_TO_ANGLE.get(label, -1))

        # 数字方向。
        if not label.isdigit():
            return -1

        try:
            angle = int(label)
        except Exception:
            return -1

        if 0 <= angle <= 359:
            return angle

        return -1

    @torch.no_grad()
    def _predict_once(
        self,
        pil_img: Image.Image,
        *,
        offset: tuple[int, int] = (0, 0),
    ) -> dict[str, Any]:
        x = self._preprocess(pil_img).unsqueeze(0).to(self.device)

        log_probs = self.model(x)

        label, confidence, steps = self._ctc_greedy_decode_with_confidence(log_probs)
        angle = self._label_to_angle(label)

        return {
            "angle": int(angle),
            "label": label,
            "confidence": float(confidence),
            "offset": offset,
            "decode_steps": steps,
        }

    def get_direction_detail(self, img: Any) -> dict[str, Any]:
        """
        返回详细预测结果，方便调试。

        返回示例：
        {
            "angle": 60,
            "label": "60",
            "confidence": 0.93,
            "offset": (0, 0),
            "decode_steps": [...],
            "all_candidates": [...]
        }
        """
        pil_img = self._to_pil(img)
        candidates: list[dict[str, Any]] = []

        for dx, dy in self._candidate_offsets():
            test_img = self._shift_image(pil_img, dx, dy)
            result = self._predict_once(test_img, offset=(dx, dy))
            candidates.append(result)

        valid_candidates = [
            item for item in candidates
            if int(item.get("angle", -1)) != -1
        ]

        if valid_candidates:
            best = max(
                valid_candidates,
                key=lambda item: float(item.get("confidence", 0.0)),
            )
        else:
            if candidates:
                best = max(
                    candidates,
                    key=lambda item: float(item.get("confidence", 0.0)),
                )
                best = dict(best)
                best["angle"] = -1
            else:
                best = {
                    "angle": -1,
                    "label": "",
                    "confidence": 0.0,
                    "offset": (0, 0),
                    "decode_steps": [],
                }

        if float(best.get("confidence", 0.0)) < self.confidence_threshold:
            best = dict(best)
            best["angle"] = -1

        best = dict(best)
        best["all_candidates"] = candidates
        return best

    def get_direction(self, img: Any) -> int:
        """
        保持原接口：只返回角度 int。
        合法返回 0~359，非法返回 -1。
        """
        result = self.get_direction_detail(img)
        return int(result.get("angle", -1))


if __name__ == "__main__":
    # 简单命令行测试：
    # python get_direction_ctc.py --ckpt ./ctc_checkpoints/best.pt --image ./test.png

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable_multi_offset", action="store_true")
    parser.add_argument("--offset_x", type=int, default=4)
    parser.add_argument("--offset_y", type=int, default=3)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    args = parser.parse_args()

    recognizer = Get_Direction(
        model_weight=args.ckpt,
        device=args.device,
        enable_multi_offset=not args.disable_multi_offset,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        confidence_threshold=args.confidence_threshold,
    )

    detail = recognizer.get_direction_detail(args.image)

    print("angle:", detail["angle"])
    print("label:", detail["label"])
    print("confidence:", detail["confidence"])
    print("offset:", detail["offset"])
    print("all_candidates:")
    for item in detail["all_candidates"]:
        print(
            "  ",
            {
                "angle": item["angle"],
                "label": item["label"],
                "confidence": round(float(item["confidence"]), 4),
                "offset": item["offset"],
            },
        )
