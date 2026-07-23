"""Local, in-process EfficientSAM3 inference for the ``sam3`` special area.

The model is loaded lazily and cached for the lifetime of the automation
process.  No socket, host, port, or sidecar server is involved.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


SAM3_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = SAM3_DIR / "runtime"
DEFAULT_CHECKPOINT_PATH = (
    SAM3_DIR.parent.parent
    / "weights"
    / "sam3_tiny"
    / "efficientsam3_tinyvit.pt"
)
DEFAULT_BPE_PATH = RUNTIME_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz"
MINIMUM_PYTHON_VERSION = (3, 10)


def _read_float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字，当前值为 {value!r}") from exc


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class LocalSam3Segmenter:
    """Owns one EfficientSAM3 model and performs serialized local inference."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        *,
        prompt: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        min_mask_area_ratio: Optional[float] = None,
        device: Optional[str] = None,
        runtime_loader: Optional[Callable[[], Tuple[Any, str]]] = None,
    ) -> None:
        configured_checkpoint = os.environ.get("AUTOGAME_SAM3_CHECKPOINT", "").strip()
        self.checkpoint_path = Path(
            configured_checkpoint or checkpoint_path or DEFAULT_CHECKPOINT_PATH
        ).expanduser()
        self.prompt = (prompt or os.environ.get("AUTOGAME_SAM3_PROMPT", "building")).strip()
        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else _read_float_env("AUTOGAME_SAM3_CONFIDENCE_THRESHOLD", 0.4)
        )
        self.min_mask_area_ratio = (
            float(min_mask_area_ratio)
            if min_mask_area_ratio is not None
            else _read_float_env("AUTOGAME_SAM3_MIN_MASK_AREA_RATIO", 0.001)
        )
        self.requested_device = (
            device or os.environ.get("AUTOGAME_SAM3_DEVICE", "auto")
        ).strip().lower()
        self._runtime_loader = runtime_loader or self._load_local_runtime
        self._processor = None
        self._effective_device = "unloaded"
        self._load_error: Optional[BaseException] = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        if not self.prompt:
            raise ValueError("SAM3 prompt 不能为空")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("SAM3 confidence_threshold 必须位于 [0, 1]")
        if not 0.0 <= self.min_mask_area_ratio <= 1.0:
            raise ValueError("SAM3 min_mask_area_ratio 必须位于 [0, 1]")

    @property
    def effective_device(self) -> str:
        return self._effective_device

    def _ensure_vendored_runtime(self) -> None:
        model_builder = RUNTIME_DIR / "sam3" / "model_builder.py"
        if not model_builder.is_file():
            raise FileNotFoundError(f"SAM3 本地源码缺失: {model_builder}")

        existing_module = sys.modules.get("sam3")
        if existing_module is not None:
            module_file = getattr(existing_module, "__file__", None)
            if module_file:
                resolved_module = Path(module_file).resolve()
                try:
                    resolved_module.relative_to(RUNTIME_DIR.resolve())
                except ValueError as exc:
                    raise RuntimeError(
                        "当前进程已经加载了其他 sam3 包，无法切换到 Auto_PUBG_ALL "
                        f"内置版本: {resolved_module}"
                    ) from exc

        runtime_text = str(RUNTIME_DIR)
        if runtime_text not in sys.path:
            sys.path.insert(0, runtime_text)

    def _resolve_device(self, torch_module: Any) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        if torch_module.cuda.is_available():
            return "cuda"
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    def _load_local_runtime(self) -> Tuple[Any, str]:
        if sys.version_info < MINIMUM_PYTHON_VERSION:
            raise RuntimeError(
                "Auto_PUBG_ALL 的本地 SAM3 需要 Python 3.10 或更高版本；"
                f"当前解释器为 {sys.version_info.major}.{sys.version_info.minor}。"
                "请用同一个 Python 3.10+ 环境启动 launcher.py。"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "SAM3 权重不存在，请将 efficientsam3_tinyvit.pt 放到: "
                f"{self.checkpoint_path}"
            )
        if not DEFAULT_BPE_PATH.is_file():
            raise FileNotFoundError(f"SAM3 tokenizer 文件不存在: {DEFAULT_BPE_PATH}")

        self._ensure_vendored_runtime()
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_efficientsam3_image_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SAM3 本地依赖未安装，请安装 perception/sam3_tiny/requirements.txt"
            ) from exc

        device = self._resolve_device(torch)
        model = build_efficientsam3_image_model(
            bpe_path=str(DEFAULT_BPE_PATH),
            device=device,
            eval_mode=True,
            checkpoint_path=str(self.checkpoint_path),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
            backbone_type="tinyvit",
            model_name="11m",
            text_encoder_type="MobileCLIP-S0",
            text_encoder_context_length=16,
        )
        model = model.to(device)
        model.eval()
        processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=self.confidence_threshold,
        )
        return processor, device

    def _ensure_loaded(self) -> Any:
        if self._processor is not None:
            return self._processor
        if self._load_error is not None:
            raise RuntimeError(f"SAM3 模型此前加载失败: {self._load_error}") from self._load_error

        with self._load_lock:
            if self._processor is not None:
                return self._processor
            if self._load_error is not None:
                raise RuntimeError(
                    f"SAM3 模型此前加载失败: {self._load_error}"
                ) from self._load_error
            try:
                processor, device = self._runtime_loader()
                self._processor = processor
                self._effective_device = str(device)
            except BaseException as exc:
                self._load_error = exc
                raise
        return self._processor

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> List[int]:
        ys, xs = np.where(mask)
        return [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]

    @staticmethod
    def _mask_contours(mask: np.ndarray, max_contours: int = 16) -> List[List[List[int]]]:
        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        result: List[List[List[int]]] = []
        for contour in contours[:max_contours]:
            if len(contour) < 3:
                continue
            epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            points = [[int(x), int(y)] for x, y in simplified.reshape(-1, 2)]
            if len(points) >= 3:
                result.append(points)
        return result

    def _select_best_mask(
        self,
        masks: Any,
        scores: Any,
        *,
        prompt: str,
    ) -> Optional[Dict[str, Any]]:
        candidates = self._collect_mask_candidates(masks, scores)
        return self._select_best_candidate(candidates, prompt=prompt)

    def _collect_mask_candidates(
        self,
        masks: Any,
        scores: Any,
        *,
        min_mask_area_ratio: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        score_values = _to_numpy(scores).reshape(-1)
        candidates: List[Dict[str, Any]] = []
        minimum_area_ratio = (
            self.min_mask_area_ratio
            if min_mask_area_ratio is None
            else float(min_mask_area_ratio)
        )
        for index in range(min(len(score_values), len(masks))):
            mask_value = masks[index]
            mask_array = _to_numpy(mask_value)
            while mask_array.ndim > 2:
                mask_array = mask_array[0]
            mask = mask_array > 0
            if not mask.any():
                continue
            area_ratio = float(mask.mean())
            if area_ratio < minimum_area_ratio:
                continue
            bbox = self._bbox_from_mask(mask)
            bbox_width = max(1, bbox[2] - bbox[0])
            bbox_height = max(1, bbox[3] - bbox[1])
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            candidates.append(
                {
                    "mask": mask,
                    "bbox": bbox,
                    "score": float(score_values[index]),
                    "area": int(mask.sum()),
                    "area_ratio": area_ratio,
                    "aspect_ratio": float(bbox_height) / float(bbox_width),
                    "center_distance": (
                        abs(center_x - mask.shape[1] / 2.0),
                        abs(center_y - mask.shape[0] / 2.0),
                    ),
                }
            )
        return candidates

    @staticmethod
    def _select_best_candidate(
        candidates: List[Dict[str, Any]],
        *,
        prompt: str,
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None

        if str(prompt).strip().casefold() == "door frame":
            vertical_candidates = [
                item for item in candidates if item["aspect_ratio"] >= 1.0
            ]
            if not vertical_candidates:
                return None
            regular_doors = [
                item for item in vertical_candidates if item["aspect_ratio"] <= 4.0
            ]
            door_candidates = regular_doors or vertical_candidates
            return min(
                door_candidates,
                key=lambda item: (
                    item["center_distance"][0],
                    item["center_distance"][1],
                    -item["area"],
                    -item["score"],
                ),
            )

        max_area = max(item["area"] for item in candidates)
        large_candidates = [
            item for item in candidates if item["area"] >= max_area * 0.75
        ]
        if len(large_candidates) > 1:
            return min(
                large_candidates,
                key=lambda item: (
                    item["center_distance"][0],
                    item["center_distance"][1],
                    -item["area"],
                    -item["score"],
                ),
            )
        return min(
            candidates,
            key=lambda item: (
                -item["area"],
                item["center_distance"][0],
                item["center_distance"][1],
                -item["score"],
            ),
        )

    def _infer_candidates(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: Optional[str],
        min_mask_area_ratio: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
        ):
            raise ValueError("SAM3 输入必须是 HxWx3 的 BGR numpy 图像")
        if image_bgr.size == 0:
            raise ValueError("SAM3 输入区域为空")

        active_prompt = (prompt or self.prompt).strip()
        processor = self._ensure_loaded()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        started_at = time.perf_counter()
        with self._inference_lock:
            state = processor.set_image(image_pil)
            output = processor.set_text_prompt(state=state, prompt=active_prompt)
        inference_ms = round((time.perf_counter() - started_at) * 1000.0, 3)

        masks = output.get("masks") if isinstance(output, dict) else None
        scores = output.get("scores") if isinstance(output, dict) else None
        if masks is None or scores is None or len(masks) == 0:
            return active_prompt, [], inference_ms
        candidates = self._collect_mask_candidates(
            masks,
            scores,
            min_mask_area_ratio=min_mask_area_ratio,
        )
        return active_prompt, candidates, inference_ms

    @classmethod
    def _candidate_visualization(
        cls,
        candidate: Dict[str, Any],
        *,
        prompt: str,
        index: int,
    ) -> Dict[str, Any]:
        bbox = candidate["bbox"]
        return {
            "type": "sam3_mask",
            "label": f"sam3:{prompt}:{index}",
            "bbox_xyxy": bbox,
            "contours": cls._mask_contours(candidate["mask"]),
            "score": candidate["score"],
            "coord": "local",
            "color_bgr": [0, 165, 255],
            "bbox_color_bgr": [0, 255, 0],
            "alpha": 0.35,
        }

    def infer_all_masks(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        max_masks: int = 12,
        min_mask_area_ratio: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return numpy-mask candidates for local template preprocessing."""
        _, candidates, _ = self._infer_candidates(
            image_bgr,
            prompt=prompt,
            min_mask_area_ratio=min_mask_area_ratio,
        )
        ordered = sorted(
            candidates,
            key=lambda item: (-item["score"], -item["area"]),
        )
        return ordered[: max(0, int(max_masks))]

    def infer(self, image_bgr: np.ndarray, prompt: Optional[str] = None) -> Dict[str, Any]:
        active_prompt, candidates, inference_ms = self._infer_candidates(
            image_bgr,
            prompt=prompt,
        )
        selected = self._select_best_candidate(candidates, prompt=active_prompt)
        if selected is None:
            return {
                "found": False,
                "prompt": active_prompt,
                "score": None,
                "bbox_xyxy_local": None,
                "mask_area_ratio": 0.0,
                "sam3_inference_ms": inference_ms,
                "device": self.effective_device,
                "__visualizations__": [],
            }

        bbox = selected["bbox"]
        normalized_prompt = active_prompt.casefold()
        if normalized_prompt in {"door frame", "window"}:
            ordered_candidates = [selected] + [
                candidate
                for candidate in sorted(
                    candidates,
                    key=lambda item: (-item["score"], -item["area"]),
                )
                if candidate is not selected
            ]
            ordered_candidates = ordered_candidates[:12]
        else:
            ordered_candidates = [selected]
        return {
            "found": True,
            "prompt": active_prompt,
            "score": selected["score"],
            "bbox_xyxy_local": bbox,
            "mask_area_ratio": selected["area_ratio"],
            "mask_count": len(ordered_candidates),
            "sam3_inference_ms": inference_ms,
            "device": self.effective_device,
            "__visualizations__": [
                self._candidate_visualization(
                    candidate,
                    prompt=active_prompt,
                    index=index,
                )
                for index, candidate in enumerate(ordered_candidates)
            ],
        }


_SAM3_SEGMENTER: Optional[LocalSam3Segmenter] = None
_SAM3_SINGLETON_LOCK = threading.Lock()


def get_sam3_segmenter() -> LocalSam3Segmenter:
    global _SAM3_SEGMENTER
    if _SAM3_SEGMENTER is None:
        with _SAM3_SINGLETON_LOCK:
            if _SAM3_SEGMENTER is None:
                _SAM3_SEGMENTER = LocalSam3Segmenter()
    return _SAM3_SEGMENTER


def segment_sam3(
    image_bgr: np.ndarray,
    seg_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Segment ``seg_name`` while reusing the process-wide SAM3 model."""
    return get_sam3_segmenter().infer(image_bgr, prompt=seg_name)
