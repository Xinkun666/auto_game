from __future__ import annotations

import gc
import inspect
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import cv2
import numpy as np
from PIL import Image

try:
    from . import config as sam3_config
    from .client import Sam3Client
except ImportError:  # Allow running from the standalone sam3/ directory.
    import config as sam3_config
    from client import Sam3Client

logger = logging.getLogger("SAM3-Segmenter")


def bgr_to_model_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr


def bgr_to_pil_rgb(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_model_rgb(image_bgr))


@dataclass
class SegmentationResult:
    segmented_bgr: np.ndarray
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    cropped_bgr: Optional[np.ndarray] = None
    cropped_mask: Optional[np.ndarray] = None
    sam3_inference_ms: Optional[float] = None


class SegmentBackend(Protocol):
    def load_model(self) -> None: ...

    def release_accelerator_cache(self) -> None: ...

    def segment_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        min_mask_area_ratio: float,
    ) -> Optional[SegmentationResult]: ...

    def segment_prompt_all(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        max_masks: int,
        min_mask_area_ratio: float,
    ) -> list[SegmentationResult]: ...


class SegmentationUtils:
    MASK_FILL_BLACK = "black"
    MASK_FILL_MEDIAN = "median"
    MASK_FILL_STRATEGIES = {MASK_FILL_BLACK, MASK_FILL_MEDIAN}

    @staticmethod
    def bbox_from_mask(mask_bool: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask_bool)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Mask is empty and cannot be converted to bbox.")
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        return x1, y1, x2, y2

    @classmethod
    def crop_with_mask(
        cls,
        image_bgr: np.ndarray,
        mask_bool: np.ndarray,
        mask_fill_strategy: str,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        if mask_fill_strategy not in cls.MASK_FILL_STRATEGIES:
            raise ValueError(
                f"Unsupported mask_fill_strategy: {mask_fill_strategy}. "
                f"Expected one of {sorted(cls.MASK_FILL_STRATEGIES)}."
            )
        x1, y1, x2, y2 = cls.bbox_from_mask(mask_bool)
        if mask_fill_strategy == cls.MASK_FILL_BLACK:
            masked = np.zeros_like(image_bgr)
        else:
            fill_color = np.median(image_bgr[mask_bool], axis=0).astype(
                image_bgr.dtype, copy=False
            )
            masked = np.empty_like(image_bgr)
            masked[:] = fill_color
        masked[mask_bool] = image_bgr[mask_bool]
        return masked[y1:y2, x1:x2], (x1, y1, x2, y2)

    @classmethod
    def crop_original_and_mask(
        cls,
        image_bgr: np.ndarray,
        mask_bool: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        x1, y1, x2, y2 = cls.bbox_from_mask(mask_bool)
        cropped_bgr = image_bgr[y1:y2, x1:x2].copy()
        cropped_mask = mask_bool[y1:y2, x1:x2].astype(np.uint8)
        return cropped_bgr, cropped_mask, (x1, y1, x2, y2)

    @classmethod
    def select_best_mask_index(
        cls,
        image_shape: tuple[int, int],
        masks,
        scores,
        min_mask_area_ratio: float,
    ) -> int:
        image_h, image_w = image_shape
        image_area = float(image_h * image_w)
        image_center_x = image_w / 2.0
        image_center_y = image_h / 2.0
        candidates = []

        for idx in range(len(masks)):
            mask_bool = masks[idx, 0].detach().cpu().numpy() > 0
            if not mask_bool.any():
                continue

            x1, y1, x2, y2 = cls.bbox_from_mask(mask_bool)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            center_x_dist = abs(center_x - image_center_x)
            center_y_dist = abs(center_y - image_center_y)
            area = float(mask_bool.sum())
            score = float(scores[idx].item())

            if image_area > 0 and area / image_area < min_mask_area_ratio:
                continue

            candidates.append(
                {
                    "idx": int(idx),
                    "area": area,
                    "center_x_dist": center_x_dist,
                    "center_y_dist": center_y_dist,
                    "score": score,
                }
            )

        if not candidates:
            raise ValueError("No suitable masks available for selection.")

        max_area = max(candidate["area"] for candidate in candidates)
        large_area_threshold = max_area * 0.75
        large_candidates = [
            candidate
            for candidate in candidates
            if candidate["area"] >= large_area_threshold
        ]
        if len(large_candidates) > 1:
            best_candidate = min(
                large_candidates,
                key=lambda candidate: (
                    candidate["center_x_dist"],
                    candidate["center_y_dist"],
                    -candidate["area"],
                    -candidate["score"],
                ),
            )
        else:
            best_candidate = min(
                candidates,
                key=lambda candidate: (
                    -candidate["area"],
                    candidate["center_x_dist"],
                    candidate["center_y_dist"],
                    -candidate["score"],
                ),
            )
        return int(best_candidate["idx"])

    @classmethod
    def mask_result_from_bool(
        cls,
        image_bgr: np.ndarray,
        mask_bool: np.ndarray,
        *,
        score: float,
        mask_fill_strategy: str,
    ) -> SegmentationResult:
        segmented_bgr, bbox_xyxy = cls.crop_with_mask(
            image_bgr,
            mask_bool,
            mask_fill_strategy,
        )
        cropped_bgr, cropped_mask, _ = cls.crop_original_and_mask(image_bgr, mask_bool)
        return SegmentationResult(
            segmented_bgr=segmented_bgr,
            mask=mask_bool.astype(np.uint8),
            bbox_xyxy=bbox_xyxy,
            score=score,
            cropped_bgr=cropped_bgr,
            cropped_mask=cropped_mask,
        )

    @classmethod
    def all_mask_results(
        cls,
        image_bgr: np.ndarray,
        masks,
        scores,
        *,
        max_masks: int,
        min_mask_area_ratio: float,
        mask_fill_strategy: str,
    ) -> list[SegmentationResult]:
        image_h, image_w = image_bgr.shape[:2]
        image_area = float(image_h * image_w)
        candidates: list[tuple[float, float, SegmentationResult]] = []
        for idx in range(len(masks)):
            mask_bool = masks[idx, 0].detach().cpu().numpy() > 0
            if not mask_bool.any():
                continue
            area = float(mask_bool.sum())
            if image_area > 0 and area / image_area < min_mask_area_ratio:
                continue
            score = float(scores[idx].item())
            result = cls.mask_result_from_bool(
                image_bgr,
                mask_bool,
                score=score,
                mask_fill_strategy=mask_fill_strategy,
            )
            candidates.append((-score, -area, result))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in candidates[: max(0, int(max_masks))]]


class FakeBackend(SegmentBackend):
    def load_model(self) -> None:
        return None

    def release_accelerator_cache(self) -> None:
        return None

    def segment_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        min_mask_area_ratio: float,
    ) -> Optional[SegmentationResult]:
        h, w = image_bgr.shape[:2]
        return SegmentationResult(
            segmented_bgr=image_bgr.copy(),
            mask=np.ones((h, w), dtype=np.uint8),
            bbox_xyxy=(0, 0, w, h),
            score=1.0,
            cropped_bgr=image_bgr.copy(),
            cropped_mask=np.ones((h, w), dtype=np.uint8),
        )

    def segment_prompt_all(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        max_masks: int,
        min_mask_area_ratio: float,
    ) -> list[SegmentationResult]:
        result = self.segment_prompt(
            image_bgr,
            prompt=prompt,
            min_mask_area_ratio=min_mask_area_ratio,
        )
        return [result] if result is not None and max_masks > 0 else []


class RemoteBackend(SegmentBackend):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_ms: int,
        mask_fill_strategy: str,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.mask_fill_strategy = mask_fill_strategy
        self._client: Optional[Sam3Client] = None

    def _ensure_client(self) -> Sam3Client:
        if self._client is None:
            _client = Sam3Client(
                host=self.host,
                port=self.port,
                timeout_ms=self.timeout_ms,
            )
            self._client = _client
        else:
            _client = self._client
        return _client

    def load_model(self) -> None:
        health = self._ensure_client().health()
        if health is None:
            logger.warning(
                "SAM3 remote server is unavailable at %s:%s",
                self.host,
                self.port,
            )

    def release_accelerator_cache(self) -> None:
        if not self._ensure_client().release_cache():
            logger.warning(
                "Failed to release remote SAM3 cache at %s:%s",
                self.host,
                self.port,
            )

    def segment_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        min_mask_area_ratio: float,
    ) -> Optional[SegmentationResult]:
        response = self._ensure_client().segment_prompt(
            image_bgr,
            prompt=prompt,
            min_mask_area_ratio=min_mask_area_ratio,
            mask_fill_strategy=self.mask_fill_strategy,
        )
        if response is None:
            return None
        return SegmentationResult(
            segmented_bgr=response["segmented_bgr"],
            mask=(response["mask"] > 0).astype(np.uint8),
            bbox_xyxy=tuple(int(x) for x in response["bbox_xyxy"]),
            score=float(response["score"]),
            cropped_bgr=response.get("cropped_bgr"),
            cropped_mask=(
                (response["cropped_mask"] > 0).astype(np.uint8)
                if response.get("cropped_mask") is not None
                else None
            ),
            sam3_inference_ms=response.get("sam3_inference_ms"),
        )

    def segment_prompt_all(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        max_masks: int,
        min_mask_area_ratio: float,
    ) -> list[SegmentationResult]:
        responses = self._ensure_client().segment_prompt_all(
            image_bgr,
            prompt=prompt,
            max_masks=max_masks,
            min_mask_area_ratio=min_mask_area_ratio,
            mask_fill_strategy=self.mask_fill_strategy,
        )
        results: list[SegmentationResult] = []
        for response in responses:
            results.append(
                SegmentationResult(
                    segmented_bgr=response.get("segmented_bgr", image_bgr),
                    mask=(response["mask"] > 0).astype(np.uint8),
                    bbox_xyxy=tuple(int(x) for x in response["bbox_xyxy"]),
                    score=float(response["score"]),
                    cropped_bgr=response.get("cropped_bgr"),
                    cropped_mask=(
                        (response["cropped_mask"] > 0).astype(np.uint8)
                        if response.get("cropped_mask") is not None
                        else None
                    ),
                    sam3_inference_ms=response.get("sam3_inference_ms"),
                )
            )
        return results


class LocalBackend(SegmentBackend):
    def __init__(
        self,
        *,
        device: str,
        confidence_threshold: float,
        checkpoint_path: str | Path | None,
        bpe_path: str | Path | None,
        load_from_hf: Optional[bool],
        mask_fill_strategy: str,
    ):
        self.device = self._resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.checkpoint_path = (
            str(Path(checkpoint_path).expanduser())
            if checkpoint_path is not None
            else None
        )
        self.bpe_path = str(Path(bpe_path).expanduser()) if bpe_path else None
        self.load_from_hf = (
            load_from_hf if load_from_hf is not None else checkpoint_path is None
        )
        self.mask_fill_strategy = mask_fill_strategy
        self._processor = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"
        except Exception as exc:
            logger.debug(
                "Failed to auto-detect SAM3 device, falling back to CPU: %s", exc
            )
        return "cpu"

    def load_model(self) -> None:
        self._ensure_processor()

    def release_accelerator_cache(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception as exc:
            logger.debug("Failed to release SAM3 accelerator cache: %s", exc)

    def _resolve_bpe_path(self) -> str | None:
        if self.bpe_path:
            resolved = Path(self.bpe_path)
            if resolved.exists():
                return str(resolved)
            raise FileNotFoundError(f"SAM3 BPE vocab file does not exist: {resolved}")

        candidate_dirs: list[Path] = []
        if self.checkpoint_path:
            checkpoint_parent = Path(self.checkpoint_path).expanduser().resolve().parent
            candidate_dirs.extend(
                [
                    checkpoint_parent,
                    checkpoint_parent / "assets",
                    checkpoint_parent / "models",
                ]
            )

        try:
            import sam3

            sam3_dir = Path(sam3.__file__).resolve().parent
            candidate_dirs.extend(
                [sam3_dir, sam3_dir / "assets", sam3_dir.parent / "assets"]
            )
        except Exception:
            pass

        seen = set()
        candidate_names = [
            "bpe_simple_vocab_16e6.txt.gz",
            "bpe_simple_vocab_16e6.txt",
        ]
        for base_dir in candidate_dirs:
            if base_dir in seen:
                continue
            seen.add(base_dir)
            for name in candidate_names:
                candidate = base_dir / name
                if candidate.exists():
                    return str(candidate)
        return None

    def _build_model(self):
        from sam3.model_builder import build_sam3_image_model

        signature = inspect.signature(build_sam3_image_model)
        accepted_names = set(signature.parameters.keys())
        kwargs: dict[str, Any] = {}

        if "device" in accepted_names:
            kwargs["device"] = self.device

        resolved_bpe_path = self._resolve_bpe_path()
        if resolved_bpe_path is not None:
            for arg_name in ("bpe_path", "tokenizer_path", "vocab_path"):
                if arg_name in accepted_names:
                    kwargs[arg_name] = resolved_bpe_path
                    break

        if self.checkpoint_path:
            checkpoint_candidates = [
                "checkpoint_path",
                "ckpt_path",
                "model_path",
                "checkpoint",
                "pretrained_model_path",
            ]
            for arg_name in checkpoint_candidates:
                if arg_name in accepted_names:
                    kwargs[arg_name] = self.checkpoint_path
                    break
            else:
                raise TypeError(
                    "build_sam3_image_model does not expose a known checkpoint path "
                    f"parameter. Supported params: {sorted(accepted_names)}"
                )
        elif "load_from_HF" in accepted_names:
            kwargs["load_from_HF"] = self.load_from_hf
        elif "load_from_hf" in accepted_names:
            kwargs["load_from_hf"] = self.load_from_hf
        else:
            logger.info(
                "SAM3 builder has no HF toggle parameter, calling with default weights"
            )

        logger.info(
            "Initializing SAM3 local backend with device=%s checkpoint=%s bpe=%s hf=%s mask_fill=%s",
            self.device,
            self.checkpoint_path,
            resolved_bpe_path,
            self.load_from_hf,
            self.mask_fill_strategy,
        )
        return build_sam3_image_model(**kwargs)

    def _ensure_processor(self):
        if self._processor is not None:
            return self._processor

        from sam3.model.sam3_image_processor import Sam3Processor

        model = self._build_model()
        self._processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )
        return self._processor

    def segment_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        min_mask_area_ratio: float,
    ) -> Optional[SegmentationResult]:
        processor = self._ensure_processor()
        pil_image = bgr_to_pil_rgb(image_bgr)

        inference_start = time.perf_counter()
        inference_state = processor.set_image(pil_image)
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        masks = output.get("masks")
        scores = output.get("scores")
        if masks is None or scores is None or len(masks) == 0:
            logger.info("SAM3 did not produce any masks for prompt '%s'", prompt)
            return None

        try:
            best_idx = SegmentationUtils.select_best_mask_index(
                image_bgr.shape[:2],
                masks,
                scores,
                min_mask_area_ratio=min_mask_area_ratio,
            )
        except ValueError as exc:
            logger.info(
                "SAM3 did not produce any suitable masks for prompt '%s': %s",
                prompt,
                exc,
            )
            return None

        score = float(scores[best_idx].item())
        mask_bool = masks[best_idx, 0].detach().cpu().numpy() > 0
        if not mask_bool.any():
            logger.info("SAM3 returned an empty best mask for prompt '%s'", prompt)
            return None

        result = SegmentationUtils.mask_result_from_bool(
            image_bgr,
            mask_bool,
            score=score,
            mask_fill_strategy=self.mask_fill_strategy,
        )
        result.sam3_inference_ms = inference_ms
        return result

    def segment_prompt_all(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        max_masks: int,
        min_mask_area_ratio: float,
    ) -> list[SegmentationResult]:
        processor = self._ensure_processor()
        pil_image = bgr_to_pil_rgb(image_bgr)

        inference_start = time.perf_counter()
        inference_state = processor.set_image(pil_image)
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        masks = output.get("masks")
        scores = output.get("scores")
        if masks is None or scores is None or len(masks) == 0:
            logger.info("SAM3 did not produce any masks for prompt '%s'", prompt)
            return []
        results = SegmentationUtils.all_mask_results(
            image_bgr,
            masks,
            scores,
            max_masks=max_masks,
            min_mask_area_ratio=min_mask_area_ratio,
            mask_fill_strategy=self.mask_fill_strategy,
        )
        for result in results:
            result.sam3_inference_ms = inference_ms
        return results


class Sam3Segmenter:
    MASK_FILL_BLACK = SegmentationUtils.MASK_FILL_BLACK
    MASK_FILL_MEDIAN = SegmentationUtils.MASK_FILL_MEDIAN
    MASK_FILL_STRATEGIES = SegmentationUtils.MASK_FILL_STRATEGIES
    BACKEND_FAKE = "fake"
    BACKEND_REMOTE = "sam3"
    BACKEND_LOCAL = "sam3_local"
    BACKENDS = {BACKEND_FAKE, BACKEND_REMOTE, BACKEND_LOCAL}

    def __init__(
        self,
        device: str = "auto",
        confidence_threshold: float = 0.4,
        backend: str = "sam3",
        checkpoint_path: str | Path | None = None,
        bpe_path: str | Path | None = None,
        load_from_hf: Optional[bool] = None,
        mask_fill_strategy: str = MASK_FILL_BLACK,
        sam3_host: Optional[str] = None,
        sam3_port: Optional[int] = None,
        sam3_timeout_ms: Optional[int] = None,
    ):
        if mask_fill_strategy not in self.MASK_FILL_STRATEGIES:
            raise ValueError(
                f"Unsupported mask_fill_strategy: {mask_fill_strategy}. "
                f"Expected one of {sorted(self.MASK_FILL_STRATEGIES)}."
            )
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unsupported SAM3 backend: {backend}. "
                f"Expected one of {sorted(self.BACKENDS)}."
            )

        self.house_prompt = sam3_config.SAM3_HOUSE_PROMPT
        self.door_prompt = sam3_config.SAM3_DOOR_PROMPT
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.backend = backend
        self.mask_fill_strategy = mask_fill_strategy
        self.sam3_host = sam3_host or sam3_config.SAM3_HOST
        self.sam3_port = int(sam3_port or sam3_config.SAM3_PORT)
        self.sam3_timeout_ms = int(sam3_timeout_ms or sam3_config.SAM3_TIMEOUT_MS)
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.load_from_hf = load_from_hf
        self._backend = self._build_backend()

    def _build_backend(self) -> SegmentBackend:
        if self.backend == self.BACKEND_FAKE:
            return FakeBackend()
        if self.backend == self.BACKEND_REMOTE:
            return RemoteBackend(
                host=self.sam3_host,
                port=self.sam3_port,
                timeout_ms=self.sam3_timeout_ms,
                mask_fill_strategy=self.mask_fill_strategy,
            )
        return LocalBackend(
            device=self.device,
            confidence_threshold=self.confidence_threshold,
            checkpoint_path=self.checkpoint_path,
            bpe_path=self.bpe_path,
            load_from_hf=self.load_from_hf,
            mask_fill_strategy=self.mask_fill_strategy,
        )

    def load_model(self) -> None:
        self._backend.load_model()

    def release_accelerator_cache(self) -> None:
        self._backend.release_accelerator_cache()

    @property
    def effective_device(self) -> str:
        return str(getattr(self._backend, "device", self.device))

    def set_mask_fill_strategy(self, mask_fill_strategy: str) -> None:
        if mask_fill_strategy not in self.MASK_FILL_STRATEGIES:
            raise ValueError(
                f"Unsupported mask_fill_strategy: {mask_fill_strategy}. "
                f"Expected one of {sorted(self.MASK_FILL_STRATEGIES)}."
            )
        self.mask_fill_strategy = mask_fill_strategy
        if hasattr(self._backend, "mask_fill_strategy"):
            self._backend.mask_fill_strategy = mask_fill_strategy

    def segment_prompt(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        min_mask_area_ratio: float = 0.03,
    ) -> Optional[SegmentationResult]:
        if image_bgr is None:
            logger.warning(
                "segment_prompt received an empty image for prompt '%s'",
                prompt,
            )
            return None
        return self._backend.segment_prompt(
            image_bgr,
            prompt=prompt,
            min_mask_area_ratio=min_mask_area_ratio,
        )

    def segment_prompt_all(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        max_masks: int = 8,
        min_mask_area_ratio: float = 0.001,
    ) -> list[SegmentationResult]:
        if image_bgr is None:
            logger.warning(
                "segment_prompt_all received an empty image for prompt '%s'",
                prompt,
            )
            return []
        return self._backend.segment_prompt_all(
            image_bgr,
            prompt=prompt,
            max_masks=max_masks,
            min_mask_area_ratio=min_mask_area_ratio,
        )

    def _segment_something(
        self, image_bgr: np.ndarray, prompt: str, min_mask_area_ratio: Optional[float]
    ) -> Optional[SegmentationResult]:
        if min_mask_area_ratio is None:
            return self.segment_prompt(image_bgr=image_bgr, prompt=prompt)
        return self.segment_prompt(
            image_bgr=image_bgr, prompt=prompt, min_mask_area_ratio=min_mask_area_ratio
        )

    def segment_house(
        self, image_bgr: np.ndarray, min_mask_area_ratio: Optional[float] = None
    ) -> Optional[SegmentationResult]:
        return self._segment_something(
            image_bgr=image_bgr,
            prompt=self.house_prompt,
            min_mask_area_ratio=min_mask_area_ratio,
        )

    def segment_door(
        self, image_bgr: np.ndarray, min_mask_area_ratio: Optional[float] = None
    ) -> Optional[SegmentationResult]:
        return self._segment_something(
            image_bgr=image_bgr,
            prompt=self.door_prompt,
            min_mask_area_ratio=min_mask_area_ratio,
        )

    def segment_door_all(
        self,
        image_bgr: np.ndarray,
        max_masks: int = 8,
        min_mask_area_ratio: float = 0.001,
    ) -> list[SegmentationResult]:
        return self.segment_prompt_all(
            image_bgr=image_bgr,
            prompt=self.door_prompt,
            max_masks=max_masks,
            min_mask_area_ratio=min_mask_area_ratio,
        )

    @staticmethod
    def _default_mask_output_path(output_path: Path) -> Path:
        if output_path.stem.endswith("_segment"):
            mask_stem = output_path.stem[: -len("_segment")] + "_mask"
        else:
            mask_stem = output_path.stem + "_mask"
        return output_path.with_name(mask_stem + ".png")

    def save_segmented_house_template(
        self,
        image_path: str | Path,
        output_path: str | Path,
        mask_output_path: str | Path | None = None,
    ) -> Optional[Path]:
        image_path = Path(image_path)
        output_path = Path(output_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(
                f"Failed to read template source image: {image_path}"
            )

        result = self.segment_house(image_bgr)
        if result is None:
            logger.warning("SAM3 could not segment a house from image: %s", image_path)
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), result.segmented_bgr)
        if not ok:
            raise RuntimeError(f"Failed to save segmented template: {output_path}")
        mask_output_path = (
            Path(mask_output_path)
            if mask_output_path is not None
            else self._default_mask_output_path(output_path)
        )
        mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_image = (
            result.cropped_mask if result.cropped_mask is not None else result.mask
        )
        mask_image = (mask_image.astype(bool) * 255).astype(np.uint8)
        ok = cv2.imwrite(str(mask_output_path), mask_image)
        if not ok:
            raise RuntimeError(
                f"Failed to save segmented template mask: {mask_output_path}"
            )
        logger.info(
            "Saved segmented house template to %s and mask to %s with score %.4f",
            output_path,
            mask_output_path,
            result.score,
        )
        return output_path
