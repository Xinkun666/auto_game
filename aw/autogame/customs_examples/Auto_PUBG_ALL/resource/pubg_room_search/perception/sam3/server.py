from __future__ import annotations
import argparse
import logging
import pickle
import signal
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import zmq

try:
    from . import config as sam3_config
    from .segmenter import Sam3Segmenter
except ImportError:  # Allow running as a standalone script.
    import config as sam3_config
    from segmenter import Sam3Segmenter

DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "sam3.pt"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SAM3-Server")


class Sam3Server:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        segmenter: Sam3Segmenter,
    ):
        self.host = host
        self.port = int(port)
        self.segmenter = segmenter

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.poller = zmq.Poller()
        self.running = False

    @staticmethod
    def _decode_request_image(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode request image.")
        return image

    @staticmethod
    def _encode_png(image: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Failed to encode image to PNG.")
        return buf.tobytes()

    def start(self) -> None:
        self.segmenter.load_model()
        address = f"tcp://{self.host}:{self.port}"
        self.socket.bind(address)
        self.running = True
        logger.info("SAM3 Server started on %s", address)

        self.poller.register(self.socket, zmq.POLLIN)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._run_loop()

    def _signal_handler(self, _, __):
        logger.info("Received shutdown signal")
        self.running = False

    def _run_loop(self):
        while self.running:
            try:
                sockets = dict(self.poller.poll(timeout=500))
                if self.socket in sockets and sockets[self.socket] == zmq.POLLIN:
                    self._process_message()
            except Exception as exc:
                logger.error("Unexpected error in main loop: %s", exc)
        self._cleanup()

    def _process_message(self):
        response = {"success": False, "error": "Internal Server Error"}
        try:
            message = self.socket.recv()
            payload = pickle.loads(message)
            response = self._handle_request(payload)
        except Exception as exc:
            logger.exception("Error processing message: %s", exc)
            response = {"success": False, "error": str(exc)}
        finally:
            self.socket.send(pickle.dumps(response))

    def _handle_request(self, payload: dict) -> dict:
        request_type = payload.get("type")
        if request_type == "health":
            return self._handle_health()
        if request_type == "release_cache":
            return self._handle_release_cache()
        if request_type == "segment_prompt":
            return self._handle_segment_prompt(payload)
        if request_type == "segment_prompt_all":
            return self._handle_segment_prompt_all(payload)
        return {"success": False, "error": f"Unknown type: {request_type}"}

    def _handle_health(self) -> dict:
        data = {
            "status": "healthy",
            "backend": self.segmenter.backend,
            "device": self.segmenter.effective_device,
        }
        return {"success": True, "data": data}

    def _handle_release_cache(self) -> dict:
        self.segmenter.release_accelerator_cache()
        return {"success": True, "data": {"status": "ok"}}

    def _decode_segment_request(
        self, payload: dict
    ) -> tuple[np.ndarray, str, float, str, float, str, int]:
        image_png = payload.get("image_png")
        if not image_png:
            raise ValueError("Missing image_png field")

        decode_start = time.perf_counter()
        image_bgr = self._decode_request_image(image_png)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        image_format = str(payload.get("image_format", "png"))
        image_encoded_bytes = int(payload.get("image_encoded_bytes", len(image_png)))
        prompt = str(payload.get("prompt", "building"))
        min_mask_area_ratio = float(payload.get("min_mask_area_ratio", 0.03))
        mask_fill_strategy = str(
            payload.get("mask_fill_strategy", self.segmenter.mask_fill_strategy)
        )
        if mask_fill_strategy not in Sam3Segmenter.MASK_FILL_STRATEGIES:
            raise ValueError(f"Unsupported mask_fill_strategy: {mask_fill_strategy}")
        return (
            image_bgr,
            prompt,
            min_mask_area_ratio,
            mask_fill_strategy,
            decode_ms,
            image_format,
            image_encoded_bytes,
        )

    def _encode_segment_result(self, result) -> dict:
        return {
            "bbox_xyxy": list(result.bbox_xyxy),
            "score": float(result.score),
            "segmented_bgr_png": self._encode_png(result.segmented_bgr),
            "mask_png": self._encode_png(
                (result.mask.astype(bool) * 255).astype(np.uint8)
            ),
            "cropped_bgr_png": (
                self._encode_png(result.cropped_bgr)
                if result.cropped_bgr is not None
                else None
            ),
            "cropped_mask_png": (
                self._encode_png(
                    (result.cropped_mask.astype(bool) * 255).astype(np.uint8)
                )
                if result.cropped_mask is not None
                else None
            ),
        }

    def _encode_segment_all_result(self, result) -> dict:
        return {
            "bbox_xyxy": list(result.bbox_xyxy),
            "score": float(result.score),
            "mask_png": self._encode_png(
                (result.mask.astype(bool) * 255).astype(np.uint8)
            ),
        }

    @staticmethod
    def _timing_profile(inference_ms: float) -> dict:
        return {"inference_ms": float(inference_ms)}

    def _handle_segment_prompt(self, payload: dict) -> dict:
        request_start = time.perf_counter()
        try:
            (
                image_bgr,
                prompt,
                min_mask_area_ratio,
                mask_fill_strategy,
                decode_ms,
                image_format,
                image_encoded_bytes,
            ) = self._decode_segment_request(payload)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        original_mask_fill_strategy = self.segmenter.mask_fill_strategy
        self.segmenter.set_mask_fill_strategy(mask_fill_strategy)
        try:
            inference_start = time.perf_counter()
            result = self.segmenter.segment_prompt(
                image_bgr=image_bgr,
                prompt=prompt,
                min_mask_area_ratio=min_mask_area_ratio,
            )
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
        finally:
            self.segmenter.set_mask_fill_strategy(original_mask_fill_strategy)
        if result is None:
            total_ms = (time.perf_counter() - request_start) * 1000.0
            logger.info(
                "SAM3 segment_prompt prompt=%r image=%sx%s format=%s bytes=%d result=none decode=%.1fms inference=%.1fms total=%.1fms",
                prompt,
                image_bgr.shape[1],
                image_bgr.shape[0],
                image_format,
                image_encoded_bytes,
                decode_ms,
                inference_ms,
                total_ms,
            )
            return {
                "success": True,
                "data": None,
                "profile": self._timing_profile(inference_ms),
            }

        encode_start = time.perf_counter()
        data = self._encode_segment_result(result)
        encode_ms = (time.perf_counter() - encode_start) * 1000.0
        total_ms = (time.perf_counter() - request_start) * 1000.0
        logger.info(
            "SAM3 segment_prompt prompt=%r image=%sx%s format=%s bytes=%d bbox=%s score=%.4f decode=%.1fms inference=%.1fms encode=%.1fms total=%.1fms",
            prompt,
            image_bgr.shape[1],
            image_bgr.shape[0],
            image_format,
            image_encoded_bytes,
            result.bbox_xyxy,
            result.score,
            decode_ms,
            inference_ms,
            encode_ms,
            total_ms,
        )
        return {
            "success": True,
            "data": data,
            "profile": self._timing_profile(inference_ms),
        }

    def _handle_segment_prompt_all(self, payload: dict) -> dict:
        request_start = time.perf_counter()
        try:
            (
                image_bgr,
                prompt,
                min_mask_area_ratio,
                mask_fill_strategy,
                decode_ms,
                image_format,
                image_encoded_bytes,
            ) = self._decode_segment_request(payload)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        max_masks = int(payload.get("max_masks", 8))

        original_mask_fill_strategy = self.segmenter.mask_fill_strategy
        self.segmenter.set_mask_fill_strategy(mask_fill_strategy)
        try:
            inference_start = time.perf_counter()
            results = self.segmenter.segment_prompt_all(
                image_bgr=image_bgr,
                prompt=prompt,
                max_masks=max_masks,
                min_mask_area_ratio=min_mask_area_ratio,
            )
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
        finally:
            self.segmenter.set_mask_fill_strategy(original_mask_fill_strategy)

        encode_start = time.perf_counter()
        data = [self._encode_segment_all_result(result) for result in results]
        encode_ms = (time.perf_counter() - encode_start) * 1000.0
        total_ms = (time.perf_counter() - request_start) * 1000.0
        logger.info(
            "SAM3 segment_prompt_all prompt=%r image=%sx%s format=%s bytes=%d masks=%d decode=%.1fms inference=%.1fms encode=%.1fms total=%.1fms",
            prompt,
            image_bgr.shape[1],
            image_bgr.shape[0],
            image_format,
            image_encoded_bytes,
            len(results),
            decode_ms,
            inference_ms,
            encode_ms,
            total_ms,
        )
        return {
            "success": True,
            "data": data,
            "profile": self._timing_profile(inference_ms),
        }

    def _cleanup(self):
        logger.info("Service stopped")
        self.socket.close()
        self.context.term()


def _warn_fake_mode_args(args: argparse.Namespace, defaults: dict[str, object]) -> None:
    ignored: list[str] = []
    if args.checkpoint_path != defaults["checkpoint_path"]:
        ignored.append(f"--checkpoint_path={args.checkpoint_path}")
    if args.bpe_path is not None:
        ignored.append(f"--bpe_path={args.bpe_path}")
    if args.load_from_hf is not None:
        ignored.append(f"--load_from_hf={args.load_from_hf}")
    if args.device != defaults["device"]:
        ignored.append(f"--device={args.device}")
    if args.confidence_threshold != defaults["confidence_threshold"]:
        ignored.append(f"--confidence_threshold={args.confidence_threshold}")
    if args.prompt != defaults["prompt"]:
        ignored.append(f"--prompt={args.prompt}")
    if args.mask_fill_strategy != defaults["mask_fill_strategy"]:
        ignored.append(f"--mask_fill_strategy={args.mask_fill_strategy}")

    if ignored:
        logger.warning(
            "Mode 'fake' ignores model-related options: %s",
            ", ".join(ignored),
        )


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=sam3_config.SAM3_PORT)
    parser.add_argument("--mode", type=str, choices=("sam3", "fake"), default="sam3")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT_PATH)
    )
    parser.add_argument("--bpe_path", type=str, default=None)
    parser.add_argument("--load_from_hf", type=str, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=0.4)
    parser.add_argument(
        "--mask_fill_strategy", type=str, default=Sam3Segmenter.MASK_FILL_BLACK
    )
    args = parser.parse_args()

    defaults = {
        "device": "auto",
        "checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
        "confidence_threshold": 0.4,
        "prompt": "building",
        "mask_fill_strategy": Sam3Segmenter.MASK_FILL_BLACK,
    }
    if args.mode == "fake":
        _warn_fake_mode_args(args, defaults)

    backend = (
        Sam3Segmenter.BACKEND_FAKE
        if args.mode == "fake"
        else Sam3Segmenter.BACKEND_LOCAL
    )

    segmenter = Sam3Segmenter(
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        backend=backend,
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
        bpe_path=Path(args.bpe_path) if args.bpe_path else None,
        load_from_hf=_parse_optional_bool(args.load_from_hf),
        mask_fill_strategy=args.mask_fill_strategy,
    )
    server = Sam3Server(host=args.host, port=args.port, segmenter=segmenter)
    server.start()


if __name__ == "__main__":
    main()
