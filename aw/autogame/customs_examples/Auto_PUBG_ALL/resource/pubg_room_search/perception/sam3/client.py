from __future__ import annotations

import logging
import math
import pickle
from typing import Optional
import cv2
import numpy as np
import zmq

logger = logging.getLogger("SAM3-Client")


class Sam3Client:
    LARGE_IMAGE_THRESHOLD_BYTES = 800 * 1024
    JPEG_QUALITY = 85

    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 30000,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)

        self.context = zmq.Context()
        self.socket = None
        self._connect_socket()
        logger.info("SAM3 client address %s:%s", self.host, self.port)

    def _connect_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

    def _reset_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self._connect_socket()

    @staticmethod
    def _encode_png(image: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Failed to encode image to PNG.")
        return buf.tobytes()

    @classmethod
    def _encode_jpeg(cls, image: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, cls.JPEG_QUALITY],
        )
        if not ok:
            raise ValueError("Failed to encode image to JPEG.")
        return buf.tobytes()

    @classmethod
    def _encode_request_image(cls, image: np.ndarray) -> tuple[bytes, str]:
        png_bytes = cls._encode_png(image)
        if len(png_bytes) <= cls.LARGE_IMAGE_THRESHOLD_BYTES:
            return png_bytes, "png"
        return cls._encode_jpeg(image), "jpeg"

    @staticmethod
    def _decode_png(image_bytes: bytes, flags: int) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, flags)
        if image is None:
            raise ValueError("Failed to decode PNG image.")
        return image

    def _send_request(self, payload: dict) -> dict:
        try:
            self.socket.send(pickle.dumps(payload))
            return pickle.loads(self.socket.recv())
        except zmq.ZMQError:
            self._reset_socket()
            raise

    @staticmethod
    def _process_response_status(response: dict) -> tuple[bool, Optional[str]]:
        success = response.get("success", False)
        if not success:
            return False, response.get("error", "unknown error")
        return True, None

    @staticmethod
    def _extract_inference_ms(response: dict) -> Optional[float]:
        profile = response.get("profile")
        if not isinstance(profile, dict):
            return None
        value = profile.get("inference_ms")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        inference_ms = float(value)
        if not math.isfinite(inference_ms) or inference_ms < 0.0:
            return None
        return inference_ms

    def health(self) -> Optional[dict]:
        try:
            response = self._send_request({"type": "health"})
        except Exception as exc:
            logger.warning("SAM3 health request failed: %s", exc)
            return None
        success, err = self._process_response_status(response)
        if not success:
            logger.warning("SAM3 health response failed: %s", err)
            return None
        return response.get("data")

    def release_cache(self) -> bool:
        try:
            response = self._send_request({"type": "release_cache"})
        except Exception as exc:
            logger.warning("SAM3 release_cache request failed: %s", exc)
            return False
        success, err = self._process_response_status(response)
        if not success:
            logger.warning("SAM3 release_cache response failed: %s", err)
            return False
        return True

    def segment_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        min_mask_area_ratio: float,
        mask_fill_strategy: str,
    ) -> Optional[dict]:
        image_bytes, image_format = self._encode_request_image(image_bgr)
        request = {
            "type": "segment_prompt",
            "image_png": image_bytes,
            "image_format": image_format,
            "image_encoded_bytes": len(image_bytes),
            "prompt": prompt,
            "min_mask_area_ratio": float(min_mask_area_ratio),
            "mask_fill_strategy": mask_fill_strategy,
        }
        try:
            response = self._send_request(request)
        except Exception as exc:
            logger.warning("SAM3 segment_prompt request failed: %s", exc)
            return None
        success, err = self._process_response_status(response)
        if not success:
            logger.warning("SAM3 segment_prompt response failed: %s", err)
            return None

        data = response.get("data")
        if data is None:
            return None
        inference_ms = self._extract_inference_ms(response)

        result: dict[str, object] = {
            "bbox_xyxy": tuple(data["bbox_xyxy"]),
            "score": float(data["score"]),
            "segmented_bgr": self._decode_png(
                data["segmented_bgr_png"], cv2.IMREAD_COLOR
            ),
            "mask": self._decode_png(data["mask_png"], cv2.IMREAD_GRAYSCALE),
        }
        cropped_bgr_png = data.get("cropped_bgr_png")
        cropped_mask_png = data.get("cropped_mask_png")
        if cropped_bgr_png is not None:
            result["cropped_bgr"] = self._decode_png(cropped_bgr_png, cv2.IMREAD_COLOR)
        if cropped_mask_png is not None:
            result["cropped_mask"] = self._decode_png(
                cropped_mask_png, cv2.IMREAD_GRAYSCALE
            )
        if inference_ms is not None:
            result["sam3_inference_ms"] = inference_ms
        return result

    def segment_prompt_all(
        self,
        image_bgr: np.ndarray,
        *,
        prompt: str,
        max_masks: int,
        min_mask_area_ratio: float,
        mask_fill_strategy: str,
    ) -> list[dict]:
        image_bytes, image_format = self._encode_request_image(image_bgr)
        request = {
            "type": "segment_prompt_all",
            "image_png": image_bytes,
            "image_format": image_format,
            "image_encoded_bytes": len(image_bytes),
            "prompt": prompt,
            "max_masks": int(max_masks),
            "min_mask_area_ratio": float(min_mask_area_ratio),
            "mask_fill_strategy": mask_fill_strategy,
        }
        try:
            response = self._send_request(request)
        except Exception as exc:
            logger.warning("SAM3 segment_prompt_all request failed: %s", exc)
            return []
        success, err = self._process_response_status(response)
        if not success:
            logger.warning("SAM3 segment_prompt_all response failed: %s", err)
            return []

        data = response.get("data")
        if not data:
            return []
        inference_ms = self._extract_inference_ms(response)

        results: list[dict] = []
        for item in data:
            result: dict[str, object] = {
                "bbox_xyxy": tuple(item["bbox_xyxy"]),
                "score": float(item["score"]),
                "mask": self._decode_png(item["mask_png"], cv2.IMREAD_GRAYSCALE),
            }
            segmented_bgr_png = item.get("segmented_bgr_png")
            cropped_bgr_png = item.get("cropped_bgr_png")
            cropped_mask_png = item.get("cropped_mask_png")
            if segmented_bgr_png is not None:
                result["segmented_bgr"] = self._decode_png(
                    segmented_bgr_png, cv2.IMREAD_COLOR
                )
            if cropped_bgr_png is not None:
                result["cropped_bgr"] = self._decode_png(
                    cropped_bgr_png, cv2.IMREAD_COLOR
                )
            if cropped_mask_png is not None:
                result["cropped_mask"] = self._decode_png(
                    cropped_mask_png, cv2.IMREAD_GRAYSCALE
                )
            if inference_ms is not None:
                result["sam3_inference_ms"] = inference_ms
            results.append(result)
        return results

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()
