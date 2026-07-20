"""南大最新版房型匹配的独立 HTTP 服务。

该服务必须由南大 demo 的 Python 3.11/3.12 环境启动，避免把新版
transformers/faiss/torch 依赖加载到 auto_game 当前 Python 3.9 进程。

示例：
    python3.11 -m \
      aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.nanda_room_matcher_service \
      --nanda-project ../pubg_test-main --sam3-port 7788
"""

from __future__ import annotations

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, Mapping, Optional

import cv2
import numpy as np


LOGGER = logging.getLogger("NandaRoomMatcherService")
MAX_IMAGE_BYTES = 24 * 1024 * 1024


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _ensure_real_dino_weights(project_root: Path) -> None:
    model_path = (
        project_root
        / "control_proxy"
        / "src"
        / "gametest_proxy"
        / "pubg_room_explore"
        / "img_similarity"
        / "dinov3_vitl16"
        / "model.safetensors"
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"缺少南大 DINOv3 权重: {model_path}")
    with model_path.open("rb") as model_file:
        header = model_file.read(256)
    if model_path.stat().st_size < 1024 or header.startswith(
        b"version https://git-lfs.github.com/spec/"
    ):
        raise RuntimeError(
            "南大 DINOv3 model.safetensors 仍是 Git LFS 指针，"
            "请先取得真实权重后再启动匹配服务"
        )


class NandaLatestRoomMatcherRuntime:
    """持久持有 DINO、房型索引和远程 SAM3 客户端。"""

    def __init__(
        self,
        project_root: Path,
        *,
        sam3_host: str,
        sam3_port: int,
        sam3_timeout_ms: int,
        match_dump_dir: Optional[str] = None,
    ):
        if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
            raise RuntimeError(
                f"南大最新版匹配服务要求 Python 3.11/3.12，当前为 {sys.version.split()[0]}"
            )
        project_root = project_root.expanduser().resolve()
        source_root = project_root / "control_proxy" / "src"
        if not source_root.is_dir():
            raise FileNotFoundError(f"南大项目目录无效，缺少: {source_root}")
        _ensure_real_dino_weights(project_root)
        sys.path.insert(0, str(source_root))

        from gametest_proxy.pubg_room_explore.img_similarity.facade_structure import (
            FacadeStructureExtractor,
        )
        from gametest_proxy.pubg_room_explore.img_similarity.room_library_process import (
            RoomLibrary,
        )
        from gametest_proxy.pubg_room_explore.img_similarity.similarity_utils import (
            ImgSimilarityWithDinoV3,
        )
        from gametest_proxy.pubg_room_explore.sam3.segmenter import Sam3Segmenter

        LOGGER.info("正在加载南大 DINOv3 和房型库，首次启动会比较慢")
        self.segmenter = Sam3Segmenter(
            backend=Sam3Segmenter.BACKEND_REMOTE,
            sam3_host=sam3_host,
            sam3_port=int(sam3_port),
            sam3_timeout_ms=int(sam3_timeout_ms),
        )
        self.segmenter.load_model()
        structure_extractor = FacadeStructureExtractor(segmenter=self.segmenter)
        self.room_library = RoomLibrary(
            extractor=ImgSimilarityWithDinoV3(),
            structure_extractor=structure_extractor,
            match_dump_dir=match_dump_dir,
        )
        self.project_root = project_root
        self._match_lock = threading.Lock()
        LOGGER.info("南大最新版房型匹配服务已就绪")

    def match(self, image_bgr: np.ndarray) -> Mapping[str, Any]:
        with self._match_lock:
            segmentation = self.segmenter.segment_house(image_bgr)
            if segmentation is None:
                return {"status": "no_match", "reason": "sam3_house_segmentation_failed"}

            room_id, replay_path, debug_payload = self.room_library.search_house(
                segmentation.segmented_bgr,
                segmented_house_mask=segmentation.cropped_mask,
                original_house_image=segmentation.cropped_bgr,
                sample_started_at=datetime.now(),
            )
            debug_payload = debug_payload if isinstance(debug_payload, dict) else {}
            decision = debug_payload.get("decision")
            decision = decision if isinstance(decision, dict) else {}
            if room_id is None or replay_path is None:
                return {
                    "status": "no_match",
                    "reason": debug_payload.get("no_match_reason") or "room_threshold_rejected",
                    "metadata": {
                        "decision": decision,
                        "thresholds": debug_payload.get("thresholds"),
                        "top2_margin": debug_payload.get("top2_margin"),
                    },
                }
            if decision.get("replay_allow_actions") is False:
                return {
                    "status": "no_match",
                    "reason": "room_replay_actions_disabled",
                    "metadata": {"decision": decision},
                }

            return {
                "status": "matched",
                "room_id": str(room_id),
                "replay_path": str(Path(replay_path).resolve()),
                "score": decision.get("total_score"),
                "metadata": {
                    "decision": decision,
                    "thresholds": debug_payload.get("thresholds"),
                    "top2_margin": debug_payload.get("top2_margin"),
                    "sam3_score": getattr(segmentation, "score", None),
                    "sam3_inference_ms": getattr(segmentation, "sam3_inference_ms", None),
                },
            }


def _handler_class(runtime: NandaLatestRoomMatcherRuntime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NandaRoomMatcher/1.0"

        def _write_json(self, status_code: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") != "/health":
                self._write_json(404, {"status": "error", "message": "not found"})
                return
            self._write_json(
                200,
                {
                    "status": "ok",
                    "ready": True,
                    "message": "南大最新版 SAM3+DINO 房型匹配服务已就绪",
                },
            )

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/match":
                self._write_json(404, {"status": "error", "message": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                content_length = 0
            if content_length <= 0 or content_length > MAX_IMAGE_BYTES:
                self._write_json(
                    400,
                    {"status": "error", "message": "invalid image payload size"},
                )
                return
            encoded = np.frombuffer(self.rfile.read(content_length), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                self._write_json(
                    400,
                    {"status": "error", "message": "cannot decode JPEG image"},
                )
                return
            try:
                result = runtime.match(image)
            except Exception as exc:
                LOGGER.exception("南大房型匹配请求执行失败")
                self._write_json(
                    500,
                    {
                        "status": "error",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                )
                return
            self._write_json(200, result)

        def log_message(self, fmt: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="南大最新版房型匹配独立服务")
    parser.add_argument(
        "--nanda-project",
        default=os.environ.get("NANDA_PUBG_PROJECT", "../pubg_test-main"),
        help="最新版 pubg_test-main 解压目录",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7789)
    parser.add_argument("--sam3-host", default="127.0.0.1")
    parser.add_argument("--sam3-port", type=int, default=7788)
    parser.add_argument("--sam3-timeout-ms", type=int, default=120000)
    parser.add_argument("--match-dump-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    runtime = NandaLatestRoomMatcherRuntime(
        Path(args.nanda_project),
        sam3_host=args.sam3_host,
        sam3_port=args.sam3_port,
        sam3_timeout_ms=args.sam3_timeout_ms,
        match_dump_dir=args.match_dump_dir,
    )
    server = ThreadingHTTPServer((args.host, args.port), _handler_class(runtime))
    LOGGER.info("监听 http://%s:%d", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        LOGGER.info("收到停止信号")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
