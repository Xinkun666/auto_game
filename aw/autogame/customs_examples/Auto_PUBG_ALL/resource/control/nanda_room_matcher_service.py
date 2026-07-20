"""auto_game 内置的南大 DINOv3 + MLP 房型配准 HTTP 服务。

该服务使用独立 Python 3.11/3.12 进程，避免把新版
transformers/faiss/torch 依赖加载到 auto_game 当前 Python 3.9 进程。
服务不启动也不连接 SAM3；它只接收搜房阶段 ``get_info("sam3")``
产生的房屋分割图、mask 和原始裁剪。

默认直接从 auto_game 的 ``resource/weights/nanda_room_matcher`` 和
``resource/nanda_room_library`` 加载资产，不再依赖一份可导入的南大源码。
``--nanda-project`` 仅作为迁移/对照旧目录的兼容选项。
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.nanda_room_matcher_runtime import (
    IntegratedNandaRoomMatcher,
    NandaMatcherAssetPaths,
)


LOGGER = logging.getLogger("NandaRoomMatcherService")
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
INPUT_CONTRACT = "sam3_special_area"


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


def _decode_special_area_payload(
    body: bytes,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as archive:
            segmented_bgr = np.asarray(archive["segmented_bgr"]).copy()
            cropped_mask = np.asarray(archive["cropped_mask"]).copy()
            cropped_bgr = np.asarray(archive["cropped_bgr"]).copy()
            crop_xyxy = np.asarray(
                archive.get("crop_xyxy", np.zeros(4, dtype=np.int32))
            ).reshape(-1)
            sam3_score = np.asarray(
                archive.get("sam3_score", np.zeros(1, dtype=np.float32))
            ).reshape(-1)
            sam3_inference_ms = np.asarray(
                archive.get("sam3_inference_ms", np.zeros(1, dtype=np.float32))
            ).reshape(-1)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"无法解码 sam3 special_area NPZ: {exc}") from exc

    if segmented_bgr.ndim != 3 or segmented_bgr.shape[2] != 3:
        raise ValueError("segmented_bgr 必须是 HxWx3 图像")
    if cropped_bgr.shape != segmented_bgr.shape:
        raise ValueError("cropped_bgr 与 segmented_bgr 尺寸不一致")
    if cropped_mask.ndim == 3:
        cropped_mask = np.any(cropped_mask != 0, axis=2).astype(np.uint8) * 255
    if cropped_mask.ndim != 2 or cropped_mask.shape != segmented_bgr.shape[:2]:
        raise ValueError("cropped_mask 与门面图像尺寸不一致")
    if not np.any(cropped_mask):
        raise ValueError("cropped_mask 为空")
    if segmented_bgr.dtype != np.uint8 or cropped_bgr.dtype != np.uint8:
        raise ValueError("门面图像必须是 uint8")
    cropped_mask = (cropped_mask > 0).astype(np.uint8) * 255
    metadata = {
        "crop_xyxy": [int(value) for value in crop_xyxy[:4]],
        "sam3_score": float(sam3_score[0]) if sam3_score.size else None,
        "sam3_inference_ms": (
            float(sam3_inference_ms[0]) if sam3_inference_ms.size else None
        ),
        "mask_area_ratio": float(np.count_nonzero(cropped_mask))
        / float(cropped_mask.size),
    }
    return segmented_bgr, cropped_mask, cropped_bgr, metadata


class NandaLatestRoomMatcherRuntime:
    """持久持有 DINO 与房型索引，只消费预分割门面。"""

    def __init__(
        self,
        asset_paths: NandaMatcherAssetPaths,
        *,
        device: Optional[str] = None,
        match_dump_dir: Optional[str] = None,
    ):
        if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
            raise RuntimeError(
                f"南大最新版匹配服务要求 Python 3.11/3.12，当前为 {sys.version.split()[0]}"
            )
        if match_dump_dir:
            LOGGER.warning("内置匹配器暂不写南大 match_dump，已忽略: %s", match_dump_dir)
        self.asset_paths = asset_paths.resolved()
        LOGGER.info(
            "正在从 auto_game 资产目录加载 DINOv3、MLP 和房型库: %s", self.asset_paths.to_jsonable(),
        )
        self.room_matcher = IntegratedNandaRoomMatcher(self.asset_paths, device=device,)
        self.input_contract = INPUT_CONTRACT
        self._match_lock = threading.Lock()
        LOGGER.info("auto_game 内置南大 DINOv3/MLP 房型配准服务已就绪")

    def _display_replay_path(self, replay_path: Path) -> str:
        resolved = replay_path.resolve()
        try:
            return str(resolved.relative_to(self.asset_paths.room_library_path))
        except ValueError:
            return resolved.name

    def health_payload(self) -> Mapping[str, Any]:
        return self.room_matcher.health_payload()

    def match(
        self,
        segmented_bgr: np.ndarray,
        cropped_mask: np.ndarray,
        cropped_bgr: np.ndarray,
        special_area_metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        with self._match_lock:
            room_id, replay_path, debug_payload = self.room_matcher.match(
                segmented_bgr, cropped_mask, cropped_bgr,
            )
            debug_payload = debug_payload if isinstance(debug_payload, dict) else {}
            decision = debug_payload.get("decision")
            decision = decision if isinstance(decision, dict) else {}
            if room_id is None or replay_path is None:
                return {
                    "status": "no_match",
                    "reason": debug_payload.get("no_match_reason")
                    or "room_threshold_rejected",
                    "metadata": {
                        "decision": decision,
                        "thresholds": debug_payload.get("thresholds"),
                        "top2_margin": debug_payload.get("top2_margin"),
                        "top_candidates": debug_payload.get("top_candidates"),
                        "matcher_elapsed_ms": debug_payload.get("elapsed_ms"),
                        "special_area": special_area_metadata,
                    },
                }
            if decision.get("replay_allow_actions") is False:
                return {
                    "status": "no_match",
                    "reason": "room_replay_actions_disabled",
                    "metadata": {"decision": decision},
                }

            replay_file = Path(replay_path).resolve()
            try:
                replay_steps = json.loads(replay_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"无法读取南大回放 DSL: {replay_file}: {exc}") from exc
            if not isinstance(replay_steps, list):
                raise ValueError(f"南大回放 DSL 不是列表: {replay_file}")

            return {
                "status": "matched",
                "room_id": str(room_id),
                "replay_path": self._display_replay_path(replay_file),
                "replay_steps": replay_steps,
                "score": decision.get("total_score"),
                "metadata": {
                    "decision": decision,
                    "thresholds": debug_payload.get("thresholds"),
                    "top2_margin": debug_payload.get("top2_margin"),
                    "top_candidates": debug_payload.get("top_candidates"),
                    "matcher_elapsed_ms": debug_payload.get("elapsed_ms"),
                    "special_area": special_area_metadata,
                    "input_contract": self.input_contract,
                    "structure_mode": "disabled_zero_vector_no_extra_sam3",
                },
            }


def _handler_class(runtime: NandaLatestRoomMatcherRuntime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NandaRoomMatcher/2.0"

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
                    "input_contract": runtime.input_contract,
                    "runtime": runtime.health_payload(),
                    "message": (
                        "auto_game 内置南大 DINOv3/MLP 房型配准已就绪，"
                        "输入使用 auto_game sam3 special_area"
                    ),
                },
            )

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/match":
                self._write_json(404, {"status": "error", "message": "not found"})
                return
            content_type = str(self.headers.get("Content-Type") or "").split(";", 1)[0]
            input_contract = str(self.headers.get("X-Nanda-Input-Contract") or "")
            if content_type != "application/x-npz" or input_contract != INPUT_CONTRACT:
                self._write_json(
                    415,
                    {
                        "status": "error",
                        "message": (
                            f"expected application/x-npz + {INPUT_CONTRACT}, "
                            f"got {content_type or 'unknown'} + {input_contract or 'unknown'}"
                        ),
                    },
                )
                return
            try:
                content_length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                content_length = 0
            if content_length <= 0 or content_length > MAX_PAYLOAD_BYTES:
                self._write_json(
                    400, {"status": "error", "message": "invalid NPZ payload size"},
                )
                return
            try:
                decoded = _decode_special_area_payload(self.rfile.read(content_length))
                result = runtime.match(*decoded)
            except ValueError as exc:
                self._write_json(400, {"status": "error", "message": str(exc)})
                return
            except Exception as exc:
                LOGGER.exception("南大房型匹配请求执行失败")
                self._write_json(
                    500,
                    {"status": "error", "message": f"{type(exc).__name__}: {exc}",},
                )
                return
            self._write_json(200, result)

        def log_message(self, fmt: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def _parse_args(argv=None):
    defaults = NandaMatcherAssetPaths.auto_game_defaults()
    parser = argparse.ArgumentParser(description="auto_game 内置南大房型配准服务")
    parser.add_argument(
        "--nanda-project",
        default=os.environ.get("NANDA_PUBG_PROJECT") or None,
        help="兼容入口：从最新版 pubg_test-main 目录读取三类资产",
    )
    parser.add_argument(
        "--dino-model-dir",
        default=(
            os.environ.get("AUTOGAME_NANDA_DINO_MODEL_DIR")
            or str(defaults.dino_model_dir)
        ),
        help="包含 config.json/preprocessor_config.json/model.safetensors 的目录",
    )
    parser.add_argument(
        "--mlp-model-path",
        default=(
            os.environ.get("AUTOGAME_NANDA_MLP_MODEL_PATH")
            or str(defaults.mlp_model_path)
        ),
        help="南大 rgb_mlp_struct_v7.pkl 路径",
    )
    parser.add_argument(
        "--room-library",
        default=(
            os.environ.get("AUTOGAME_NANDA_ROOM_LIBRARY")
            or str(defaults.room_library_path)
        ),
        help="包含 rooms/ 的南大 room_library 目录",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7789)
    parser.add_argument("--device", default=os.environ.get("AUTOGAME_NANDA_DEVICE"))
    parser.add_argument("--match-dump-dir", default=None)
    return parser.parse_args(argv)


def _resolve_asset_paths(args) -> NandaMatcherAssetPaths:
    if args.nanda_project:
        legacy = NandaMatcherAssetPaths.from_nanda_project(Path(args.nanda_project))
        defaults = NandaMatcherAssetPaths.auto_game_defaults()
        # 只有未显式覆盖集成默认值时，--nanda-project 才接管对应路径。
        dino_model_dir = (
            legacy.dino_model_dir
            if Path(args.dino_model_dir).expanduser() == defaults.dino_model_dir
            else Path(args.dino_model_dir)
        )
        mlp_model_path = (
            legacy.mlp_model_path
            if Path(args.mlp_model_path).expanduser() == defaults.mlp_model_path
            else Path(args.mlp_model_path)
        )
        room_library_path = (
            legacy.room_library_path
            if Path(args.room_library).expanduser() == defaults.room_library_path
            else Path(args.room_library)
        )
        return NandaMatcherAssetPaths(
            dino_model_dir=dino_model_dir,
            mlp_model_path=mlp_model_path,
            room_library_path=room_library_path,
        )
    return NandaMatcherAssetPaths(
        dino_model_dir=Path(args.dino_model_dir),
        mlp_model_path=Path(args.mlp_model_path),
        room_library_path=Path(args.room_library),
    )


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    runtime = NandaLatestRoomMatcherRuntime(
        _resolve_asset_paths(args),
        device=args.device,
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
