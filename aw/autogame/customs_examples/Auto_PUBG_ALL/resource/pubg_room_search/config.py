import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from aw.autogame.tools.Utils import _read_autogame_config


DEFAULT_CONFIG = {
    "enabled": False,
    "embedded_enabled": True,
    "embedded_first": True,
    "external_pubg_test_enabled": False,
    "mode": "mixed_yolo",
    "pubg_test_path": "",
    "yolo_host": "localhost",
    "yolo_port": 6666,
    "sam3_host": "localhost",
    "sam3_port": 12345,
    "door_calibration_backend": "sam3",
    "room_match_dump_dir": "",
    "fallback_to_legacy": True,
    "finish_stage": "跑图阶段",
    "frame_color": "rgb",
    "frame_refresh_interval_sec": 0.12,
    "frame_refresh_mode": "worker_refresh",
    "embedded_door_refine_attempts": 3,
    "embedded_door_center_tolerance_px": 80,
    "embedded_door_view_scale": 0.33,
    "embedded_enter_after_refine": False,
    "embedded_enter_move_duration_ms": 360,
    "embedded_enter_wait_ms": 900,
    "embedded_allow_legacy_fallback": True,
    "replay_skip_idle": True,
    "replay_time_scale": 1.0,
    "replay_min_wait_sec": 0.001,
    "replay_preserve_duplicate_timestamps": True,
    "replay_queue_stop_timeout_sec": 3.0,
    "qwen_agent_enabled": True,
    "qwen_base_url": "http://10.41.182.148:8000/v1",
    "qwen_model": "qwen2.5-vl-7b",
    "qwen_api_key": "EMPTY",
    "qwen_max_tokens": 384,
    "qwen_timeout_sec": 20.0,
    "qwen_max_houses": 5,
    "qwen_fallback_to_legacy": True,
    "qwen_max_consecutive_errors": 3,
    "qwen_jpeg_quality": 80,
    "qwen_control_system_prompt": "",
    "qwen_trace_enabled": True,
    "qwen_trace_prompt": True,
    "qwen_memory_enabled": True,
    "qwen_memory_window_size": 5,
    "qwen_memory_summary_events": 6,
    "qwen_memory_max_text_chars": 700,
    "qwen_memory_max_event_chars": 180,
    "qwen_memory_max_field_chars": 80,
    "qwen_state_max_field_chars": 120,
    "qwen_state_step_samples": 2,
    "qwen_http_error_body_chars": 1200,
    "qwen_nav_arrival_distance": 1.0,
    "qwen_nav_precise_distance": 5.0,
    "qwen_nav_max_steps": 8,
    "qwen_nav_timeout_sec": 12.0,
    "qwen_nav_no_progress_limit": 2,
    "move_finger_id": 0,
    "view_finger_id": 1,
    "button_finger_start": 2,
    "move_radius_px": 300,
    "view_default_center": [0.75, 0.53],
    "move_default_center": [0.1965, 0.7563],
    "button_mapping": {
        "jump": "跳跃",
        "door": "开门",
        "pick_btn": "拾取",
        "map": "地图",
        "attack": "攻击",
    },
}


def get_pubg_room_search_config() -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    raw = _read_autogame_config().get("pubg_room_search", {})
    if isinstance(raw, dict):
        config.update(raw)
        button_mapping = dict(DEFAULT_CONFIG["button_mapping"])
        button_mapping.update(raw.get("button_mapping") or {})
        config["button_mapping"] = button_mapping

    env_enabled = os.environ.get("AUTOGAME_PUBG_ROOM_SEARCH")
    if env_enabled is not None:
        config["enabled"] = env_enabled.strip().lower() in {"1", "true", "yes", "on"}

    env_path = os.environ.get("PUBG_TEST_PATH")
    if env_path:
        config["pubg_test_path"] = env_path

    return config


def resolve_pubg_test_path(config: Optional[Dict[str, Any]] = None) -> Path:
    config = config or get_pubg_room_search_config()
    configured = str(config.get("pubg_test_path") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    auto_game_root = Path(__file__).resolve().parents[4]
    return (auto_game_root.parent / "pubg_test").resolve()


def ensure_pubg_test_import_path(config: Optional[Dict[str, Any]] = None) -> Path:
    pubg_test_root = resolve_pubg_test_path(config)
    proxy_src = pubg_test_root / "control_proxy" / "src"
    if not proxy_src.exists():
        raise FileNotFoundError(f"未找到 pubg_test control_proxy/src: {proxy_src}")

    proxy_src_text = str(proxy_src)
    if proxy_src_text not in sys.path:
        sys.path.insert(0, proxy_src_text)
    return pubg_test_root
