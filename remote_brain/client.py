import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from .protocol import encode_frame_to_base64, json_dumps
except ImportError:
    from protocol import encode_frame_to_base64, json_dumps


class RemoteBrainClient:
    def __init__(
        self,
        server_url: str,
        project_case: str = "Auto_PUBG_ALL",
        game_case: str = "auto_pubg",
        image_format: str = "jpg",
        jpeg_quality: int = 85,
        timeout: float = 8.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.project_case = project_case
        self.game_case = game_case
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.timeout = float(timeout)
        self.session_id = str(uuid.uuid4())
        self._started = False

    def start_session(self, screen: Optional[Tuple[int, int]] = None, current_stage: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "session_id": self.session_id,
            "project_case": self.project_case,
            "game_case": self.game_case,
            "screen": {"width": screen[0], "height": screen[1]} if screen else None,
            "current_stage": current_stage,
        }
        response = self._post_json("/session/start", payload)
        self._started = True
        return response

    def stop_session(self) -> Dict[str, Any]:
        response = self._post_json("/session/stop", {"session_id": self.session_id})
        self._started = False
        return response

    def tick(
        self,
        frame_rgb: np.ndarray,
        current_stage: Optional[str],
        frame_id: int,
        screen: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        if not self._started:
            self.start_session(screen=screen, current_stage=current_stage)

        image_b64 = encode_frame_to_base64(
            frame_rgb,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
        )
        payload = {
            "session_id": self.session_id,
            "frame_id": frame_id,
            "project_case": self.project_case,
            "game_case": self.game_case,
            "current_stage": current_stage,
            "screen": {"width": screen[0], "height": screen[1]} if screen else None,
            "image_format": self.image_format,
            "image": image_b64,
        }
        return self._post_json("/tick", payload)

    def tick_worker(self, worker: Any, execute: bool = True) -> Dict[str, Any]:
        frame = getattr(worker, "frame", None)
        if frame is None:
            raise ValueError("worker.frame is None")

        height, width = frame.shape[:2]
        response = self.tick(
            frame_rgb=frame,
            current_stage=getattr(worker, "current_stage", None),
            frame_id=int(getattr(worker, "frame_index", 0)),
            screen=(width, height),
        )

        worker.stage_info = response.get("stage_info") or {}
        if response.get("current_stage"):
            worker.current_stage = response["current_stage"]

        if execute:
            execute_actions(worker, response.get("actions") or [])
        return response

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.server_url + path,
            data=json_dumps(payload),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote brain HTTP {exc.code}: {body}") from exc


def execute_actions(worker: Any, actions: Iterable[Dict[str, Any]], ignore_refresh_frame: bool = True):
    for action in actions or []:
        action_type = action.get("type")
        args = action.get("args") or []
        kwargs = action.get("kwargs") or {}

        if action_type == "refresh_frame" and ignore_refresh_frame:
            continue
        if action_type == "sleep":
            time.sleep(float(kwargs.get("seconds", args[0] if args else 0.05)))
            continue
        if action_type == "stop":
            worker.stop()
            continue
        if action_type == "change_stage":
            worker.change_stage(*args, **kwargs)
            continue

        method = getattr(worker, action_type, None)
        if method is None:
            print(f"[RemoteBrainClient] 忽略未知动作: {action_type}")
            continue
        method(*args, **kwargs)
