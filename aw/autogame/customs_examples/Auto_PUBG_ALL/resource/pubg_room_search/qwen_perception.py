from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.qwen_tools import (
    QwenHouseSearchTools,
)


@dataclass
class QwenRoomPerceptionSnapshot:
    observation: Dict[str, Any]
    frame_data_url: Optional[str] = None


class QwenRoomPerceptionAgent:
    """Build the structured observation sent to the controller/model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.jpeg_quality = int(config.get("qwen_jpeg_quality") or 80)

    def observe(
        self,
        worker: Any,
        tools: QwenHouseSearchTools,
        *,
        task: str,
        state_snapshot: Dict[str, Any],
        memory_snapshot: Optional[Dict[str, Any]] = None,
    ) -> QwenRoomPerceptionSnapshot:
        observation = tools.build_observation(task=task)
        observation["agent_state"] = state_snapshot
        if memory_snapshot is not None:
            observation["agent_memory"] = memory_snapshot
        return QwenRoomPerceptionSnapshot(
            observation=observation,
            frame_data_url=self._encode_frame_data_url(getattr(worker, "frame", None)),
        )

    def _encode_frame_data_url(self, frame) -> Optional[str]:
        if frame is None or not hasattr(frame, "shape"):
            return None
        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, buffer = cv2.imencode(
                ".jpg",
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                return None
            image_b64 = base64.b64encode(buffer).decode("ascii")
            return f"data:image/jpeg;base64,{image_b64}"
        except Exception:
            return None
