from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional


class QwenRoomMemoryAgent:
    """Bounded short-term memory for the Qwen room-search loop."""

    def __init__(self, *, window_size: int = 5, max_summary_events: int = 12):
        self.window_size = max(1, int(window_size))
        self.max_summary_events = max(1, int(max_summary_events))
        self.recent_steps: Deque[Dict[str, Any]] = deque(maxlen=self.window_size)
        self.summary_events: Deque[str] = deque(maxlen=self.max_summary_events)

    def reset(self):
        self.recent_steps.clear()
        self.summary_events.clear()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "summary": self._summary_text(),
            "recent_steps": list(self.recent_steps),
            "window_size": self.window_size,
        }

    def record_round(
        self,
        *,
        round_index: int,
        observation: Optional[Dict[str, Any]],
        decision: Dict[str, Any],
        result: Dict[str, Any],
    ):
        observation = observation or {}
        state = observation.get("state") or {}
        result_observation = result.get("observation") or {}
        state_after = result_observation.get("state_after") or {}

        step = {
            "round": int(round_index),
            "before": self._brief_state(state),
            "decision": {
                "tool_name": decision.get("tool_name"),
                "args": decision.get("args") or {},
                "reason": decision.get("reason", ""),
                "confidence": decision.get("confidence"),
            },
            "result": self._brief_result(result),
            "after": self._brief_state(state_after),
        }
        self.recent_steps.append(step)
        self.summary_events.append(self._event_text(step))

    def record_error(self, *, round_index: int, error: str):
        step = {
            "round": int(round_index),
            "decision": {"tool_name": "exception"},
            "result": {"ok": False, "error": str(error)},
        }
        self.recent_steps.append(step)
        self.summary_events.append(f"round {round_index}: exception {error}")

    def _summary_text(self) -> str:
        if not self.summary_events:
            return "暂无历史动作。"
        return " | ".join(self.summary_events)

    def _brief_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "scene": state.get("house_scene_name"),
            "status": state.get("status"),
            "location": state.get("location"),
            "current_house_id": state.get("current_house_id"),
            "has_active_entry": bool(state.get("active_entry") or state.get("has_active_entry")),
            "distance_to_entry": state.get("distance_to_entry"),
            "active_supply_id": state.get("active_supply_id"),
            "active_door_id": state.get("active_door_id"),
            "completed_house_count": state.get("completed_house_count"),
        }

    def _brief_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        observation = result.get("observation") or {}
        return {
            "ok": result.get("ok"),
            "tool_name": result.get("tool_name"),
            "error": result.get("error", ""),
            "result_type": observation.get("result_type"),
            "action": observation.get("action"),
            "step_count": observation.get("step_count"),
            "moved_distance": observation.get("moved_distance"),
            "distance_delta": observation.get("distance_delta"),
            "at_entry": observation.get("at_entry"),
        }

    def _event_text(self, step: Dict[str, Any]) -> str:
        result = step.get("result") or {}
        decision = step.get("decision") or {}
        after = step.get("after") or {}
        parts: List[str] = [
            f"round {step.get('round')}",
            f"tool={decision.get('tool_name')}",
            f"ok={result.get('ok')}",
        ]
        if result.get("result_type"):
            parts.append(f"result={result.get('result_type')}")
        if result.get("distance_delta") is not None:
            parts.append(f"delta={result.get('distance_delta')}")
        if after.get("status"):
            parts.append(f"status={after.get('status')}")
        if result.get("error"):
            parts.append(f"error={result.get('error')}")
        return ", ".join(parts)
