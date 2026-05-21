from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional


class QwenRoomMemoryAgent:
    """Bounded short-term memory for the Qwen room-search loop."""

    def __init__(
        self,
        *,
        window_size: int = 5,
        max_summary_events: int = 6,
        max_text_chars: int = 700,
        max_event_chars: int = 180,
        max_field_chars: int = 80,
    ):
        self.window_size = max(1, int(window_size))
        self.max_summary_events = max(1, int(max_summary_events))
        self.max_text_chars = max(120, int(max_text_chars))
        self.max_event_chars = max(80, int(max_event_chars))
        self.max_field_chars = max(20, int(max_field_chars))
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

    def last_event_text(self) -> str:
        if not self.summary_events:
            return "暂无历史动作。"
        return self.summary_events[-1]

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
                "reason": self._truncate(decision.get("reason", "")),
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
            "result": {"ok": False, "error": self._truncate(str(error))},
        }
        self.recent_steps.append(step)
        self.summary_events.append(
            self._truncate(f"第 {round_index} 轮发生异常：{error}。", self.max_event_chars)
        )

    def _summary_text(self) -> str:
        if not self.summary_events:
            return "暂无历史动作。"
        return self._truncate(" ".join(self.summary_events), self.max_text_chars)

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
            "error": self._truncate(result.get("error", "")),
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
        before = step.get("before") or {}
        after = step.get("after") or {}
        tool_name = decision.get("tool_name")
        result_type = result.get("result_type") or ("ok" if result.get("ok") else "failed")

        text = (
            f"第 {step.get('round')} 轮：之前位于{self._scene_text(before)}，"
            f"状态为 {before.get('status') or 'unknown'}，决定调用 {tool_name}。"
            f"工具返回 {result_type}，"
        )
        if result.get("distance_delta") is not None:
            text += f"距离变化 {self._fmt(result.get('distance_delta'))}，"
        if result.get("moved_distance") is not None:
            text += f"移动 {self._fmt(result.get('moved_distance'))}，"
        text += (
            f"之后位于{self._scene_text(after)}，状态为 {after.get('status') or 'unknown'}。"
        )
        if result.get("at_entry"):
            text += " 已到入户点附近，下一步通常应扫描入口门。"
        elif result_type in {"no_progress", "stuck", "timeout"}:
            text += " 该步骤没有顺利推进，下一步应优先恢复、刷新或调整策略。"
        if result.get("error"):
            text += f" 错误：{result.get('error')}。"
        return self._truncate(text, self.max_event_chars)

    def _scene_text(self, state: Dict[str, Any]) -> str:
        scene = state.get("scene") or "unknown"
        if scene == "outdoor":
            return "房间外"
        if scene == "indoor":
            return "房间内"
        if scene == "rooftop":
            return "屋顶"
        return str(scene)

    def _fmt(self, value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    def _truncate(self, value: Any, limit: Optional[int] = None) -> str:
        text = str(value or "")
        limit = int(limit or self.max_field_chars)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."
