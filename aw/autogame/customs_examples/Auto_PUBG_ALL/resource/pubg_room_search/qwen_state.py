from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QwenRoomAgentState:
    max_houses: int
    max_errors: int
    entered_current_house: bool = False
    consecutive_errors: int = 0
    last_decision: Optional[Dict[str, Any]] = None
    last_tool_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QwenRoomStateAgent:
    """Deterministic state owner for the Qwen room-search flow."""

    def __init__(self, searcher: Any, *, max_houses: int, max_errors: int):
        self.searcher = searcher
        config = getattr(searcher, "qwen_room_search_config", {}) or {}
        self.max_field_chars = max(20, int(config.get("qwen_state_max_field_chars") or 120))
        self.max_step_samples = max(1, int(config.get("qwen_state_step_samples") or 2))
        self.state = QwenRoomAgentState(
            max_houses=int(max_houses),
            max_errors=max(1, int(max_errors)),
        )

    def reset(self):
        self.state.entered_current_house = False
        self.state.consecutive_errors = 0
        self.state.last_decision = None
        self.state.last_tool_result = None

    def snapshot(self) -> Dict[str, Any]:
        data = self.state.to_dict()
        data["completed_house_count"] = self.completed_house_count
        data["should_finish"] = self.should_finish()
        return data

    @property
    def completed_house_count(self) -> int:
        return int(getattr(self.searcher, "searching_number", 0))

    @property
    def entered_current_house(self) -> bool:
        return bool(self.state.entered_current_house)

    def should_finish(self) -> bool:
        return self.completed_house_count >= self.state.max_houses

    def sync_from_observation(self, observation: Dict[str, Any]):
        if observation.get("state", {}).get("house_scene_name") == "indoor":
            self.state.entered_current_house = True

    def record_decision(self, decision: Dict[str, Any]):
        self.state.last_decision = self._compact_decision(decision)

    def record_tool_result(self, tool_name: str, result: Dict[str, Any]):
        observation = result.get("observation") or {}
        if tool_name == "select_next_house" and result.get("ok"):
            self.state.entered_current_house = False
        elif tool_name in {"enter_house", "enter_door"}:
            if observation.get("entered") or observation.get("house_scene") == "indoor":
                self.state.entered_current_house = True
        elif tool_name == "mark_house_done" and result.get("ok"):
            self.state.entered_current_house = False

        if result.get("ok"):
            self.state.consecutive_errors = 0
        else:
            self.state.consecutive_errors += 1
        self.state.last_tool_result = self._compact_tool_result(result)

    def record_error(self):
        self.state.consecutive_errors += 1

    def too_many_errors(self) -> bool:
        return self.state.consecutive_errors >= self.state.max_errors

    def _compact_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "tool_name": decision.get("tool_name"),
            "args": self._compact_value(decision.get("args") or {}),
            "reason": self._truncate(decision.get("reason", "")),
            "confidence": decision.get("confidence"),
        }

    def _compact_tool_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        observation = result.get("observation") or {}
        compact_observation = self._compact_observation(observation)
        return {
            "ok": result.get("ok"),
            "tool_name": result.get("tool_name"),
            "error": self._truncate(result.get("error", "")),
            "observation": compact_observation,
        }

    def _compact_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        keep_keys = {
            "result_type",
            "action",
            "status",
            "stage",
            "found",
            "aligned",
            "entered",
            "picked",
            "interacted",
            "button",
            "selected",
            "target_type",
            "at_entry",
            "before_distance",
            "after_distance",
            "distance_delta",
            "moved_distance",
            "step_count",
            "duration_ms",
            "wait_sec",
            "stuck",
            "angle",
            "house_scene",
        }
        compact = {
            key: self._compact_value(observation.get(key))
            for key in keep_keys
            if key in observation
        }
        if "state_after" in observation:
            compact["state_after"] = self._brief_state(observation.get("state_after") or {})
        if "state" in observation:
            compact["state"] = self._brief_state(observation.get("state") or {})
        if "room_memory" in observation:
            compact["room_memory"] = self._brief_room_memory(observation.get("room_memory") or {})
        if "steps" in observation:
            compact["steps"] = self._brief_steps(observation.get("steps") or [])
        return compact

    def _brief_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "house_scene_name": state.get("house_scene_name"),
            "status": state.get("status"),
            "location": state.get("location"),
            "current_house_id": state.get("current_house_id"),
            "has_active_entry": bool(state.get("active_entry") or state.get("has_active_entry")),
            "distance_to_entry": state.get("distance_to_entry"),
            "active_supply_id": state.get("active_supply_id"),
            "active_door_id": state.get("active_door_id"),
            "completed_house_count": state.get("completed_house_count"),
        }

    def _brief_room_memory(self, room_memory: Dict[str, Any]) -> Dict[str, Any]:
        supplies = room_memory.get("supplies") or []
        doors = room_memory.get("doors") or []
        return {
            "supply_count": len(supplies) if isinstance(supplies, list) else 0,
            "door_count": len(doors) if isinstance(doors, list) else 0,
            "visited_supply_ids": room_memory.get("visited_supply_ids") or [],
            "visited_door_ids": room_memory.get("visited_door_ids") or [],
        }

    def _brief_steps(self, steps: List[Any]) -> Dict[str, Any]:
        if not isinstance(steps, list):
            return {"count": 0, "samples": []}
        samples = []
        if steps:
            sample_items = steps[-self.max_step_samples :]
            for item in sample_items:
                samples.append(self._compact_value(item))
        return {"count": len(steps), "samples": samples}

    def _compact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._truncate(value)
        if isinstance(value, dict):
            return {
                self._truncate(key): self._compact_value(item)
                for key, item in value.items()
                if key not in {"frame_data_url", "image", "image_url"}
            }
        if isinstance(value, list):
            limit = max(1, self.max_step_samples)
            return [self._compact_value(item) for item in value[:limit]]
        if isinstance(value, tuple):
            return [self._compact_value(item) for item in value[: self.max_step_samples]]
        return value

    def _truncate(self, value: Any) -> str:
        text = str(value or "")
        if len(text) <= self.max_field_chars:
            return text
        return text[: max(0, self.max_field_chars - 3)] + "..."
