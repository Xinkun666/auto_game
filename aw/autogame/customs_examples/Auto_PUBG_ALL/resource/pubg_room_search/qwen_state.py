from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


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
        self.state.last_decision = decision

    def record_tool_result(self, tool_name: str, result: Dict[str, Any]):
        self.state.last_tool_result = result
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

    def record_error(self):
        self.state.consecutive_errors += 1

    def too_many_errors(self) -> bool:
        return self.state.consecutive_errors >= self.state.max_errors
