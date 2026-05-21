from __future__ import annotations

from typing import Any, Dict, Optional


class QwenRoomToolPolicyAgent:
    """Rule-based tool router that validates Qwen decisions before execution."""

    def __init__(self, config: Dict[str, Any], state_agent: Any):
        self.config = config
        self.state_agent = state_agent
        self.nav_arrival_distance = float(config.get("qwen_nav_arrival_distance") or 1.0)

    def validate_or_override_decision(
        self,
        decision: Dict[str, Any],
        *,
        observation: Dict[str, Any],
        runtime_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return the original decision if valid, otherwise a deterministic override."""

        observation = observation or {}
        runtime_state = runtime_state or {}
        tool_name = decision.get("tool_name")

        if self._should_finish():
            return self._override(decision, "finish_house_search", "已达到目标搜房数量，结束搜房")

        entry_decision = self._entry_phase_decision(observation, runtime_state)
        if entry_decision and tool_name in {
            "select_next_house",
            "navigate_to_house_entry",
            "mark_house_done",
        }:
            return self._override(decision, entry_decision["tool_name"], entry_decision["reason"])

        if tool_name == "select_next_house" and self._has_current_entry(runtime_state):
            return self._override(decision, "navigate_to_house_entry", "已有当前房子和入口，继续导航到入口点")

        if tool_name == "mark_house_done" and not self._entered_current_house():
            if entry_decision:
                return self._override(decision, entry_decision["tool_name"], "未进入过当前房子，先继续入口进门流程")
            if self._has_current_entry(runtime_state):
                return self._override(decision, "navigate_to_house_entry", "未进入过当前房子，继续靠近入口")
            return self._override(decision, "select_next_house", "未进入过当前房子且没有入口目标，先选择房子")

        return decision

    def _entry_phase_decision(
        self,
        observation: Dict[str, Any],
        runtime_state: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        if runtime_state.get("house_scene_name") != "outdoor":
            return None
        if not self._has_current_entry(runtime_state):
            return None

        interactions = runtime_state.get("interactions") or {}
        status = str(runtime_state.get("status") or "")
        if status == "INTERACT" or interactions.get("open_door") or interactions.get("close_door"):
            return {"tool_name": "enter_house", "reason": "入口处已有门交互按钮，执行进门流程"}
        if status == "VISUAL_APPROACH":
            return {"tool_name": "approach_entry_door", "reason": "入口门已锁定，继续靠近入口门"}

        if self._is_entry_arrived(observation, runtime_state):
            if self._has_visible_door(observation):
                return {"tool_name": "approach_entry_door", "reason": "已到入口且画面可见门，优先靠近入口门"}
            return {"tool_name": "scan_entry_door", "reason": "已到入口，禁止继续导航，开始扫描入口门"}

        return None

    def _is_entry_arrived(self, observation: Dict[str, Any], runtime_state: Dict[str, Any]) -> bool:
        if str(runtime_state.get("status") or "") == "SCANNING":
            return True

        distance = self._float_or_none(runtime_state.get("distance_to_entry"))
        if distance is not None and distance <= self.nav_arrival_distance:
            return True

        last_result = self._last_tool_result(observation)
        last_observation = last_result.get("observation") or {}
        if last_observation.get("result_type") == "arrived":
            return True
        if last_observation.get("at_entry") is True:
            return True

        last_distance = self._float_or_none(
            last_observation.get("after_distance")
            or (last_observation.get("state_after") or {}).get("distance_to_entry")
        )
        return last_distance is not None and last_distance <= self.nav_arrival_distance

    def _last_tool_result(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        agent_state = observation.get("agent_state") or {}
        return agent_state.get("last_tool_result") or {}

    def _has_visible_door(self, observation: Dict[str, Any]) -> bool:
        visible_objects = observation.get("visible_objects") or []
        if isinstance(visible_objects, dict):
            counts = visible_objects.get("counts") or {}
            if int(counts.get("door") or 0) > 0:
                return True
            visible_objects = visible_objects.get("objects") or []
        if not isinstance(visible_objects, list):
            return False
        return any(isinstance(item, dict) and item.get("type") == "door" for item in visible_objects)

    def _has_current_entry(self, state: Dict[str, Any]) -> bool:
        return state.get("current_house_id") is not None and state.get("active_entry") is not None

    def _entered_current_house(self) -> bool:
        return bool(getattr(self.state_agent, "entered_current_house", False))

    def _should_finish(self) -> bool:
        should_finish = getattr(self.state_agent, "should_finish", None)
        return bool(should_finish()) if callable(should_finish) else False

    def _override(self, decision: Dict[str, Any], tool_name: str, reason: str) -> Dict[str, Any]:
        if decision.get("tool_name") == tool_name:
            return decision
        return {
            "tool_name": tool_name,
            "args": {},
            "reason": reason,
            "confidence": 1.0,
            "policy_override": True,
            "policy_source": "QwenRoomToolPolicyAgent",
            "original_tool_name": decision.get("tool_name"),
            "original_reason": decision.get("reason", ""),
        }

    def _float_or_none(self, value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
