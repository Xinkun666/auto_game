from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.qwen_perception import (
    QwenRoomPerceptionSnapshot,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.qwen_prompts import (
    CONTROL_SYSTEM_PROMPT,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.qwen_state import (
    QwenRoomStateAgent,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.qwen_tools import (
    QwenHouseSearchTools,
)


class QwenRoomControlAgent:
    """Choose the next whitelisted tool from perception + deterministic state."""

    DEFAULT_SYSTEM_PROMPT = CONTROL_SYSTEM_PROMPT

    def __init__(self, searcher: Any, config: Dict[str, Any], state_agent: QwenRoomStateAgent):
        self.searcher = searcher
        self.config = config
        self.state_agent = state_agent
        self.base_url = str(config.get("qwen_base_url") or "http://10.41.182.148:8000/v1").rstrip("/")
        self.model = str(config.get("qwen_model") or "qwen2.5-vl-7b")
        self.api_key = str(config.get("qwen_api_key") or "EMPTY")
        self.max_tokens = min(128, int(config.get("qwen_max_tokens") or 128))
        self.timeout_sec = float(config.get("qwen_timeout_sec") or 20.0)
        self.http_error_body_chars = max(120, int(config.get("qwen_http_error_body_chars") or 1200))
        self.prompt_observation_max_chars = max(
            800,
            int(config.get("qwen_prompt_observation_max_chars") or 2200),
        )
        self.prompt_summary_max_chars = max(120, int(config.get("qwen_prompt_summary_max_chars") or 420))
        self.prompt_recent_steps = max(0, int(config.get("qwen_prompt_recent_steps") or 2))
        self.prompt_visible_object_limit = max(0, int(config.get("qwen_prompt_visible_object_limit") or 5))
        self.trace_payload_size = bool(config.get("qwen_trace_payload_size", False))
        self.system_prompt = str(config.get("qwen_control_system_prompt") or self.DEFAULT_SYSTEM_PROMPT)

    def decide(
        self,
        snapshot: QwenRoomPerceptionSnapshot,
        tools: QwenHouseSearchTools,
    ) -> Dict[str, Any]:
        decision = self._model_decision(snapshot)
        if decision is None:
            decision = self.fallback_decision(tools)
        decision = self._normalize_decision(decision)
        return self._guard_decision(decision, tools)

    def prompt_for_trace(self, snapshot: QwenRoomPerceptionSnapshot) -> Dict[str, Any]:
        return self._build_payload(snapshot, trace=True)

    def fallback_decision(self, tools: QwenHouseSearchTools) -> Dict[str, Any]:
        state = tools.get_game_state().observation
        interactions = state.get("interactions") or {}
        house_scene = state.get("house_scene_name")
        status = str(state.get("status") or "")
        room_memory = state.get("room_memory") or {}

        if self.state_agent.should_finish():
            return {"tool_name": "finish_house_search", "args": {}}
        if self._is_stuck(tools):
            return {"tool_name": "recover_from_stuck", "args": {}}
        if (
            house_scene == "outdoor"
            and state.get("current_house_id") is not None
            and self.state_agent.entered_current_house
        ):
            return {"tool_name": "mark_house_done", "args": {}}
        if house_scene == "outdoor":
            if state.get("active_entry") is None:
                return {"tool_name": "select_next_house", "args": {}}
            if status == "SCANNING":
                return {"tool_name": "scan_entry_door", "args": {}}
            if status == "VISUAL_APPROACH":
                return {"tool_name": "approach_entry_door", "args": {}}
            if status == "INTERACT" or interactions.get("open_door") or interactions.get("close_door"):
                return {"tool_name": "enter_house", "args": {}}
            return {"tool_name": "navigate_to_house_entry", "args": {}}

        if house_scene == "indoor":
            if interactions.get("pickup_first"):
                return {"tool_name": "pickup_item", "args": {}}
            if status in {"SCAN_ROOM", "IDLE"} or not room_memory.get("supplies") and not room_memory.get("doors"):
                return {"tool_name": "scan_room", "args": {}}
            if state.get("active_supply_id"):
                return {"tool_name": "move_to_object", "args": {"target_type": "supply"}}
            unvisited_supplies = [
                item for item in room_memory.get("supplies", [])
                if item and str(item.get("id")) not in set(room_memory.get("visited_supply_ids", []))
            ]
            if unvisited_supplies:
                return {"tool_name": "select_next_supply", "args": {}}
            if state.get("active_door_id"):
                if interactions.get("open_door") or interactions.get("close_door"):
                    return {"tool_name": "open_door", "args": {}}
                return {"tool_name": "enter_door", "args": {}}
            return {"tool_name": "select_next_door", "args": {}}

        return {"tool_name": "wait_and_refresh", "args": {"wait_sec": 0.2}}

    def _model_decision(self, snapshot: QwenRoomPerceptionSnapshot) -> Optional[Dict[str, Any]]:
        payload = self._build_payload(snapshot, trace=False)
        response = self._post_chat_completions(payload)
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not content:
            return None
        return self._parse_json(content)

    def _build_payload(self, snapshot: QwenRoomPerceptionSnapshot, *, trace: bool) -> Dict[str, Any]:
        compact_observation = self._compact_prompt_observation(snapshot.observation)
        observation_text = json.dumps(compact_observation, ensure_ascii=False, separators=(",", ":"))
        if len(observation_text) > self.prompt_observation_max_chars:
            compact_observation = self._minimal_prompt_observation(compact_observation)
            observation_text = json.dumps(compact_observation, ensure_ascii=False, separators=(",", ":"))
        if len(observation_text) > self.prompt_observation_max_chars:
            compact_observation = self._emergency_prompt_observation(compact_observation)
            observation_text = json.dumps(compact_observation, ensure_ascii=False, separators=(",", ":"))

        text = (
            "根据画面和紧凑状态选择一个工具，只输出 JSON。"
            f"观察：{observation_text}"
        )
        if snapshot.frame_data_url:
            content = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": "frame" if trace else snapshot.frame_data_url},
                },
            ]
        else:
            content = text

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
        }

    def _post_chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.trace_payload_size:
            print(
                "[QwenRoomControl] "
                f"request_bytes={len(data)}, max_tokens={self.max_tokens}"
            )
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if len(body) > self.http_error_body_chars:
                body = body[: self.http_error_body_chars - 3] + "..."
            detail = f"HTTP {exc.code} {exc.reason}"
            if body:
                detail = f"{detail}: {body}"
            raise RuntimeError(
                f"Qwen request failed: {detail}; request_bytes={len(data)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen request failed: {exc}") from exc

    def _parse_json(self, content: str) -> Dict[str, Any]:
        text = str(content).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    def _compact_prompt_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        observation = observation or {}
        state = observation.get("state") or {}
        agent_state = observation.get("agent_state") or {}
        agent_memory = observation.get("agent_memory") or {}
        visible_objects = observation.get("visible_objects") or []
        return {
            "task": self._truncate(observation.get("task", ""), 80),
            "state": self._compact_state(state),
            "visible_objects": self._compact_visible_objects(visible_objects),
            "agent_state": self._compact_agent_state(agent_state),
            "agent_memory": self._compact_agent_memory(agent_memory),
            "available_tools": self._compact_available_tools(observation.get("available_tools") or []),
        }

    def _minimal_prompt_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        memory = observation.get("agent_memory") or {}
        if memory.get("recent_steps"):
            memory["recent_steps"] = memory["recent_steps"][-1:]
        if memory.get("summary"):
            memory["summary"] = self._truncate(memory.get("summary"), 220)

        state = observation.get("state") or {}
        state["room_memory"] = {
            "supply_count": state.get("room_memory", {}).get("supply_count"),
            "door_count": state.get("room_memory", {}).get("door_count"),
            "room_stack_depth": state.get("room_memory", {}).get("room_stack_depth"),
        }
        return {
            "task": observation.get("task"),
            "state": state,
            "visible_objects": {"counts": observation.get("visible_objects", {}).get("counts", {})},
            "agent_state": observation.get("agent_state"),
            "agent_memory": memory,
            "available_tools": observation.get("available_tools"),
        }

    def _emergency_prompt_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        state = observation.get("state") or {}
        agent_state = observation.get("agent_state") or {}
        return {
            "task": observation.get("task"),
            "state": {
                "location": state.get("location"),
                "scene": state.get("scene"),
                "status": state.get("status"),
                "current_house_id": state.get("current_house_id"),
                "distance_to_entry": state.get("distance_to_entry"),
                "completed_house_count": state.get("completed_house_count"),
                "active_supply_id": state.get("active_supply_id"),
                "active_door_id": state.get("active_door_id"),
                "interactions": state.get("interactions"),
            },
            "agent_state": {
                "max_houses": agent_state.get("max_houses"),
                "entered_current_house": agent_state.get("entered_current_house"),
                "consecutive_errors": agent_state.get("consecutive_errors"),
                "completed_house_count": agent_state.get("completed_house_count"),
                "last_action": agent_state.get("last_action"),
            },
            "agent_memory": {
                "summary": self._truncate(
                    (observation.get("agent_memory") or {}).get("summary", ""),
                    120,
                )
            },
            "visible_objects": {"counts": (observation.get("visible_objects") or {}).get("counts", {})},
            "available_tools": observation.get("available_tools"),
        }

    def _compact_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "location": state.get("location"),
            "direction": state.get("direction"),
            "scene": state.get("house_scene_name"),
            "status": state.get("status"),
            "current_room_id": state.get("current_room_id"),
            "current_house_id": state.get("current_house_id"),
            "active_entry": self._compact_entry(state.get("active_entry")),
            "distance_to_entry": state.get("distance_to_entry"),
            "completed_house_count": state.get("completed_house_count"),
            "active_supply_id": state.get("active_supply_id"),
            "active_door_id": state.get("active_door_id"),
            "interactions": state.get("interactions") or {},
            "room_memory": self._compact_room_memory(state.get("room_memory") or {}),
        }

    def _compact_entry(self, entry: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return None
        return {
            "location": entry.get("location"),
            "direction": entry.get("direction"),
        }

    def _compact_room_memory(self, room_memory: Dict[str, Any]) -> Dict[str, Any]:
        supplies = room_memory.get("supplies") or []
        doors = room_memory.get("doors") or []
        visited_supply_ids = room_memory.get("visited_supply_ids") or []
        visited_door_ids = room_memory.get("visited_door_ids") or []
        return {
            "room_id": room_memory.get("room_id"),
            "loot_count": room_memory.get("loot_count"),
            "supply_count": len(supplies) if isinstance(supplies, list) else 0,
            "door_count": len(doors) if isinstance(doors, list) else 0,
            "visited_supply_count": len(visited_supply_ids) if isinstance(visited_supply_ids, list) else 0,
            "visited_door_count": len(visited_door_ids) if isinstance(visited_door_ids, list) else 0,
            "room_stack_depth": room_memory.get("room_stack_depth"),
            "next_supply": self._compact_target(supplies[0]) if isinstance(supplies, list) and supplies else None,
            "next_door": self._compact_target(doors[0]) if isinstance(doors, list) and doors else None,
        }

    def _compact_visible_objects(self, objects: Any) -> Dict[str, Any]:
        if not isinstance(objects, list):
            return {"counts": {}, "objects": []}
        counts: Dict[str, int] = {}
        compact = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            obj_type = str(obj.get("type") or "other")
            counts[obj_type] = counts.get(obj_type, 0) + 1
            if len(compact) >= self.prompt_visible_object_limit:
                continue
            compact.append({
                "type": obj_type,
                "center_offset_px": obj.get("center_offset_px"),
                "area": obj.get("area"),
                "box_h": obj.get("box_h"),
                "abs_angle": obj.get("abs_angle"),
            })
        return {"counts": counts, "objects": compact}

    def _compact_agent_state(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        last_tool_result = agent_state.get("last_tool_result") or {}
        result_observation = last_tool_result.get("observation") or {}
        last_decision = agent_state.get("last_decision") or {}
        return {
            "max_houses": agent_state.get("max_houses"),
            "entered_current_house": agent_state.get("entered_current_house"),
            "consecutive_errors": agent_state.get("consecutive_errors"),
            "completed_house_count": agent_state.get("completed_house_count"),
            "should_finish": agent_state.get("should_finish"),
            "last_decision": {
                "tool_name": last_decision.get("tool_name"),
                "reason": self._truncate(last_decision.get("reason", ""), 60),
            },
            "last_action": {
                "tool_name": last_tool_result.get("tool_name"),
                "ok": last_tool_result.get("ok"),
                "error": self._truncate(last_tool_result.get("error", ""), 80),
                "result_type": result_observation.get("result_type"),
                "action": result_observation.get("action"),
                "at_entry": result_observation.get("at_entry"),
                "after_distance": result_observation.get("after_distance"),
                "moved_distance": result_observation.get("moved_distance"),
                "distance_delta": result_observation.get("distance_delta"),
                "state_after": self._compact_state_after(result_observation.get("state_after") or {}),
            },
        }

    def _compact_agent_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        recent_steps = memory.get("recent_steps") or []
        if not isinstance(recent_steps, list):
            recent_steps = []
        if self.prompt_recent_steps <= 0:
            compact_steps = []
        else:
            compact_steps = [
                self._compact_memory_step(step)
                for step in recent_steps[-self.prompt_recent_steps :]
            ]
        return {
            "summary": self._truncate(memory.get("summary", ""), self.prompt_summary_max_chars),
            "recent_steps": compact_steps,
        }

    def _compact_memory_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(step, dict):
            return {}
        result = step.get("result") or {}
        before = step.get("before") or {}
        after = step.get("after") or {}
        decision = step.get("decision") or {}
        return {
            "round": step.get("round"),
            "tool_name": decision.get("tool_name"),
            "result_type": result.get("result_type"),
            "ok": result.get("ok"),
            "error": self._truncate(result.get("error", ""), 60),
            "before": self._compact_step_state(before),
            "after": self._compact_step_state(after),
            "moved_distance": result.get("moved_distance"),
            "distance_delta": result.get("distance_delta"),
            "at_entry": result.get("at_entry"),
        }

    def _compact_state_after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "scene": state.get("house_scene_name") or state.get("scene"),
            "status": state.get("status"),
            "location": state.get("location"),
            "current_house_id": state.get("current_house_id"),
            "distance_to_entry": state.get("distance_to_entry"),
            "completed_house_count": state.get("completed_house_count"),
        }

    def _compact_step_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "scene": state.get("scene") or state.get("house_scene_name"),
            "status": state.get("status"),
            "location": state.get("location"),
            "completed_house_count": state.get("completed_house_count"),
        }

    def _compact_available_tools(self, tools: Any):
        if not isinstance(tools, list):
            return list(QwenHouseSearchTools.TOOL_NAMES)
        names = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("name"):
                names.append(tool["name"])
            elif isinstance(tool, str):
                names.append(tool)
        return names or list(QwenHouseSearchTools.TOOL_NAMES)

    def _compact_target(self, target: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(target, dict):
            return None
        return {
            "id": target.get("id"),
            "kind": target.get("kind"),
            "abs_angle": target.get("abs_angle"),
            "box_h": target.get("box_h"),
            "area": target.get("area"),
        }

    def _truncate(self, value: Any, limit: int) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _normalize_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(decision, dict):
            raise ValueError("Qwen decision must be a dict")
        tool_name = str(decision.get("tool_name") or "").strip()
        tool_name = {
            "navigate_to_house_try": "navigate_to_house_entry",
            "navigate_to_entry": "navigate_to_house_entry",
            "go_to_house_entry": "navigate_to_house_entry",
            "precise_nav": "navigate_to_house_entry",
            "fast_nav": "navigate_to_house_entry",
        }.get(tool_name, tool_name)
        if tool_name not in QwenHouseSearchTools.TOOL_NAMES:
            raise ValueError(f"invalid tool_name from Qwen: {tool_name}")
        args = decision.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        return {
            "tool_name": tool_name,
            "args": args,
            "reason": str(decision.get("reason") or ""),
            "confidence": float(decision.get("confidence") or 0.0),
        }

    def _guard_decision(
        self,
        decision: Dict[str, Any],
        tools: QwenHouseSearchTools,
    ) -> Dict[str, Any]:
        state = tools.get_game_state().observation
        if decision["tool_name"] == "select_next_house":
            if state.get("current_house_id") is not None and state.get("active_entry") is not None:
                return {
                    "tool_name": "navigate_to_house_entry",
                    "args": {},
                    "reason": "已存在当前房子和入户点，继续向入户点移动",
                    "confidence": 1.0,
                }

        if decision["tool_name"] != "mark_house_done" or self.state_agent.entered_current_house:
            return decision

        if state.get("active_entry") is None:
            return {
                "tool_name": "select_next_house",
                "args": {},
                "reason": "尚未进入当前房子，不能标记完成，先选择房子",
                "confidence": 1.0,
            }
        return {
            "tool_name": "navigate_to_house_entry",
            "args": {},
            "reason": "尚未进入当前房子，不能标记完成，继续靠近入户点",
            "confidence": 1.0,
        }

    def _is_stuck(self, tools: QwenHouseSearchTools) -> bool:
        result = tools.check_stuck().to_dict()
        return bool(result.get("observation", {}).get("stuck"))
