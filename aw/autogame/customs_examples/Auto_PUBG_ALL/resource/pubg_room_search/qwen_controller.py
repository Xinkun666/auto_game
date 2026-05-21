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
        self.max_tokens = int(config.get("qwen_max_tokens") or 384)
        self.timeout_sec = float(config.get("qwen_timeout_sec") or 20.0)
        self.http_error_body_chars = max(120, int(config.get("qwen_http_error_body_chars") or 1200))
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
        text = (
            "请根据当前画面和结构化观察选择下一步工具。"
            "只输出 JSON。\n\n"
            f"观察：{json.dumps(snapshot.observation, ensure_ascii=False)}"
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
