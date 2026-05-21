from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.config import (
    get_pubg_room_search_config,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.qwen_controller import (
    QwenRoomControlAgent,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.qwen_perception import (
    QwenRoomPerceptionAgent,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.qwen_state import (
    QwenRoomStateAgent,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.pubg_room_search.qwen_tools import (
    QwenHouseSearchTools,
)


class QwenRoomSearchAgent:
    """Coordinator for the Qwen room-search agent system."""

    def __init__(self, searcher: Any, config: Optional[Dict[str, Any]] = None):
        self.searcher = searcher
        self.config = config or get_pubg_room_search_config()
        self.enabled = bool(self.config.get("qwen_agent_enabled", False))
        self.max_houses = int(self.config.get("qwen_max_houses") or 5)
        self.fallback_to_legacy = bool(self.config.get("qwen_fallback_to_legacy", True))
        self.trace_enabled = bool(self.config.get("qwen_trace_enabled", True))
        self.trace_prompt = bool(self.config.get("qwen_trace_prompt", True))
        self.round_index = 0

        max_errors = int(self.config.get("qwen_max_consecutive_errors") or 3)
        self.state_agent = QwenRoomStateAgent(
            searcher,
            max_houses=self.max_houses,
            max_errors=max_errors,
        )
        self.perception_agent = QwenRoomPerceptionAgent(self.config)
        self.control_agent = QwenRoomControlAgent(searcher, self.config, self.state_agent)

    @classmethod
    def from_config(cls, searcher: Any) -> Optional["QwenRoomSearchAgent"]:
        config = get_pubg_room_search_config()
        env_enabled = os.environ.get("AUTOGAME_QWEN_ROOM_AGENT")
        if env_enabled is not None:
            config["qwen_agent_enabled"] = env_enabled.strip().lower() in {"1", "true", "yes", "on"}
        if not config.get("qwen_agent_enabled", False):
            return None
        return cls(searcher, config)

    def reset(self):
        self.state_agent.reset()
        self.round_index = 0

    def process(self, worker: Any) -> bool:
        if not self.enabled:
            return False

        tools = QwenHouseSearchTools(self.searcher, worker)
        self.round_index += 1
        self._trace("Coordinator", "开始新一轮 Qwen 搜房闭环")
        if self.state_agent.should_finish():
            self._trace("Coordinator -> Tools", {"tool_name": "finish_house_search", "args": {}})
            result = tools.dispatch("finish_house_search", {})
            self.state_agent.record_tool_result("finish_house_search", result)
            self._trace("Tools -> State", result)
            return True

        try:
            state_before = self.state_agent.snapshot()
            self._trace(
                "Coordinator -> Perception",
                {
                    "task": f"搜索房屋，搜满 {self.max_houses} 个房子后结束",
                    "agent_state": state_before,
                },
            )
            snapshot = self.perception_agent.observe(
                worker,
                tools,
                task=f"搜索房屋，搜满 {self.max_houses} 个房子后结束",
                state_snapshot=state_before,
            )
            self._trace(
                "Perception -> Coordinator",
                {
                    "has_frame": bool(snapshot.frame_data_url),
                    "frame": "frame" if snapshot.frame_data_url else None,
                    "observation": snapshot.observation,
                },
            )
            self.state_agent.sync_from_observation(snapshot.observation)
            self._trace("Coordinator -> State", {"action": "sync_from_observation"})
            self._trace("State -> Coordinator", self.state_agent.snapshot())
            if self.trace_prompt:
                self._trace("Coordinator -> Control prompt", self.control_agent.prompt_for_trace(snapshot))
            decision = self.control_agent.decide(snapshot, tools)
            self._trace("Control -> Coordinator decision", decision)
            self.state_agent.record_decision(decision)

            self._trace(
                "Coordinator -> Tools",
                {
                    "tool_name": decision["tool_name"],
                    "args": decision.get("args") or {},
                    "reason": decision.get("reason", ""),
                },
            )
            result = tools.dispatch(decision["tool_name"], decision.get("args") or {})
            self._trace("Tools -> Coordinator result", result)
            self.state_agent.record_tool_result(decision["tool_name"], result)
            self._trace("Coordinator -> State", {"action": "record_tool_result"})
            self._trace("State -> Coordinator", self.state_agent.snapshot())
            observation = result.get("observation") or {}
            state = observation.get("state_after") or tools.get_game_state().observation
            print(
                f"[QwenRoomAgent] tool={decision['tool_name']}, "
                f"ok={result.get('ok')}, "
                f"scene={state.get('house_scene_name')}, "
                f"house={state.get('current_house_id')}, "
                f"entry={bool(state.get('has_active_entry', state.get('active_entry')))}, "
                f"dist={state.get('distance_to_entry')}, "
                f"status={state.get('status')}, "
                f"action={observation.get('action', '')}, "
                f"moved={observation.get('moved_distance', '')}, "
                f"delta={observation.get('distance_delta', '')}, "
                f"reason={decision.get('reason', '')}, "
                f"error={result.get('error', '')}"
            )
            return True
        except Exception as exc:
            self.state_agent.record_error()
            self._trace(
                "Coordinator exception",
                {
                    "error": str(exc),
                    "consecutive_errors": self.state_agent.state.consecutive_errors,
                    "max_errors": self.state_agent.state.max_errors,
                },
            )
            print(
                "[QwenRoomAgent] Qwen 搜房异常 "
                f"{self.state_agent.state.consecutive_errors}/{self.state_agent.state.max_errors}: {exc}"
            )
            if self.fallback_to_legacy and self.state_agent.too_many_errors():
                print("[QwenRoomAgent] 连续异常达到上限，回退旧搜房逻辑")
                self.reset()
                return False

            decision = self.control_agent.fallback_decision(tools)
            self._trace("Control fallback -> Coordinator decision", decision)
            self._trace(
                "Coordinator -> Tools",
                {"tool_name": decision["tool_name"], "args": decision.get("args") or {}},
            )
            result = tools.dispatch(decision["tool_name"], decision.get("args") or {})
            self._trace("Tools -> Coordinator result", result)
            self.state_agent.record_decision(decision)
            self.state_agent.record_tool_result(decision["tool_name"], result)
            self._trace("State -> Coordinator", self.state_agent.snapshot())
            return True

    def _trace(self, channel: str, payload: Any = None):
        if not self.trace_enabled:
            return
        prefix = f"[QwenRoomTrace][round={self.round_index}][{channel}]"
        if payload is None:
            print(prefix)
            return
        if isinstance(payload, str):
            print(f"{prefix} {payload}")
            return
        print(f"{prefix} {json.dumps(self._trace_safe(payload), ensure_ascii=False, default=str)}")

    def _trace_safe(self, value: Any):
        if isinstance(value, dict):
            safe = {}
            for key, item in value.items():
                if key in {"frame_data_url", "url"} and isinstance(item, str) and item.startswith("data:image"):
                    safe[key] = "frame"
                else:
                    safe[key] = self._trace_safe(item)
            return safe
        if isinstance(value, list):
            return [self._trace_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._trace_safe(item) for item in value]
        if isinstance(value, str) and value.startswith("data:image"):
            return "frame"
        return value
