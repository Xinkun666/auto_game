from __future__ import annotations

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

    def process(self, worker: Any) -> bool:
        if not self.enabled:
            return False

        tools = QwenHouseSearchTools(self.searcher, worker)
        if self.state_agent.should_finish():
            result = tools.dispatch("finish_house_search", {})
            self.state_agent.record_tool_result("finish_house_search", result)
            return True

        try:
            snapshot = self.perception_agent.observe(
                worker,
                tools,
                task=f"搜索房屋，搜满 {self.max_houses} 个房子后结束",
                state_snapshot=self.state_agent.snapshot(),
            )
            self.state_agent.sync_from_observation(snapshot.observation)
            decision = self.control_agent.decide(snapshot, tools)
            self.state_agent.record_decision(decision)

            result = tools.dispatch(decision["tool_name"], decision.get("args") or {})
            self.state_agent.record_tool_result(decision["tool_name"], result)
            state = tools.get_game_state().observation
            print(
                f"[QwenRoomAgent] tool={decision['tool_name']}, "
                f"ok={result.get('ok')}, "
                f"scene={state.get('house_scene_name')}, "
                f"house={state.get('current_house_id')}, "
                f"entry={bool(state.get('active_entry'))}, "
                f"status={state.get('status')}, "
                f"reason={decision.get('reason', '')}, "
                f"error={result.get('error', '')}"
            )
            return True
        except Exception as exc:
            self.state_agent.record_error()
            print(
                "[QwenRoomAgent] Qwen 搜房异常 "
                f"{self.state_agent.state.consecutive_errors}/{self.state_agent.state.max_errors}: {exc}"
            )
            if self.fallback_to_legacy and self.state_agent.too_many_errors():
                print("[QwenRoomAgent] 连续异常达到上限，回退旧搜房逻辑")
                self.reset()
                return False

            decision = self.control_agent.fallback_decision(tools)
            result = tools.dispatch(decision["tool_name"], decision.get("args") or {})
            self.state_agent.record_decision(decision)
            self.state_agent.record_tool_result(decision["tool_name"], result)
            return True
