"""PUBG room-search helpers.

Imports are intentionally lazy so perception-only callers do not pull in
optional Qwen, sklearn, or replay dependencies at package import time.
"""

__all__ = [
    "HouseSearchAdapter",
    "HouseSearchRunResult",
    "EmbeddedHouseSearchAdapter",
    "EmbeddedHouseSearchRunResult",
    "QwenHouseSearchTools",
    "QwenToolResult",
    "QwenRoomSearchAgent",
    "QwenRoomControlAgent",
    "QwenRoomMemoryAgent",
    "QwenRoomPerceptionAgent",
    "QwenRoomPerceptionSnapshot",
    "QwenRoomAgentState",
    "QwenRoomStateAgent",
    "QwenAgentRolePrompt",
    "QWEN_AGENT_ROLE_PROMPTS",
    "COORDINATOR_SYSTEM_PROMPT",
    "STATE_SYSTEM_PROMPT",
    "PERCEPTION_SYSTEM_PROMPT",
    "CONTROL_SYSTEM_PROMPT",
    "get_qwen_agent_role_prompts",
]


def __getattr__(name: str):
    if name in {"HouseSearchAdapter", "HouseSearchRunResult"}:
        from .adapter import HouseSearchAdapter, HouseSearchRunResult

        return {
            "HouseSearchAdapter": HouseSearchAdapter,
            "HouseSearchRunResult": HouseSearchRunResult,
        }[name]
    if name in {"EmbeddedHouseSearchAdapter", "EmbeddedHouseSearchRunResult"}:
        from .embedded_adapter import (
            EmbeddedHouseSearchAdapter,
            EmbeddedHouseSearchRunResult,
        )

        return {
            "EmbeddedHouseSearchAdapter": EmbeddedHouseSearchAdapter,
            "EmbeddedHouseSearchRunResult": EmbeddedHouseSearchRunResult,
        }[name]
    if name in {"QwenHouseSearchTools", "QwenToolResult"}:
        from .qwen_tools import QwenHouseSearchTools, QwenToolResult

        return {
            "QwenHouseSearchTools": QwenHouseSearchTools,
            "QwenToolResult": QwenToolResult,
        }[name]
    if name == "QwenRoomSearchAgent":
        from .qwen_agent import QwenRoomSearchAgent

        return QwenRoomSearchAgent
    if name == "QwenRoomControlAgent":
        from .qwen_controller import QwenRoomControlAgent

        return QwenRoomControlAgent
    if name == "QwenRoomMemoryAgent":
        from .qwen_memory import QwenRoomMemoryAgent

        return QwenRoomMemoryAgent
    if name in {"QwenRoomPerceptionAgent", "QwenRoomPerceptionSnapshot"}:
        from .qwen_perception import QwenRoomPerceptionAgent, QwenRoomPerceptionSnapshot

        return {
            "QwenRoomPerceptionAgent": QwenRoomPerceptionAgent,
            "QwenRoomPerceptionSnapshot": QwenRoomPerceptionSnapshot,
        }[name]
    if name in {
        "CONTROL_SYSTEM_PROMPT",
        "COORDINATOR_SYSTEM_PROMPT",
        "PERCEPTION_SYSTEM_PROMPT",
        "QWEN_AGENT_ROLE_PROMPTS",
        "QwenAgentRolePrompt",
        "STATE_SYSTEM_PROMPT",
        "get_qwen_agent_role_prompts",
    }:
        from . import qwen_prompts

        return getattr(qwen_prompts, name)
    if name in {"QwenRoomAgentState", "QwenRoomStateAgent"}:
        from .qwen_state import QwenRoomAgentState, QwenRoomStateAgent

        return {
            "QwenRoomAgentState": QwenRoomAgentState,
            "QwenRoomStateAgent": QwenRoomStateAgent,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
