"""南大房型匹配/回放方案的稳定接入协议。

当前搜房状态机负责把人物送到入门点附近，并提供入门方向、门检测框和
当前画面。本模块定义四个边界：

1. ``NandaEntryPosePreparer`` 把人物收敛到统一的门前位姿；
2. ``NandaRoomMatcher`` 把标准门前画面匹配到房型和回放记录；
3. ``NandaReplayExecutor`` 在当前设备上执行回放；
4. ``NandaHouseSearchStrategy`` 组合三者，并用统一结果协议回传给现有状态机。

未配置南大方案时，策略返回 ``DISABLED``，现有搜房流程保持不变。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


DoorBox = Tuple[float, float, float, float]
Location = Tuple[int, int]


class NandaSearchStatus(str, Enum):
    """南大方案对现有搜房状态机的处理结果。"""

    DISABLED = "disabled"
    NO_MATCH = "no_match"
    RETRY = "retry"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True)
class NandaSearchContext:
    """开始房型匹配前，现有搜房状态机提供的门前上下文。

    ``frame`` 是当前帧的只读快照引用；匹配器如需跨调用保留，应自行复制。
    执行器可以通过 ``worker`` 使用现有 HOS/HDC 控制抽象，不应依赖南大
    demo 的 scrcpy ``control_core``。
    """

    worker: "FrameWorker"
    frame: Optional[np.ndarray]
    house_id: Optional[str]
    entry: Mapping[str, Any]
    entry_location: Optional[Location]
    entry_direction: Optional[float]
    current_location: Optional[Location]
    current_direction: Optional[float]
    distance_to_entry: Optional[float]
    door_box: Optional[DoorBox]
    door_center_offset_px: Optional[float]
    door_area_ratio: Optional[float]
    phase_label: str
    refresh_frame: Callable[[str], bool]
    should_abort: Callable[[], bool]
    is_outside: Callable[[], bool]


@dataclass(frozen=True)
class NandaRoomMatch:
    """房型匹配器输出，与 SAM3/DINO 的具体实现解耦。"""

    room_id: str
    replay_path: str
    score: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NandaSearchResult:
    """南大方案的统一返回值。"""

    status: NandaSearchStatus
    message: str = ""
    room_id: Optional[str] = None
    replay_path: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def completed(
        cls,
        match: NandaRoomMatch,
        message: str = "南大回放搜房完成",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "NandaSearchResult":
        return cls(
            status=NandaSearchStatus.COMPLETED,
            message=message,
            room_id=match.room_id,
            replay_path=match.replay_path,
            metadata=dict(metadata or {}),
        )


class NandaRoomMatcher(ABC):
    """SAM3、DINO 或其他房型检索实现的接入点。"""

    @abstractmethod
    def match(self, context: NandaSearchContext) -> Optional[NandaRoomMatch]:
        """匹配当前房型；无可靠匹配时返回 ``None``。"""

    def reset(self) -> None:
        """新一轮测试开始时清理可选运行态。"""


class NandaEntryPosePreparer(ABC):
    """门前距离、入门方向和门中心位置的收敛接口。"""

    @abstractmethod
    def prepare(self, context: NandaSearchContext) -> Optional[NandaSearchResult]:
        """
        已达到标准位姿时返回 ``None``。

        本轮已做横移/前后/方向校正、需等待新帧时返回 ``RETRY``；
        位姿无法收敛时返回 ``FAILED``。
        """

    def reset(self) -> None:
        """新一轮测试开始时清理可选运行态。"""


class NandaReplayExecutor(ABC):
    """HOS 单指回放器等设备端执行实现的接入点。"""

    @abstractmethod
    def replay(
        self,
        context: NandaSearchContext,
        match: NandaRoomMatch,
    ) -> NandaSearchResult:
        """执行回放并返回统一结果。"""

    def reset(self) -> None:
        """新一轮测试开始时释放持续按压等可选运行态。"""


class NandaHouseSearchStrategy:
    """组合门前位姿、房型匹配和设备回放的默认管线。"""

    def __init__(
        self,
        matcher: Optional[NandaRoomMatcher] = None,
        replay_executor: Optional[NandaReplayExecutor] = None,
        pose_preparer: Optional[NandaEntryPosePreparer] = None,
    ):
        self.matcher = matcher
        self.replay_executor = replay_executor
        self.pose_preparer = pose_preparer

    @property
    def enabled(self) -> bool:
        return (
            self.pose_preparer is not None
            and self.matcher is not None
            and self.replay_executor is not None
        )

    def reset(self) -> None:
        for component in (self.pose_preparer, self.matcher, self.replay_executor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def run(self, context: NandaSearchContext) -> NandaSearchResult:
        if not self.enabled:
            return NandaSearchResult(
                NandaSearchStatus.DISABLED,
                "南大方案需同时配置门前位姿准备器、房型匹配器和回放执行器",
            )
        if context.should_abort():
            return NandaSearchResult(NandaSearchStatus.ABORTED, "搜房阶段已中止")

        try:
            pose_result = self.pose_preparer.prepare(context)
        except Exception as exc:
            return NandaSearchResult(
                NandaSearchStatus.FAILED,
                f"门前位姿标准化异常: {exc}",
                metadata={"phase": "pose", "exception": type(exc).__name__},
            )
        if pose_result is not None:
            if not isinstance(pose_result, NandaSearchResult):
                return NandaSearchResult(
                    NandaSearchStatus.FAILED,
                    f"位姿准备器返回了无效类型: {type(pose_result).__name__}",
                    metadata={"phase": "pose"},
                )
            return pose_result

        try:
            match = self.matcher.match(context)
        except Exception as exc:
            return NandaSearchResult(
                NandaSearchStatus.FAILED,
                f"房型匹配异常: {exc}",
                metadata={"phase": "match", "exception": type(exc).__name__},
            )

        if match is None:
            return NandaSearchResult(NandaSearchStatus.NO_MATCH, "未匹配到可用回放房型")
        if context.should_abort():
            return NandaSearchResult(
                NandaSearchStatus.ABORTED,
                "房型匹配后搜房阶段已中止",
                room_id=match.room_id,
                replay_path=match.replay_path,
            )

        try:
            result = self.replay_executor.replay(context, match)
        except Exception as exc:
            return NandaSearchResult(
                NandaSearchStatus.FAILED,
                f"回放执行异常: {exc}",
                room_id=match.room_id,
                replay_path=match.replay_path,
                metadata={"phase": "replay", "exception": type(exc).__name__},
            )

        if not isinstance(result, NandaSearchResult):
            return NandaSearchResult(
                NandaSearchStatus.FAILED,
                f"回放执行器返回了无效类型: {type(result).__name__}",
                room_id=match.room_id,
                replay_path=match.replay_path,
            )
        if result.room_id is None or result.replay_path is None:
            return NandaSearchResult(
                status=result.status,
                message=result.message,
                room_id=result.room_id or match.room_id,
                replay_path=result.replay_path or match.replay_path,
                metadata=dict(result.metadata),
            )
        return result


__all__ = [
    "NandaEntryPosePreparer",
    "NandaHouseSearchStrategy",
    "NandaReplayExecutor",
    "NandaRoomMatch",
    "NandaRoomMatcher",
    "NandaSearchContext",
    "NandaSearchResult",
    "NandaSearchStatus",
]
