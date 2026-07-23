"""南大房型匹配/回放方案的稳定接入协议。

当前搜房状态机负责把人物送到入门点附近，使用现有 uinput 方向模块完成
入门方向校准，再提供门检测框和当前画面。本模块定义四个边界：

1. ``NandaEntryPosePreparer`` 把人物收敛到统一的门前位姿；
2. ``NandaRoomMatcher`` 把标准门前画面匹配到房型和回放记录；
3. ``NandaReplayExecutor`` 在当前设备上执行回放；
4. ``NandaHouseSearchStrategy`` 组合三者，并用统一结果协议回传给现有状态机。

未配置南大方案时，策略返回 ``DISABLED``，现有搜房流程保持不变；生产
配置启用南大方案后使用 ``exclusive`` 模式，接管后任何异常都应
终止当前用例并上报，不再回退原室内搜房逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

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


class NandaViewPreparationError(RuntimeError):
    """door frame 取景阶段无法生成可用房屋视野。"""


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
    refresh_context: Optional[
        Callable[[str], Optional["NandaSearchContext"]]
    ] = None


@dataclass(frozen=True)
class NandaRoomMatch:
    """房型匹配器输出，与 SAM3/DINO 的具体实现解耦。"""

    room_id: str
    replay_path: str
    score: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    replay_steps: Optional[Sequence[Mapping[str, Any]]] = None


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

    def is_available(self) -> bool:
        """匹配器当前是否可用；默认的进程内实现始终可用。"""
        return True

    def warmup(self) -> None:
        """可选预加载入口；默认匹配器不需要预热。"""

    @abstractmethod
    def match(self, context: NandaSearchContext) -> Optional[NandaRoomMatch]:
        """匹配当前房型；无可靠匹配时返回 ``None``。"""

    def reset(self) -> None:
        """新一轮测试开始时清理可选运行态。"""


class NandaEntryPosePreparer(ABC):
    """门前距离和门中心位置的收敛接口；入门方向由现有导航模块负责。"""

    @abstractmethod
    def prepare(self, context: NandaSearchContext) -> Optional[NandaSearchResult]:
        """
        已达到标准位姿时返回 ``None``。

        本轮已做横移/前后校正、需等待新帧时返回 ``RETRY``；
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
        exclusive: bool = False,
    ):
        self.matcher = matcher
        self.replay_executor = replay_executor
        self.pose_preparer = pose_preparer
        self.exclusive = bool(exclusive)
        self._ready_checked = False
        self._ready_result: Optional[NandaSearchResult] = None

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

    def validate_ready(self) -> Optional[NandaSearchResult]:
        """在人物移动前检查南大管线的本地资产与依赖。"""
        if self._ready_checked:
            return self._ready_result
        self._ready_result = self._validate_ready_once()
        self._ready_checked = True
        return self._ready_result

    def _validate_ready_once(self) -> Optional[NandaSearchResult]:
        if not self.enabled:
            return NandaSearchResult(
                NandaSearchStatus.DISABLED,
                "南大方案需同时配置门前位姿准备器、房型匹配器和回放执行器",
                metadata={"phase": "preflight"},
            )
        try:
            matcher_available = self.matcher.is_available()
        except Exception as exc:
            return NandaSearchResult(
                NandaSearchStatus.FAILED,
                f"本地房型匹配检查异常: {exc}",
                metadata={"phase": "preflight", "exception": type(exc).__name__},
            )
        if not matcher_available:
            unavailable_reason = getattr(
                self.matcher,
                "unavailable_reason",
                "本地房型匹配当前不可用",
            )
            return NandaSearchResult(
                NandaSearchStatus.FAILED,
                str(unavailable_reason or "本地房型匹配当前不可用"),
                metadata={"phase": "preflight", "matcher_unavailable": True},
            )
        try:
            self.matcher.warmup()
        except Exception as exc:
            return NandaSearchResult(
                NandaSearchStatus.FAILED,
                f"本地房型匹配预加载异常: {exc}",
                metadata={"phase": "preflight", "exception": type(exc).__name__},
            )
        return None

    def _realign_pose_for_replay(
        self,
        context: NandaSearchContext,
        match: NandaRoomMatch,
    ) -> tuple[NandaSearchContext, Optional[NandaSearchResult]]:
        if not bool(match.metadata.get("requires_pose_realign")):
            return context, None
        refresh_context = context.refresh_context
        if not callable(refresh_context):
            return context, NandaSearchResult(
                NandaSearchStatus.FAILED,
                "door frame 取景改变了人物位置，但当前上下文无法重新执行门前位姿校准",
                room_id=match.room_id,
                replay_path=match.replay_path,
                metadata={"phase": "pose_restore"},
            )

        settings = getattr(self.pose_preparer, "settings", None)
        max_actions = int(getattr(settings, "max_pose_actions", 18) or 18)
        stable_count = int(getattr(settings, "stable_required_count", 2) or 2)
        max_cycles = max(4, max_actions + stable_count + 4)
        current_context = context
        reset_pose = getattr(self.pose_preparer, "reset", None)
        if callable(reset_pose):
            reset_pose()
        context.worker.frame_log(
            f"[NandaPoseRestore] 房型已确定为 {match.room_id}，"
            "door frame 取景后拉改变了回放起点；"
            "只重新按YOLO门中心和门框面积校准，方向不由南大模块控制"
        )
        for cycle in range(1, max_cycles + 1):
            if current_context.should_abort():
                return current_context, NandaSearchResult(
                    NandaSearchStatus.ABORTED,
                    "恢复标准门前回放位姿时搜房阶段已中止",
                    room_id=match.room_id,
                    replay_path=match.replay_path,
                    metadata={"phase": "pose_restore", "cycle": cycle},
                )
            refresh_context = current_context.refresh_context
            if not callable(refresh_context):
                return current_context, NandaSearchResult(
                    NandaSearchStatus.FAILED,
                    "恢复门前位姿过程中丢失刷新上下文接口",
                    room_id=match.room_id,
                    replay_path=match.replay_path,
                    metadata={"phase": "pose_restore", "cycle": cycle},
                )
            fresh_context = refresh_context(
                f"NandaPoseRestore 第 {cycle}/{max_cycles} 轮刷新门框"
            )
            if fresh_context is None:
                return current_context, NandaSearchResult(
                    NandaSearchStatus.FAILED,
                    "恢复标准门前回放位姿时无法刷新门框上下文",
                    room_id=match.room_id,
                    replay_path=match.replay_path,
                    metadata={"phase": "pose_restore", "cycle": cycle},
                )
            current_context = fresh_context
            try:
                pose_result = self.pose_preparer.prepare(current_context)
            except Exception as exc:
                return current_context, NandaSearchResult(
                    NandaSearchStatus.FAILED,
                    f"恢复标准门前回放位姿异常: {exc}",
                    room_id=match.room_id,
                    replay_path=match.replay_path,
                    metadata={
                        "phase": "pose_restore",
                        "cycle": cycle,
                        "exception": type(exc).__name__,
                    },
                )
            if pose_result is None:
                context.worker.frame_log(
                    f"[NandaPoseRestore] 标准门前回放位姿恢复完成："
                    f"cycle={cycle}/{max_cycles}，room={match.room_id}"
                )
                return current_context, None
            if not isinstance(pose_result, NandaSearchResult):
                return current_context, NandaSearchResult(
                    NandaSearchStatus.FAILED,
                    f"恢复位姿时准备器返回无效类型: {type(pose_result).__name__}",
                    room_id=match.room_id,
                    replay_path=match.replay_path,
                    metadata={"phase": "pose_restore", "cycle": cycle},
                )
            if pose_result.status != NandaSearchStatus.RETRY:
                return current_context, NandaSearchResult(
                    status=pose_result.status,
                    message=pose_result.message,
                    room_id=match.room_id,
                    replay_path=match.replay_path,
                    metadata={
                        **dict(pose_result.metadata),
                        "phase": "pose_restore",
                        "cycle": cycle,
                    },
                )

        return current_context, NandaSearchResult(
            NandaSearchStatus.FAILED,
            f"恢复标准门前回放位姿超过最大循环次数 {max_cycles}",
            room_id=match.room_id,
            replay_path=match.replay_path,
            metadata={"phase": "pose_restore", "max_cycles": max_cycles},
        )

    def run(self, context: NandaSearchContext) -> NandaSearchResult:
        if context.should_abort():
            return NandaSearchResult(NandaSearchStatus.ABORTED, "搜房阶段已中止")

        # 本地匹配资产或依赖不可用时必须在位姿校准前退出。
        ready_error = self.validate_ready()
        if ready_error is not None:
            return ready_error

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
        except NandaViewPreparationError as exc:
            return NandaSearchResult(
                NandaSearchStatus.FAILED,
                f"door frame 房屋取景准备异常: {exc}",
                metadata={
                    "phase": "view_preparation",
                    "exception": type(exc).__name__,
                },
            )
        except Exception as exc:
            return NandaSearchResult(
                NandaSearchStatus.FAILED,
                f"房型匹配异常: {exc}",
                metadata={"phase": "match", "exception": type(exc).__name__},
            )

        if match is None:
            return NandaSearchResult(
                NandaSearchStatus.NO_MATCH,
                "未匹配到可用回放房型",
                metadata={"phase": "match"},
            )
        if context.should_abort():
            return NandaSearchResult(
                NandaSearchStatus.ABORTED,
                "房型匹配后搜房阶段已中止",
                room_id=match.room_id,
                replay_path=match.replay_path,
            )

        replay_context, pose_restore_result = self._realign_pose_for_replay(
            context,
            match,
        )
        if pose_restore_result is not None:
            return pose_restore_result

        try:
            result = self.replay_executor.replay(replay_context, match)
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
    "NandaViewPreparationError",
]
