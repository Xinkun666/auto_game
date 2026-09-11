from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


def on_stage(w: "FrameWorker"):
    if w.current_stage == "示例阶段":
        return handle_example_stage(w)

    if w.current_stage == "结束阶段":
        w.frame_log("进入结束阶段，停止自动化")
        w.stop()
        return

    w.frame_log(f"当前阶段 {w.current_stage} 暂无处理逻辑，等待下一帧")


def handle_example_stage(w: "FrameWorker"):
    if w.get_info("示例按钮"):
        w.frame_log("看到示例按钮，点击示例按钮")
        w.click("示例按钮")
        w.refresh_frame()
        return

    w.frame_log("示例阶段暂时没有看到示例按钮，继续等待")
