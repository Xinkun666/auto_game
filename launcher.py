import argparse
import ast
import base64
import importlib
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, NamedTuple, Optional
from xml.etree import ElementTree

from PyQt6.QtCore import QByteArray, QObject, QProcess, QProcessEnvironment, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut, QTextCursor, QTextOption
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aw.autogame.tools.ProcessUtils import hidden_subprocess_context, hidden_subprocess_kwargs, install_hidden_subprocess_patch, resolve_hdc_executable, start_hidden_subprocess_window_suppressor
from aw.autogame.tools.GameLaunchProfile import (
    DEFAULT_PUBG_GAME_PACKAGE,
    DEFAULT_SP_PACKAGE,
    TEST_PROFILE_MARATHON,
    should_use_sp_recording_for_profile,
)
from aw.autogame.tools.FrameLog import FrameLogType, parse_frame_log_transport
from aw.autogame.tools.HdcDebugLog import (
    HdcDebugRunCapture,
    restart_hdc_debug_server,
    resolve_hdc_debug_level,
)
from aw.autogame.tools.HilogCapture import HilogRunCapture
from aw.autogame.tools.MemoryCapture import MemoryRunCapture
from aw.autogame.tools.Utils import LATEST_PREVIEW_POINTER_FILENAME, archive_run_artifacts, get_display_rotation, get_resolution, get_screen_mode, prune_run_archive_artifacts, resolve_run_archive_dir, select_scene_resolution
from aw.autogame.tools.AreaResolver import resolve_area_rect_for_frame
from aw.autogame.common.SPController.SPArea import (
    SP_CONTROLLER_STATE_FILE,
    SP_SAVE_PROTECTION_LOG_MARKER,
    build_sp_save_shell_command,
    calculate_sp_save_settle_seconds,
)

class AppPaths(NamedTuple):
    app_dir: Path
    internal_dir: Path
    root_dir: Path


class ValidationIssues:
    """Collect launch validation results so one user action produces one dialog."""

    def __init__(self):
        self.errors: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []

    def add_error(self, title: str, message: str):
        self.errors.append((str(title).strip(), str(message).strip()))

    def add_warning(self, title: str, message: str):
        self.warnings.append((str(title).strip(), str(message).strip()))

    def has_errors(self) -> bool:
        return bool(self.errors)


def resolve_app_paths(
    frozen: Optional[bool] = None,
    executable: Optional[Path] = None,
    file_path: Optional[Path] = None,
) -> AppPaths:
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))

    if frozen:
        app_dir = Path(executable or sys.executable).resolve().parent
        internal_dir = app_dir / "_internal"
        root_dir = internal_dir if internal_dir.exists() else app_dir
        return AppPaths(app_dir=app_dir, internal_dir=internal_dir, root_dir=root_dir)

    app_dir = Path(file_path or __file__).resolve().parent
    return AppPaths(app_dir=app_dir, internal_dir=app_dir, root_dir=app_dir)


def resolve_runtime_temp_dir(app_dir: Optional[Path] = None) -> Path:
    return Path(app_dir or APP_DIR).resolve() / "aw" / "autogame" / "temp"


def resolve_history_temp_dir() -> Path:
    return Path("aw") / "autogame" / "temp"


APP_PATHS = resolve_app_paths()
APP_DIR = APP_PATHS.app_dir
INTERNAL_DIR = APP_PATHS.internal_dir
ROOT_DIR = APP_PATHS.root_dir
AUTOGAME_CONFIG_FILE = ROOT_DIR / "aw" / "autogame" / "config" / "config.json"
HOSCRCPY_FRAME_RATE_OPTIONS = (15, 30, 60, 120)
DEFAULT_HOSCRCPY_FRAME_RATE = HOSCRCPY_FRAME_RATE_OPTIONS[0]
TESTCASES_DIR = APP_DIR / "testcases"
CUSTOMS_EXAMPLES_DIR = ROOT_DIR / "aw" / "autogame" / "customs_examples"
CUSTOMS_GAME_EXAMPLES_DIR = ROOT_DIR / "aw" / "autogame" / "customs_game_examples"
GAME_RECORDING_PROJECT_DIR = CUSTOMS_EXAMPLES_DIR / "Game_Recording"
TEMP_DIR = resolve_runtime_temp_dir(APP_DIR)
PACKAGE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+){2,}")
LOGGER = logging.getLogger("launcher")
LAUNCHER_FILE_HANDLER_MARKER = "_autogame_run_file_handler"
PREVIEW_FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}
PYINSTALLER_SUPPRESS_SPLASH_ENV = "PYINSTALLER_SUPPRESS_SPLASH_SCREEN"
PYINSTALLER_SPLASH_IPC_ENV = "_PYI_SPLASH_IPC"
PROCESS_TRACE_ENV = "AUTOGAME_PROCESS_TRACE"
OUTPUT_CONSOLE_MAX_BLOCKS = 3000
OUTPUT_MEMORY_MAX_ENTRIES = 5000
STREAM_CONNECTED_MARKERS = (
    "[Stream] Start receiving...",
    "[HDC] First frame received.",
)
REBOOT_RELAUNCH_DELAY_SECONDS = 80
STREAM_DISCONNECT_GRACEFUL_STOP_TIMEOUT_MS = 60000
STREAM_DISCONNECT_FORCE_KILL_TIMEOUT_MS = 5000
RUN_STOP_FORCE_KILL_TIMEOUT_MS = 15000
STREAM_DISCONNECT_PATTERNS = (
    "[Stream] Channel ready timeout.",
    "[Stream] Receive loop ended unexpectedly.",
    "[Stream] gRPC Error:",
    "[Stream] Runtime Error:",
    "[HOS] Runtime Error:",
)
SP_RECORD_EVER_STARTED_MARKERS = (
    "sp 记录已开始",
    "sp 记录已暂停",
    "sp 记录已恢复",
    "sp 数据已保存",
)
LAUNCHER_FAILURE_SIGNAL_FILE = "launcher_failure_signal.json"
DISMISS_REBOOT_PROMPT_ENV = "AUTOGAME_DISMISS_REBOOT_PROMPT"
DEVICE_LOG_SETTLE_TIMEOUT_SECONDS = 3.0
DEVICE_LOG_SETTLE_INTERVAL_SECONDS = 0.2
HDC_SHELL_TIMEOUT_SECONDS = float(os.environ.get("AUTOGAME_HDC_SHELL_TIMEOUT_SECONDS", "5"))
SP_ARTIFACT_PULL_TIMEOUT_SECONDS = float(
    os.environ.get("AUTOGAME_SP_ARTIFACT_PULL_TIMEOUT_SECONDS", "120")
)
SP_ARTIFACT_REMOTE_PATHS = (
    ("db", "/data/app/el2/100/database/com.huawei.hmsapp.hismartperf/entry/rdb/"),
    ("data", "/data/app/el2/100/base/com.huawei.hmsapp.hismartperf/files"),
    ("daemon", "/data/local/tmp/smartperf"),
    ("daemon", "/data/local/tmp/smartperfDevice"),
)
STREAM_DISCONNECT_POLICY_PRESERVE = "preserve"
STREAM_DISCONNECT_POLICY_DISABLED = "disabled"
STREAM_DISCONNECT_POLICY_STOP_ONLY = "stop_only"
STREAM_DISCONNECT_POLICY_STREAM_ONLY = "stream_only"
PUBG_CASE_TARGET_CASE = "auto_pubg"
PUBG_CASE_DEFAULT_LOOP_COUNT = 1
PUBG_CASE_RUNTIME_DESCRIPTION = "和平精英用例默认10分钟搜房、10分钟开车、10分钟跑图，单次循环总测试时长约30分钟。"
LOG_FILTER_ALL = FrameLogType.ALL.value
LOG_CATEGORY_SYSTEM = FrameLogType.SYSTEM.value
LOG_CATEGORY_TIME = FrameLogType.TIME.value
LOG_CATEGORY_LOGIC = FrameLogType.LOGIC.value
LOG_CATEGORY_UI = FrameLogType.UI_CONTROL.value
LOG_CATEGORY_OTHER = FrameLogType.OTHER.value
LOG_FILTERS = (
    LOG_FILTER_ALL,
    LOG_CATEGORY_SYSTEM,
    LOG_CATEGORY_TIME,
    LOG_CATEGORY_LOGIC,
    LOG_CATEGORY_UI,
    LOG_CATEGORY_OTHER,
)
LOG_CATEGORIES = set(LOG_FILTERS) - {LOG_FILTER_ALL}

# 这些字段是每帧都会刷新、只供算法使用的原始识别结果。它们仍完整展示在
# “识别信息”里，但不应冒充为“日志信息”中的业务结论。
HISTORY_ROUTINE_INFO_KEYS = frozenset({
    "direction",
    "location",
    "speed",
    "white_angle",
    "forward_scene",
    "house_scene",
})
HISTORY_MAX_VISIBLE_LOG_LINES = 3
HISTORY_KEY_LOG_MARKERS = (
    "异常",
    "失败",
    "错误",
    "超时",
    "结束",
    "停止",
    "切换",
    "进入",
    "退出",
    "重试",
    "恢复",
    "卡死",
    "成功",
)


def build_launcher_process_args(*helper_args: str) -> list[str]:
    args = [str(arg) for arg in helper_args]
    if getattr(sys, "frozen", False):
        return args
    return [str(ROOT_DIR / "launcher.py"), *args]


STRUCTURED_LOG_RE = re.compile(r"^\[AutoLog\](?:\[(?P<category>[^\]]+)\])?")
TIME_LOG_MARKERS = (
    "[Timer]",
    "[PhaseTimer]",
    "运行信息：",
    "阶段",
    "搜房计时",
    "剩余",
    "remaining=",
)
UI_LOG_MARKERS = (
    "执行点击",
    "执行单指操作",
    "执行双指操作",
    "执行 uinput",
    "执行按下",
    "执行抬起",
    "touch_down",
    "touch_move",
    "touch_up",
    "move_to",
    "move_press",
    "move_up",
    "控点",
    "按钮",
    "摇杆",
)
LOGIC_LOG_MARKERS = (
    "[AutoLog]",
    "[Parachute]",
    "[Searching]",
    "[搜房]",
    "[SceneSearch]",
    "[SceneEntry]",
    "[SceneRotate]",
    "[SceneExit]",
    "[HouseExit]",
    "[Nav]",
    "[NavBypass]",
    "[Unstuck]",
    "[Jump]",
    "[Smart]",
    "[Running]",
    "[Driving]",
    "[Entry]",
    "[Interact]",
    "[Scan]",
    "[Visual]",
    "[Finish]",
    "[Flow]",
    "[TurnCalibration]",
)
RESTART_BAT_CMD_TITLE = "AutoGame restart.bat"
SYSTEM_LOG_MARKERS = (
    "[Launcher]",
    "[Stream]",
    "[HDC]",
    "[StartGame]",
    "[Popup]",
    "[End]",
    "[FrameWorker]",
    "[Visualizer]",
    "[Resolution]",
    "[Rotation]",
    "[Data]",
    "[Log]",
    "[ERROR]",
    "hdc ",
    "hdc.exe",
    "shell",
    "subprocess",
    "force-stop",
    "Traceback",
    "Exception",
    "命令执行失败",
    "成功加载业务逻辑",
)


def classify_output_line(line: str) -> str:
    text = str(line or "").strip()
    if not text:
        return LOG_CATEGORY_OTHER

    frame_log_entry = parse_frame_log_transport(text)
    if frame_log_entry:
        return frame_log_entry["category"]

    structured_match = STRUCTURED_LOG_RE.match(text)
    if structured_match:
        category = structured_match.group("category")
        if category in LOG_CATEGORIES:
            return category
        return LOG_CATEGORY_LOGIC

    if any(marker in text for marker in TIME_LOG_MARKERS):
        return LOG_CATEGORY_TIME
    if any(marker in text for marker in LOGIC_LOG_MARKERS):
        return LOG_CATEGORY_LOGIC
    if any(marker in text for marker in UI_LOG_MARKERS):
        return LOG_CATEGORY_UI
    if any(marker in text for marker in SYSTEM_LOG_MARKERS):
        return LOG_CATEGORY_SYSTEM
    return LOG_CATEGORY_OTHER


def decode_output_line(line: str) -> tuple[str, str]:
    raw_line = str(line or "")
    frame_log_entry = parse_frame_log_transport(raw_line)
    if not frame_log_entry:
        return classify_output_line(raw_line), raw_line

    if raw_line.endswith("\r\n"):
        line_ending = "\r\n"
    elif raw_line.endswith("\n"):
        line_ending = "\n"
    elif raw_line.endswith("\r"):
        line_ending = "\r"
    else:
        line_ending = ""
    return (
        frame_log_entry["category"],
        frame_log_entry["message"] + line_ending,
    )


def decode_output_text(text: str) -> list[tuple[str, str]]:
    return [
        decode_output_line(line)
        for line in str(text or "").splitlines(keepends=True)
    ]


def filter_output_text(text: str, selected_filter: str) -> str:
    return "".join(
        line
        for category, line in decode_output_text(text)
        if selected_filter == LOG_FILTER_ALL or category == selected_filter
    )


class CaptureStreamCheckResult(NamedTuple):
    ok: bool
    message: str


def _run_log_path_from_environment() -> Optional[Path]:
    archive_dir = str(os.environ.get("AUTOGAME_RUN_ARCHIVE_DIR") or "").strip()
    if not archive_dir:
        return None
    return Path(archive_dir).expanduser().resolve() / "launcher_debug.log"


def set_launcher_log_file(log_file: Optional[Path]) -> Optional[Path]:
    for handler in list(LOGGER.handlers):
        if not getattr(handler, LAUNCHER_FILE_HANDLER_MARKER, False):
            continue
        LOGGER.removeHandler(handler)
        handler.close()

    if log_file is None:
        return None

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [pid=%(process)d] %(message)s"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    setattr(file_handler, LAUNCHER_FILE_HANDLER_MARKER, True)
    LOGGER.addHandler(file_handler)
    LOGGER.info("launcher run logging attached, log_file=%s", log_file)
    return log_file


def setup_logging():
    LOGGER.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [pid=%(process)d] %(message)s"
    )

    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in LOGGER.handlers
    ):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)

    LOGGER.propagate = False
    LOGGER.info("launcher logging initialized; run logs are stored per run")
    set_launcher_log_file(_run_log_path_from_environment())


def log_exception(context: str, exc_info=None):
    LOGGER.exception("%s", context, exc_info=exc_info)


def install_global_exception_hooks():
    def _excepthook(exc_type, exc_value, exc_traceback):
        log_exception(
            "uncaught exception",
            (exc_type, exc_value, exc_traceback),
        )
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _excepthook


def ensure_pyqt6_platform_plugin_path():
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return

    try:
        import PyQt6

        platforms_dir = Path(PyQt6.__file__).resolve().parent / "Qt6" / "plugins" / "platforms"
    except Exception:
        LOGGER.debug("failed to resolve PyQt6 platform plugin path", exc_info=True)
        return

    if platforms_dir.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)
        LOGGER.debug("QT_QPA_PLATFORM_PLUGIN_PATH set to %s", platforms_dir)


def parse_case_vars(py_file: Path) -> Dict[str, str]:
    LOGGER.debug("parse_case_vars: file=%s", py_file)
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    result: Dict[str, str] = {}

    for node in tree.body:
        target_name = None
        value_node = None

        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target_name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value

        if target_name not in {
            "project_case",
            "target_case",
            "testcase_description",
        } or value_node is None:
            continue

        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
            result[target_name] = value_node.value

    return result


def extract_package_names(py_file: Path) -> list[str]:
    LOGGER.debug("extract_package_names: file=%s exists=%s", py_file, py_file.exists())
    if not py_file.exists():
        return []

    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
    except Exception:
        return []

    packages = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if PACKAGE_NAME_RE.fullmatch(value):
                packages.add(value)
    return sorted(packages)


def is_pubg_testcase_target_case(target_case: Optional[str]) -> bool:
    return str(target_case or "").strip() == PUBG_CASE_TARGET_CASE


def is_pubg_testcase_file(py_file: Optional[Path], parsed: Optional[dict] = None) -> bool:
    if parsed is None and py_file is not None:
        try:
            parsed = parse_case_vars(py_file)
        except Exception:
            parsed = {}
    parsed = parsed or {}
    return is_pubg_testcase_target_case(parsed.get("target_case"))


def resolve_label_project_dir(project_case: str) -> Optional[Path]:
    project_case = str(project_case or "").strip()
    if not project_case:
        return None

    project_dir = CUSTOMS_EXAMPLES_DIR / project_case
    if (project_dir / "info.py").exists():
        return project_dir
    return None


def get_testcase_button_texts(has_selection: bool) -> tuple[str, str]:
    return ("已选择" if has_selection else "选择用例", "重选")


def is_multiprocessing_child(argv: Optional[list[str]] = None) -> bool:
    argv = argv or sys.argv
    return any(str(arg) == "--multiprocessing-fork" for arg in argv)


def apply_pyinstaller_splash_suppression(env) -> None:
    env.insert(PYINSTALLER_SUPPRESS_SPLASH_ENV, "1")
    env.insert(PYINSTALLER_SPLASH_IPC_ENV, "0")


def close_pyinstaller_splash(context: str) -> bool:
    try:
        import pyi_splash
    except Exception:
        return False

    try:
        pyi_splash.close()
        LOGGER.info("pyinstaller splash closed: context=%s", context)
        return True
    except Exception:
        LOGGER.debug("pyinstaller splash close failed: context=%s", context, exc_info=True)
        return False


def resolve_screen_mode_for_test_profile(
    test_profile: str,
    target_case: Optional[str] = None,
    config_path: Path = AUTOGAME_CONFIG_FILE,
) -> str:
    del test_profile, target_case
    return read_screen_mode_config(config_path)


def resolve_test_profile_from_radio_selection(
    power_checked: bool,
    function_checked: bool,
    marathon_checked: bool = False,
) -> str:
    if marathon_checked:
        return TEST_PROFILE_MARATHON
    if function_checked:
        return "function"
    return "power"


def normalize_launcher_screen_mode(screen_mode: str) -> str:
    raw_mode = str(screen_mode).strip()
    if raw_mode not in {"0", "1", "2"}:
        raise ValueError(f"unsupported screen_mode: {screen_mode}")
    return raw_mode


def read_screen_mode_config(config_path: Path = AUTOGAME_CONFIG_FILE) -> str:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a json object: {config_path}")
    return normalize_launcher_screen_mode(config.get("screen_mode", "0"))


def write_screen_mode_config(screen_mode: str, config_path: Path = AUTOGAME_CONFIG_FILE) -> None:
    config_path = Path(config_path)
    screen_mode = normalize_launcher_screen_mode(screen_mode)

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"config must be a json object: {config_path}")

    config["screen_mode"] = screen_mode
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    tmp_path.replace(config_path)


def read_hoscrcpy_frame_rate_config(config_path: Path = AUTOGAME_CONFIG_FILE) -> int:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a json object: {config_path}")

    try:
        frame_rate = int(config.get("hoscrcpy_frame_rate"))
    except (TypeError, ValueError) as exc:
        raise ValueError("hoscrcpy_frame_rate must be an integer") from exc
    if frame_rate not in HOSCRCPY_FRAME_RATE_OPTIONS:
        allowed = ", ".join(str(value) for value in HOSCRCPY_FRAME_RATE_OPTIONS)
        raise ValueError(f"hoscrcpy_frame_rate must be one of: {allowed}")
    return frame_rate


def write_hoscrcpy_frame_rate_config(
    frame_rate: int,
    config_path: Path = AUTOGAME_CONFIG_FILE,
) -> None:
    config_path = Path(config_path)
    try:
        frame_rate = int(frame_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("hoscrcpy_frame_rate must be an integer") from exc
    if frame_rate not in HOSCRCPY_FRAME_RATE_OPTIONS:
        allowed = ", ".join(str(value) for value in HOSCRCPY_FRAME_RATE_OPTIONS)
        raise ValueError(f"hoscrcpy_frame_rate must be one of: {allowed}")

    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"config must be a json object: {config_path}")

    config["hoscrcpy_frame_rate"] = frame_rate
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    tmp_path.replace(config_path)


def build_launcher_plan_env_values(plan: Optional[dict]) -> dict[str, str]:
    plan = plan or {}
    test_profile = str(plan.get("test_profile") or "power")
    target_case = str(plan.get("target_case") or "")
    screen_mode = normalize_launcher_screen_mode(
        plan.get("screen_mode") or resolve_screen_mode_for_test_profile(test_profile, target_case)
    )
    case_loop_count = int(plan.get("case_loop_count") or 1)
    marathon_duration_minutes = max(
        0.0,
        float(plan.get("marathon_duration_minutes") or 0.0),
    )
    marathon_end_battery_percent = max(
        0,
        min(100, int(plan.get("marathon_end_battery_percent") or 0)),
    )
    env_values = {
        "AUTOGAME_TEST_PROFILE": test_profile,
        "AUTOGAME_SCREEN_MODE": screen_mode,
        "AUTOGAME_SINGLE_CASE_LOOPS": str(max(1, case_loop_count)),
        "AUTOGAME_MARATHON_DURATION_MINUTES": str(marathon_duration_minutes),
        "AUTOGAME_MARATHON_END_BATTERY_PERCENT": str(marathon_end_battery_percent),
        "AUTOGAME_SP_RECORDING_ENABLED": "1" if should_use_sp_recording_for_profile(test_profile) else "0",
        "AUTOGAME_PRESERVE_GAME_PROCESS": (
            "1" if should_preserve_game_process_for_plan(plan) else "0"
        ),
        "AUTOGAME_DISABLE_SAVE_FRAMES": "1",
        "AUTOGAME_TMP_FRAMES_DIR": str(TEMP_DIR / "tmp_frames"),
    }
    screen_width = _positive_int(plan.get("screen_width"))
    screen_height = _positive_int(plan.get("screen_height"))
    if screen_width and screen_height:
        env_values["AUTOGAME_SCREEN_WIDTH"] = str(screen_width)
        env_values["AUTOGAME_SCREEN_HEIGHT"] = str(screen_height)
    return env_values


def should_preserve_game_process_for_plan(plan: Optional[dict]) -> bool:
    plan = plan or {}
    if is_marathon_plan(plan):
        return False
    return bool(plan.get("preserve_game_process"))


def is_marathon_plan(plan: Optional[dict]) -> bool:
    plan = plan or {}
    if str(plan.get("test_profile") or "").strip().lower() == TEST_PROFILE_MARATHON:
        return True
    try:
        return float(plan.get("marathon_duration_minutes") or 0.0) > 0
    except (TypeError, ValueError):
        return False


def has_reached_plan_run_limit(plan: Optional[dict], completed_runs: int) -> bool:
    plan = plan or {}
    if is_marathon_plan(plan):
        return False
    try:
        run_count = max(1, int(plan.get("run_count") or 1))
    except (TypeError, ValueError):
        run_count = 1
    return max(0, int(completed_runs)) >= run_count


def _decode_process_output(output) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, (bytes, bytearray)):
        for encoding in ("utf-8", "gbk", "cp936"):
            try:
                return bytes(output).decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return bytes(output).decode("utf-8", errors="replace")
    return str(output)


def _completed_process_text(result: subprocess.CompletedProcess) -> str:
    stdout = _decode_process_output(result.stdout)
    stderr = _decode_process_output(result.stderr)
    text = (stdout + stderr).strip()
    return text[:500]


def pull_saved_sp_artifacts(
    run_archive_dir: Path,
    run_index: int,
    hdc_executable: Optional[str] = None,
    run_command=subprocess.run,
) -> tuple[Path, list[dict]]:
    batch_dir = Path(run_archive_dir).parent
    record_dir = batch_dir / f"第{max(1, int(run_index))}次sp记录"
    for local_name in ("db", "data", "daemon"):
        (record_dir / local_name).mkdir(parents=True, exist_ok=True)

    hdc = hdc_executable or resolve_hdc_executable()
    results = []
    for local_name, remote_path in SP_ARTIFACT_REMOTE_PATHS:
        local_path = record_dir / local_name
        command = [hdc, "file", "recv", remote_path, str(local_path)]
        try:
            completed = run_command(
                command,
                cwd=str(APP_DIR),
                capture_output=True,
                timeout=SP_ARTIFACT_PULL_TIMEOUT_SECONDS,
                **hidden_subprocess_kwargs(),
            )
            results.append(
                {
                    "remote_path": remote_path,
                    "local_path": str(local_path),
                    "ok": completed.returncode == 0,
                    "detail": _completed_process_text(completed),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "remote_path": remote_path,
                    "local_path": str(local_path),
                    "ok": False,
                    "detail": str(exc),
                }
            )
    return record_dir, results


def check_capture_stream_for_screen_mode(
    screen_mode: str,
    temp_root: Path = TEMP_DIR,
    timeout: float = 8.0,
) -> CaptureStreamCheckResult:
    screen_mode = str(screen_mode).strip()
    if screen_mode == "0":
        return CaptureStreamCheckResult(
            True,
            "低功耗拉流模式会在用例启动后由 launcher 监听首帧和断流信号。",
        )
    if screen_mode == "2":
        return CaptureStreamCheckResult(
            True,
            "HOScrcpy 拉流模式会在用例启动后自动推送并启动手机端投屏服务。",
        )
    if screen_mode != "1":
        return CaptureStreamCheckResult(False, f"未知 screen_mode: {screen_mode}")

    from PIL import Image

    temp_root = Path(temp_root)
    check_dir = temp_root / "launcher_capture_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    remote_path = f"/data/local/tmp/autogame_launcher_capture_check_{timestamp}.jpeg"
    local_path = check_dir / f"capture_check_{timestamp}.jpeg"
    hdc_executable = resolve_hdc_executable()

    try:
        snap_result = subprocess.run(
            [hdc_executable, "shell", "snapshot_display", "-f", remote_path],
            cwd=str(APP_DIR),
            capture_output=True,
            timeout=timeout,
            **hidden_subprocess_kwargs(),
        )
        if snap_result.returncode != 0:
            return CaptureStreamCheckResult(
                False,
                f"HDC 截图失败: {_completed_process_text(snap_result)}",
            )

        recv_result = subprocess.run(
            [hdc_executable, "file", "recv", remote_path, str(local_path)],
            cwd=str(APP_DIR),
            capture_output=True,
            timeout=timeout,
            **hidden_subprocess_kwargs(),
        )
        if recv_result.returncode != 0:
            return CaptureStreamCheckResult(
                False,
                f"HDC 拉取截图失败: {_completed_process_text(recv_result)}",
            )

        if not local_path.exists() or local_path.stat().st_size <= 0:
            return CaptureStreamCheckResult(False, "HDC 截图文件为空")

        with Image.open(local_path) as img:
            img.verify()
            width, height = img.size
        if width <= 0 or height <= 0:
            return CaptureStreamCheckResult(False, "HDC 截图尺寸异常")

        return CaptureStreamCheckResult(True, f"HDC 截图预检通过: {width}x{height}")
    except subprocess.TimeoutExpired as exc:
        return CaptureStreamCheckResult(False, f"HDC 截图预检超时: {exc}")
    except Exception as exc:
        return CaptureStreamCheckResult(False, f"HDC 截图预检异常: {exc}")
    finally:
        try:
            subprocess.run(
                [hdc_executable, "shell", "rm", remote_path],
                cwd=str(APP_DIR),
                capture_output=True,
                timeout=2,
                **hidden_subprocess_kwargs(),
            )
        except Exception:
            pass
        try:
            if local_path.exists():
                local_path.unlink()
        except Exception:
            pass


def stream_frame_to_qpixmap(frame) -> QPixmap:
    if frame is None:
        return QPixmap()

    try:
        if hasattr(frame, "convert") and hasattr(frame, "size"):
            rgb_frame = frame.convert("RGB")
            width, height = rgb_frame.size
            raw = rgb_frame.tobytes("raw", "RGB")
            image = QImage(raw, width, height, width * 3, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(image.copy())

        import numpy as np

        array = np.asarray(frame)
        if array.size <= 0:
            return QPixmap()
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            gray = np.ascontiguousarray(array)
            height, width = gray.shape
            image = QImage(
                gray.data,
                width,
                height,
                gray.strides[0],
                QImage.Format.Format_Grayscale8,
            )
            return QPixmap.fromImage(image.copy())

        if array.ndim == 3 and array.shape[2] >= 3:
            rgb = np.ascontiguousarray(array[:, :, :3])
            height, width, _channels = rgb.shape
            image = QImage(
                rgb.data,
                width,
                height,
                rgb.strides[0],
                QImage.Format.Format_RGB888,
            )
            return QPixmap.fromImage(image.copy())
    except Exception:
        log_exception("stream frame to pixmap failed")

    return QPixmap()


def _positive_int(value) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _size_from_mapping(value) -> tuple[Optional[int], Optional[int]]:
    if not isinstance(value, dict):
        return None, None
    width = _positive_int(value.get("width") or value.get("screen_width"))
    height = _positive_int(value.get("height") or value.get("screen_height"))
    if width and height:
        return width, height

    for key in ("size", "screen_size", "frame_size", "resolution"):
        raw_size = value.get(key)
        if isinstance(raw_size, dict):
            width, height = _size_from_mapping(raw_size)
            if width and height:
                return width, height
        elif isinstance(raw_size, (list, tuple)) and len(raw_size) >= 2:
            width = _positive_int(raw_size[0])
            height = _positive_int(raw_size[1])
            if width and height:
                return width, height

    return None, None


def resolve_preview_render_screen_size(
    payload,
    pixmap=None,
    locked_screen_size=None,
) -> tuple[Optional[int], Optional[int]]:
    payload = payload if isinstance(payload, dict) else {}
    locked_width = None
    locked_height = None
    if isinstance(locked_screen_size, (list, tuple)) and len(locked_screen_size) >= 2:
        locked_width = _positive_int(locked_screen_size[0])
        locked_height = _positive_int(locked_screen_size[1])

    for source in (payload.get("screen"), payload):
        width, height = _size_from_mapping(source)
        if width and height:
            return width, height

    if locked_width and locked_height:
        return locked_width, locked_height

    frame_info = payload.get("frame")
    width, height = _size_from_mapping(frame_info)
    if width and height:
        return width, height

    if pixmap is not None:
        width = _positive_int(pixmap.width())
        height = _positive_int(pixmap.height())
        if width and height:
            return width, height

    return None, None


def resolve_preview_payload_stage_name(payload) -> str:
    payload = payload if isinstance(payload, dict) else {}

    for source in (
        payload.get("stage"),
        payload.get("phase"),
        (payload.get("semantic_log") or {}).get("current_stage")
        if isinstance(payload.get("semantic_log"), dict)
        else None,
    ):
        if isinstance(source, dict):
            for key in ("name", "stage"):
                value = str(source.get(key) or "").strip()
                if value:
                    return value
        elif source:
            value = str(source).strip()
            if value:
                return value

    return ""


def resolve_preview_payload_group_name(payload) -> str:
    payload = payload if isinstance(payload, dict) else {}

    for source in (
        payload.get("stage"),
        payload.get("phase"),
        (payload.get("semantic_log") or {}).get("current_stage")
        if isinstance(payload.get("semantic_log"), dict)
        else None,
    ):
        if not isinstance(source, dict):
            continue
        value = str(source.get("group") or "").strip()
        if value:
            return value

    return ""


PREVIEW_INFO_VALUE_MAX_LENGTH = 50


def _format_timed_preview_info(value):
    """Put decorated special-area inference time before its result."""
    candidate = value
    if isinstance(candidate, str):
        candidate_text = candidate.strip()
        if not candidate_text.startswith(("[", "(")):
            return None
        try:
            candidate = ast.literal_eval(candidate_text)
        except (SyntaxError, ValueError):
            return None

    if not isinstance(candidate, (list, tuple)) or len(candidate) != 2:
        return None
    timing = candidate[1]
    if not isinstance(timing, (list, tuple)) or len(timing) != 1:
        return None
    try:
        elapsed_ms = float(timing[0])
    except (TypeError, ValueError):
        return None

    info = candidate[0]
    if isinstance(info, str):
        info_text = info
    elif isinstance(info, (list, tuple, dict)):
        info_text = json.dumps(
            info,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    else:
        info_text = str(info)

    timing_text = f"{elapsed_ms:.3f}".rstrip("0").rstrip(".")
    return f"[{timing_text} ms], {info_text}"


def _truncate_preview_info_value(value, max_length: int = PREVIEW_INFO_VALUE_MAX_LENGTH):
    """Keep a single preview info value compact without changing source data."""
    timed_display = _format_timed_preview_info(value)
    if timed_display is not None:
        display_text = timed_display
    elif isinstance(value, str):
        display_text = value
    elif isinstance(value, (list, tuple, dict)):
        display_text = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    else:
        display_text = str(value)

    if len(display_text) <= max_length:
        return display_text if timed_display is not None else value
    return f"{display_text[: max_length - 3]}..."


def format_preview_frame_info(payload) -> str:
    """Render only the current frame's visual info, never runtime logs."""
    payload = payload if isinstance(payload, dict) else {}
    stage_name = resolve_preview_payload_stage_name(payload)
    info_payload = payload.get("info")
    if isinstance(info_payload, dict):
        display_info = {}
        if stage_name:
            display_info["stage"] = stage_name
        display_info.update(
            {
                key: _truncate_preview_info_value(value)
                for key, value in info_payload.items()
            }
        )
        if stage_name:
            display_info["stage"] = stage_name
        return (
            json.dumps(display_info, ensure_ascii=False, indent=2)
            if display_info
            else "当前帧暂无画面识别信息。"
        )
    if info_payload is not None:
        if stage_name:
            return json.dumps(
                {"stage": stage_name, "info": info_payload},
                ensure_ascii=False,
                indent=2,
            )
        return str(info_payload)
    if stage_name:
        return json.dumps({"stage": stage_name}, ensure_ascii=False, indent=2)
    return "当前帧暂无画面识别信息。"


def format_preview_info_item_value(value, max_length: int = PREVIEW_INFO_VALUE_MAX_LENGTH) -> str:
    """Format one recognition result for a stable preview-list cell."""
    compact_value = _truncate_preview_info_value(value, max_length=max_length)
    if isinstance(compact_value, str):
        return compact_value
    if isinstance(compact_value, (list, tuple, dict)):
        return json.dumps(
            compact_value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    return str(compact_value)


def format_preview_info_detail(value) -> str:
    """Format one selected item's full frame value without truncating it."""
    candidate = value
    if isinstance(candidate, str):
        candidate_text = candidate.strip()
        if not candidate_text:
            return "(空字符串)"
        if candidate_text.startswith(("[", "{", "(")):
            try:
                candidate = json.loads(candidate_text)
            except (json.JSONDecodeError, TypeError):
                try:
                    candidate = ast.literal_eval(candidate_text)
                except (SyntaxError, ValueError):
                    return candidate
        else:
            return candidate

    if isinstance(candidate, (list, tuple, dict)):
        return json.dumps(
            candidate,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    return str(candidate)


def resolve_preview_group_filter(stage_entry, group_name: str):
    """Match the runtime stage-group selection used by StageLogicController."""
    group_name = str(group_name or "").strip()
    if not group_name or group_name == "默认" or not isinstance(stage_entry, dict):
        return None

    groups = stage_entry.get("groups", {})
    if not isinstance(groups, dict):
        return None
    group_data = groups.get(group_name)
    if group_data is None:
        return set()
    if isinstance(group_data, dict) and group_data.get("all"):
        return None

    allowed = set()
    raw_items = group_data.get("items", []) if isinstance(group_data, dict) else []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        scene_name = str(item.get("scene") or "").strip()
        item_type = str(item.get("type") or "").strip()
        item_name = str(item.get("name") or "").strip()
        if scene_name and item_name and item_type in {"area", "special_area"}:
            allowed.add((scene_name, item_type, item_name))
    return allowed


def is_preview_stage_item_visible(group_filter, scene_name: str, item_type: str, item_name: str) -> bool:
    if group_filter is None:
        return True

    scene_name = str(scene_name or "").strip()
    if item_type == "points":
        # Runtime groups only enumerate recognition areas. Points belong to an
        # enabled scene, so keep that scene's actionable controls visible.
        return any(scene == scene_name for scene, _kind, _name in group_filter)

    group_item_type = {"areas": "area", "special_areas": "special_area"}.get(item_type)
    return bool(group_item_type) and (scene_name, group_item_type, item_name) in group_filter


def resolve_preview_area_search_scope(area_data):
    """Return a drawable configured search scope for one template area."""
    if not isinstance(area_data, dict):
        return None

    scope = area_data.get("search_scope")
    if not isinstance(scope, (list, tuple)) or len(scope) != 4:
        return None

    try:
        x1, y1, x2, y2 = (float(value) for value in scope)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return {"rect": scope}


def stream_disconnect_policy_for_screen_mode(screen_mode: str) -> str:
    mode = str(screen_mode).strip()
    if mode == "1":
        return STREAM_DISCONNECT_POLICY_DISABLED
    if mode == "2":
        return STREAM_DISCONNECT_POLICY_STREAM_ONLY
    return STREAM_DISCONNECT_POLICY_PRESERVE


def _powershell_single_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def is_process_trace_enabled(os_name: Optional[str] = None) -> bool:
    if (os_name or os.name) != "nt":
        return False

    value = os.environ.get(PROCESS_TRACE_ENV, "").strip().lower()
    if value in {"0", "false", "no", "off", "disabled"}:
        return False
    return True


class WindowsProcessLaunchTracer:
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        os_name: Optional[str] = None,
        root_pid: Optional[int] = None,
    ):
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.os_name = os_name
        self.root_pid = int(root_pid or os.getpid())
        self.log_path: Optional[Path] = None
        self._proc: Optional[subprocess.Popen] = None

    def start(self, label: str) -> Optional[Path]:
        if self._proc is not None:
            return self.log_path
        if not is_process_trace_enabled(self.os_name):
            LOGGER.info(
                "process launch trace disabled: os_name=%s env=%s",
                self.os_name or os.name,
                os.environ.get(PROCESS_TRACE_ENV),
            )
            return None

        if self.log_dir is None:
            LOGGER.warning("process launch trace skipped: no run archive directory")
            return None

        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        self.log_path = self.log_dir / f"process_launch_trace_{timestamp}_{self.root_pid}.log"
        script = self._build_powershell_script(self.log_path, label)
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-WindowStyle",
            "Hidden",
            "-EncodedCommand",
            encoded,
        ]

        try:
            self._proc = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **hidden_subprocess_kwargs(os_name=self.os_name),
            )
            LOGGER.info(
                "process launch trace started: pid=%s log_path=%s label=%s root_pid=%s",
                self._proc.pid,
                self.log_path,
                label,
                self.root_pid,
            )
            return self.log_path
        except Exception:
            log_exception("process launch trace start failed")
            self._proc = None
            return None

    def stop(self) -> Optional[Path]:
        proc = self._proc
        self._proc = None
        if proc is None:
            return self.log_path

        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            LOGGER.info(
                "process launch trace stopped: pid=%s returncode=%s log_path=%s",
                proc.pid,
                proc.returncode,
                self.log_path,
            )
        except Exception:
            log_exception("process launch trace stop failed")
        return self.log_path

    def _build_powershell_script(self, log_path: Path, label: str) -> str:
        log_path_literal = _powershell_single_quote(str(log_path))
        label_literal = _powershell_single_quote(label)
        return f"""
$ErrorActionPreference = 'SilentlyContinue'
$global:AutoGameTraceLogPath = {log_path_literal}
$global:AutoGameTraceRootPid = {self.root_pid}
$global:AutoGameTraceLabel = {label_literal}

Add-Content -LiteralPath $global:AutoGameTraceLogPath -Encoding UTF8 -Value ("{{0:o}}`tTRACE_START`tRootPID={{1}}`tLabel={{2}}" -f (Get-Date), $global:AutoGameTraceRootPid, $global:AutoGameTraceLabel)

function Get-AutoGameProcesses {{
    $items = Get-CimInstance Win32_Process
    if (-not $items) {{ $items = Get-WmiObject Win32_Process }}
    return $items
}}

function Get-AutoGameProcessById([int]$ProcessIdValue) {{
    $item = Get-CimInstance Win32_Process -Filter ("ProcessId={{0}}" -f $ProcessIdValue)
    if (-not $item) {{ $item = Get-WmiObject Win32_Process -Filter ("ProcessId={{0}}" -f $ProcessIdValue) }}
    return $item
}}

function Write-AutoGameProcessLine([string]$Kind, [int]$ProcessIdValue, [int]$ParentProcessIdValue, [string]$FallbackName) {{
    $proc = Get-AutoGameProcessById $ProcessIdValue
    $parent = Get-AutoGameProcessById $ParentProcessIdValue

    $nameValue = $FallbackName
    $pathValue = ""
    $cmdValue = ""
    $parentNameValue = ""
    $parentCmdValue = ""
    if ($proc) {{
        if ($proc.Name) {{ $nameValue = [string]$proc.Name }}
        if ($proc.ExecutablePath) {{ $pathValue = [string]$proc.ExecutablePath }}
        if ($proc.CommandLine) {{ $cmdValue = [string]$proc.CommandLine }}
    }}
    if ($parent) {{
        if ($parent.Name) {{ $parentNameValue = [string]$parent.Name }}
        if ($parent.CommandLine) {{ $parentCmdValue = [string]$parent.CommandLine }}
    }}

    Add-Content -LiteralPath $global:AutoGameTraceLogPath -Encoding UTF8 -Value ("{{0:o}}`t{{1}}`tPID={{2}}`tPPID={{3}}`tName={{4}}`tPath={{5}}`tCmd={{6}}`tParentName={{7}}`tParentCmd={{8}}" -f (Get-Date), $Kind, $ProcessIdValue, $ParentProcessIdValue, $nameValue, $pathValue, $cmdValue, $parentNameValue, $parentCmdValue)
}}

$seen = @{{}}
foreach ($proc in Get-AutoGameProcesses) {{
    $seen[[int]$proc.ProcessId] = $true
}}

try {{
    Register-WmiEvent -Class Win32_ProcessStartTrace -SourceIdentifier AutoGameProcessTrace | Out-Null
    Add-Content -LiteralPath $global:AutoGameTraceLogPath -Encoding UTF8 -Value ("{{0:o}}`tTRACE_READY`tMode=event+poll" -f (Get-Date))
}} catch {{
    Add-Content -LiteralPath $global:AutoGameTraceLogPath -Encoding UTF8 -Value ("{{0:o}}`tTRACE_READY`tMode=poll`tRegisterError={{1}}" -f (Get-Date), $_.Exception.Message)
}}

while ($true) {{
    foreach ($eventItem in Get-Event -SourceIdentifier AutoGameProcessTrace) {{
        $e = $eventItem.SourceEventArgs.NewEvent
        $pidValue = [int]$e.ProcessID
        $parentPidValue = [int]$e.ParentProcessID
        $seen[$pidValue] = $true
        Write-AutoGameProcessLine "EVENT_CREATE" $pidValue $parentPidValue ([string]$e.ProcessName)
        Remove-Event -EventIdentifier $eventItem.EventIdentifier
    }}

    $currentSeen = @{{}}
    foreach ($proc in Get-AutoGameProcesses) {{
        $pidValue = [int]$proc.ProcessId
        $currentSeen[$pidValue] = $true
        if (-not $seen.ContainsKey($pidValue)) {{
            $seen[$pidValue] = $true
            Write-AutoGameProcessLine "POLL_CREATE" $pidValue ([int]$proc.ParentProcessId) ([string]$proc.Name)
        }}
    }}
    $seen = $currentSeen

    Start-Sleep -Milliseconds 1000
}}
"""


def terminate_popen_process_tree(proc: subprocess.Popen, force: bool) -> None:
    """Terminate a subprocess and its descendants without depending on Qt."""
    if proc is None or proc.poll() is not None:
        return

    if os.name == "nt":
        if not force:
            ctrl_break_event = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break_event is not None:
                try:
                    proc.send_signal(ctrl_break_event)
                    return
                except Exception:
                    LOGGER.debug(
                        "graceful CTRL_BREAK_EVENT failed, fallback to taskkill: pid=%s",
                        proc.pid,
                        exc_info=True,
                    )
        command = ["taskkill", "/PID", str(int(proc.pid)), "/T"]
        if force:
            command.append("/F")
        try:
            subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                **hidden_subprocess_kwargs(),
            )
            return
        except Exception:
            pass
    elif getattr(proc, "pid", None):
        try:
            pgid = os.getpgid(int(proc.pid))
            os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
            return
        except Exception:
            pass

    try:
        if force:
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        pass


class HiddenSubprocess(QObject):
    readyReadStandardOutput = pyqtSignal()
    finished = pyqtSignal(int, object)
    errorOccurred = pyqtSignal(object)
    FORCED_STOP_FINISH_TIMEOUT_SECONDS = 5.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._program = ""
        self._arguments: list[str] = []
        self._working_directory = None
        self._environment = None
        self._proc: Optional[subprocess.Popen] = None
        self._state = QProcess.ProcessState.NotRunning
        self._error = QProcess.ProcessError.UnknownError
        self._error_string = ""
        self._output_buffer = bytearray()
        self._output_lock = threading.Lock()
        self._finish_lock = threading.Lock()
        self._finished_emitted = False
        self._forced_stop_watcher_started = False
        self._forced_stop_requested = False

    def setProgram(self, program: str):
        self._program = str(program)

    def setWorkingDirectory(self, working_directory: str):
        self._working_directory = str(working_directory)

    def setProcessChannelMode(self, _mode):
        pass

    def setProcessEnvironment(self, environment):
        self._environment = environment

    def setArguments(self, arguments):
        self._arguments = [str(arg) for arg in arguments]

    def start(self):
        self._state = QProcess.ProcessState.Starting
        command = [self._program, *self._arguments]
        popen_kwargs = hidden_subprocess_kwargs()
        if os.name == "nt":
            create_new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if create_new_group:
                popen_kwargs["creationflags"] = int(popen_kwargs.get("creationflags", 0)) | create_new_group
        else:
            popen_kwargs["start_new_session"] = True

        try:
            self._proc = subprocess.Popen(
                command,
                cwd=self._working_directory,
                env=self._environment_to_dict(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                **popen_kwargs,
            )
        except Exception as exc:
            self._proc = None
            self._state = QProcess.ProcessState.NotRunning
            self._error = QProcess.ProcessError.FailedToStart
            self._error_string = str(exc)
            self.errorOccurred.emit(self._error)
            return

        self._state = QProcess.ProcessState.Running
        self._finished_emitted = False
        self._forced_stop_watcher_started = False
        self._forced_stop_requested = False
        reader = threading.Thread(target=self._read_process_output, daemon=True)
        reader.start()

    def waitForStarted(self, _msecs: int) -> bool:
        return self._proc is not None

    def state(self):
        if self._proc is not None and self._state == QProcess.ProcessState.Running:
            if self._proc.poll() is not None:
                self._state = QProcess.ProcessState.NotRunning
        return self._state

    def processId(self) -> int:
        if self._proc is None or self._proc.pid is None:
            return 0
        return int(self._proc.pid)

    def error(self):
        return self._error

    def errorString(self) -> str:
        return self._error_string

    def terminate(self):
        if self._proc is not None and self._proc.poll() is None:
            self._terminate_process_tree(force=False)

    def kill(self):
        if self._proc is None:
            return
        self._forced_stop_requested = True
        if self._proc.poll() is None:
            self._terminate_process_tree(force=True)
        self._close_stdout_pipe()
        self._start_forced_stop_watcher()

    def _terminate_process_tree(self, force: bool):
        proc = self._proc
        if proc is None:
            return
        terminate_popen_process_tree(proc, force=force)

    def _close_stdout_pipe(self):
        proc = self._proc
        stdout = getattr(proc, "stdout", None) if proc is not None else None
        if stdout is None:
            return
        try:
            stdout.close()
        except Exception:
            pass

    def _start_forced_stop_watcher(self):
        if self._forced_stop_watcher_started:
            return
        self._forced_stop_watcher_started = True
        watcher = threading.Thread(
            target=self._finish_after_forced_stop,
            name="hidden-subprocess-forced-stop",
            daemon=True,
        )
        watcher.start()

    def _finish_after_forced_stop(self):
        proc = self._proc
        if proc is None:
            return

        try:
            exit_code = proc.wait(timeout=self.FORCED_STOP_FINISH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._terminate_process_tree(force=True)
            try:
                exit_code = proc.wait(timeout=1)
            except Exception:
                exit_code = proc.returncode if proc.returncode is not None else -9
        except Exception:
            exit_code = proc.returncode if proc.returncode is not None else 1

        self._close_stdout_pipe()
        self._emit_finished_once(
            int(exit_code if exit_code is not None else -9),
            QProcess.ExitStatus.CrashExit,
        )

    def _emit_finished_once(self, exit_code: int, exit_status):
        with self._finish_lock:
            if self._finished_emitted:
                return
            self._finished_emitted = True
            self._state = QProcess.ProcessState.NotRunning
        self.finished.emit(int(exit_code), exit_status)

    def readAllStandardOutput(self) -> QByteArray:
        with self._output_lock:
            data = bytes(self._output_buffer)
            self._output_buffer.clear()
        return QByteArray(data)

    def _environment_to_dict(self) -> Optional[dict]:
        if self._environment is None:
            return None
        if hasattr(self._environment, "keys") and hasattr(self._environment, "value"):
            return {
                str(key): self._environment.value(str(key))
                for key in self._environment.keys()
            }
        if isinstance(self._environment, dict):
            return {str(key): str(value) for key, value in self._environment.items()}
        return None

    def _read_process_output(self):
        assert self._proc is not None
        try:
            if self._proc.stdout is not None:
                while True:
                    chunk = self._proc.stdout.read(4096)
                    if not chunk:
                        break
                    with self._output_lock:
                        self._output_buffer.extend(chunk)
                    self.readyReadStandardOutput.emit()
            exit_code = self._proc.wait()
            self._emit_finished_once(
                int(exit_code),
                QProcess.ExitStatus.NormalExit,
            )
        except Exception as exc:
            self._error = QProcess.ProcessError.Crashed
            self._error_string = str(exc)
            if not self._forced_stop_requested:
                self.errorOccurred.emit(self._error)
            self._emit_finished_once(
                self._proc.returncode if self._proc.returncode is not None else 1,
                QProcess.ExitStatus.CrashExit,
            )


def _count_files(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for child in path.rglob("*") if child.is_file())


def _count_frame_json_files(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for child in path.glob("frame_*.json") if child.is_file())


def _read_history_text(path: Path, max_chars: int = 200000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def _looks_like_history_archive_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "archive_info.json").is_file():
        return True
    if (path / "logs").is_dir():
        return True
    if any(
        (path / name).is_file()
        for name in ("launcher_output.txt", "hilog.txt", "hdc_debug.log", "memory.log")
    ):
        return True
    if (path / "process_temp_logs").is_dir():
        return True
    if (path / "process_save_frames").is_dir():
        return True
    if (path / "preview_10fps.mp4").is_file():
        return True
    return False


def _looks_like_history_batch_dir(path: Path) -> bool:
    """Accept both the current date_target-case layout and legacy archives."""
    if not path.exists() or not path.is_dir():
        return False
    name = path.name
    return name.startswith("game_cases_") or bool(
        re.fullmatch(r"\d{8,14}_.+", name)
    )


def _iter_history_archive_dirs(temp_dir: Path) -> list[Path]:
    archive_dirs: list[Path] = []
    seen: set[Path] = set()

    def add_archive_dir(path: Path):
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            return
        seen.add(key)
        archive_dirs.append(path)

    for batch_dir in sorted(temp_dir.iterdir()):
        if not _looks_like_history_batch_dir(batch_dir):
            continue
        run_dirs = [
            child
            for child in sorted(batch_dir.iterdir())
            if child.is_dir() and _looks_like_history_archive_dir(child)
        ]
        if run_dirs:
            for run_dir in run_dirs:
                add_archive_dir(run_dir)
        elif _looks_like_history_archive_dir(batch_dir):
            add_archive_dir(batch_dir)

    for info_path in sorted(temp_dir.rglob("archive_info.json")):
        add_archive_dir(info_path.parent)

    return archive_dirs


def _read_archive_metadata(info_path: Optional[Path]) -> dict:
    if info_path is None or not info_path.exists() or not info_path.is_file():
        return {}
    try:
        metadata = json.loads(info_path.read_text(encoding="utf-8"))
        return metadata if isinstance(metadata, dict) else {}
    except Exception:
        return {}


def discover_history_outputs(temp_dir: Path = TEMP_DIR) -> list[dict]:
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        return []

    records = []
    for archive_dir in _iter_history_archive_dirs(temp_dir):
        info_path = archive_dir / "archive_info.json"
        if not info_path.exists():
            info_path = None
        metadata = _read_archive_metadata(info_path)
        legacy_logs_dir = archive_dir / "logs"

        def history_log_path(name: str) -> Path:
            direct_path = archive_dir / name
            return direct_path if direct_path.is_file() else legacy_logs_dir / name

        launcher_output = _read_history_text(history_log_path("launcher_output.txt"))
        if not launcher_output:
            launcher_output = _read_history_text(history_log_path("launcher_output_partial.txt"))

        preview_video = archive_dir / "preview_10fps.mp4"
        hilog_path = history_log_path("hilog.txt")
        battery_log_path = archive_dir.parent / "battery.log"
        mtime_path = info_path or archive_dir
        record = {
            "archive_dir": archive_dir,
            "batch_dir": archive_dir.parent,
            "archive_info_path": info_path,
            "archive_time": str(metadata.get("archive_time") or ""),
            "run_index": metadata.get("run_index", ""),
            "mode": metadata.get("mode", ""),
            "project_case": metadata.get("project_case", ""),
            "target_case": metadata.get("target_case", ""),
            "testcase_label": metadata.get("testcase_label", ""),
            "exit_code": metadata.get("exit_code", ""),
            "timed_out": metadata.get("timed_out", ""),
            "marathon_duration_minutes": metadata.get("marathon_duration_minutes", ""),
            "marathon_end_battery_percent": metadata.get("marathon_end_battery_percent", ""),
            "battery_stop_requested": metadata.get("battery_stop_requested", ""),
            "stream_disconnected": metadata.get("stream_disconnected", ""),
            "stream_disconnect_startup": metadata.get("stream_disconnect_startup", ""),
            "archive_metadata": metadata,
            "launcher_output": launcher_output,
            "hilog_path": hilog_path,
            "hilog_exists": hilog_path.exists() and hilog_path.is_file(),
            "battery_log_path": battery_log_path,
            "battery_log_exists": battery_log_path.exists() and battery_log_path.is_file(),
            "log_file_count": sum(
                1
                for name in (
                    "launcher_output.txt",
                    "launcher_output_partial.txt",
                    "launcher_debug.log",
                    "hilog.txt",
                    "hdc_debug.log",
                    "memory.log",
                )
                if history_log_path(name).is_file()
            ),
            "process_temp_file_count": _count_files(archive_dir / "process_temp_logs"),
            "process_save_frame_count": _count_files(archive_dir / "process_save_frames"),
            "frame_log_count": _count_frame_json_files(archive_dir / "process_temp_logs"),
            "preview_video_path": preview_video,
            "preview_video_exists": preview_video.exists() and preview_video.is_file(),
            "mtime": mtime_path.stat().st_mtime,
        }
        records.append(record)

    records.sort(key=lambda item: (str(item.get("archive_time") or ""), float(item.get("mtime") or 0)), reverse=True)
    return records


def format_history_record_summary(record: dict) -> str:
    archive_dir = record.get("archive_dir")
    preview_text = "有" if record.get("preview_video_exists") else "无"
    launcher_text = "有" if str(record.get("launcher_output") or "").strip() else "无"
    hilog_text = "有" if record.get("hilog_exists") else "无"
    battery_text = "有" if record.get("battery_log_exists") else "无"
    frame_log_count = int(record.get("frame_log_count") or 0)
    lines = [
        f"Launcher 运行日志: {launcher_text}",
        f"hilog 日志: {hilog_text}",
        f"运行帧 JSON: {frame_log_count}",
        f"battery.log: {battery_text}",
        f"预览视频: {preview_text}",
        f"归档目录: {archive_dir}",
    ]
    return "\n".join(lines)


def _preview_frame_sequence(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.stem)
    if not match:
        return -1
    return int(match.group(1))


def find_latest_preview_frame(preview_dir: Path) -> Optional[Path]:
    pointer_path = preview_dir / LATEST_PREVIEW_POINTER_FILENAME
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    image_name = str(payload.get("image") or "").strip()
    if not image_name or Path(image_name).name != image_name:
        return None
    latest_image = preview_dir / image_name
    if (
        not latest_image.is_file()
        or latest_image.suffix.lower() not in PREVIEW_FRAME_SUFFIXES
        or not latest_image.stem.startswith("frame_")
    ):
        return None
    return latest_image


def _history_frame_sort_key(path: Path):
    return (_preview_frame_sequence(path), path.name)


def _read_json_payload(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "json 读取失败", "path": str(path)}
    return payload if isinstance(payload, dict) else {"raw": payload}


def load_history_frame_records(record: dict) -> list[dict]:
    if not isinstance(record, dict):
        return []
    archive_dir = Path(record.get("archive_dir") or "")
    frame_dir = archive_dir / "process_temp_logs"
    if not frame_dir.exists():
        frame_dir = archive_dir / "process_save_frames"
    if not frame_dir.exists():
        return []

    frames = []
    image_paths = [
        path
        for path in frame_dir.glob("frame_*")
        if path.is_file() and path.suffix.lower() in PREVIEW_FRAME_SUFFIXES
    ]
    for image_path in sorted(image_paths, key=_history_frame_sort_key):
        json_path = image_path.with_suffix(".json")
        payload = _read_json_payload(json_path) if json_path.exists() else {
            "schema_version": 1,
            "frame": {"image": image_path.name, "index": _preview_frame_sequence(image_path)},
            "stage": {"name": ""},
            "info": {},
            "frame_summary": "未找到同名帧 JSON。",
        }
        frames.append({
            "index": _preview_frame_sequence(image_path),
            "image_path": image_path,
            "json_path": json_path,
            "payload": payload,
        })
    return frames


def _clean_history_text(value, default: str = "-") -> str:
    text = str(value or "").strip()
    return text if text and text != "-" else default


def _strip_seen_prefix(text: str) -> str:
    value = _clean_history_text(text, "")
    for prefix in ("看到 ", "看到"):
        if value.startswith(prefix):
            return value[len(prefix):].strip(" ：:")
    return value


def _is_routine_history_info_key(key) -> bool:
    """Return whether an info key is routine telemetry rather than a scene event."""
    value = str(key or "").strip().lower()
    if "__" in value:
        value = value.rsplit("__", 1)[-1]
    return value in HISTORY_ROUTINE_INFO_KEYS


def _is_routine_recognition_history_log(text: str) -> bool:
    """Suppress bare `看到 direction/location` lines from the decision log."""
    value = _clean_history_text(text, "")
    if not value:
        return False
    normalized = value.lower().replace("，", ",").replace("、", ",")
    normalized = re.sub(r"^\[[^\]]+\]\s*", "", normalized).strip(" 。.!！")
    match = re.fullmatch(r"(?:看到|看到了|识别到|检测到|发现)\s*(.+)", normalized)
    names_text = match.group(1) if match else normalized
    names = [part.strip() for part in names_text.split(",") if part.strip()]
    return bool(names) and all(_is_routine_history_info_key(name) for name in names)


def _compact_history_logic_logs(frame_logs: list) -> list[str]:
    """Keep the history view readable without deleting the raw frame log payload."""
    unique_logs = []
    seen_logs = set()
    for item in frame_logs if isinstance(frame_logs, list) else []:
        text = _clean_history_text(item, "")
        if not text or text in seen_logs or _is_routine_recognition_history_log(text):
            continue
        unique_logs.append(text)
        seen_logs.add(text)

    # 阻塞式转向每一步都是连续控制的关键证据；保留全过程，便于核对
    # current_direction 到 target_direction 的实际收敛过程。
    if any("视角调整" in text and "current_direction=" in text for text in unique_logs):
        return [f"- {text}" for text in unique_logs]

    if len(unique_logs) <= HISTORY_MAX_VISIBLE_LOG_LINES:
        return [f"- {text}" for text in unique_logs]

    key_indexes = [
        index
        for index, text in enumerate(unique_logs)
        if any(marker in text for marker in HISTORY_KEY_LOG_MARKERS)
    ]
    selected_indexes = key_indexes[:HISTORY_MAX_VISIBLE_LOG_LINES]
    for index in range(len(unique_logs) - 1, -1, -1):
        if len(selected_indexes) >= HISTORY_MAX_VISIBLE_LOG_LINES:
            break
        if index not in selected_indexes:
            selected_indexes.append(index)
    selected_indexes.sort()

    lines = [f"- {unique_logs[index]}" for index in selected_indexes]
    hidden_count = len(unique_logs) - len(selected_indexes)
    lines.append(f"- 其余 {hidden_count} 条普通日志已折叠（原始记录仍保留在该帧 JSON）。")
    return lines


def _extract_seen_text(semantic_perception: dict, seen: dict, info_payload: dict) -> str:
    for source in (semantic_perception, seen):
        if isinstance(source, dict):
            info_keys = source.get("info_keys")
            if isinstance(info_keys, list) and info_keys:
                meaningful_keys = [
                    str(key) for key in info_keys
                    if not _is_routine_history_info_key(key)
                ]
                if meaningful_keys:
                    return ", ".join(meaningful_keys)
            summary = _strip_seen_prefix(str(source.get("summary") or ""))
            if summary and not _is_routine_recognition_history_log(summary):
                return summary
    if isinstance(info_payload, dict):
        active_keys = []
        for key, value in info_payload.items():
            text = str(value).strip()
            if (
                text
                and text not in {"False", "None", "[]", "{}"}
                and not _is_routine_history_info_key(key)
            ):
                active_keys.append(str(key))
        if active_keys:
            return ", ".join(active_keys)
    return "-"


def _format_history_info(info_payload: dict) -> list[str]:
    if not isinstance(info_payload, dict) or not info_payload:
        return ["- info: -"]
    lines = ["- info:"]
    for key, value in list(info_payload.items())[:80]:
        lines.append(f"  {key}: {value}")
    return lines


def _format_history_logic(
    *,
    seen_text: str,
    stage_name: str,
    frame_log: str,
    frame_logs: list,
    semantic_judgment: dict,
    semantic_branch: dict,
    decision_payload: dict,
    code_branch: dict,
    next_action: str,
) -> list[str]:
    if isinstance(frame_logs, list):
        lines = _compact_history_logic_logs(frame_logs)
        if lines:
            return lines

    plain_log = _clean_history_text(frame_log, "")
    if plain_log:
        return [f"- {plain_log}"]

    branch_name = _clean_history_text(
        semantic_branch.get("name")
        or code_branch.get("target")
        or decision_payload.get("target")
        or stage_name,
        "",
    )
    observation = _clean_history_text(
        semantic_judgment.get("reason")
        or decision_payload.get("observation")
        or code_branch.get("observation"),
        "",
    )
    action = _clean_history_text(
        semantic_judgment.get("decision")
        or decision_payload.get("decision")
        or decision_payload.get("action")
        or code_branch.get("action")
        or next_action,
        "",
    )
    method = _clean_history_text(
        semantic_judgment.get("evidence")
        or decision_payload.get("method")
        or code_branch.get("method"),
        "",
    )
    result = _clean_history_text(
        semantic_judgment.get("result_expectation")
        or decision_payload.get("result")
        or code_branch.get("result"),
        "",
    )

    lines = []
    if observation and observation != seen_text:
        lines.append(f"- 判断: {observation}")
    if action:
        lines.append(f"- 决策: {action}")
    if result and any(marker in result for marker in HISTORY_KEY_LOG_MARKERS):
        lines.append(f"- 结果: {result}")
    if lines:
        return lines[:HISTORY_MAX_VISIBLE_LOG_LINES]
    if branch_name and branch_name != _clean_history_text(stage_name, ""):
        return [f"- 当前逻辑: {branch_name}"]
    return ["- 本帧暂无新的业务决策日志。"]


def _strip_control_history_log_prefix(text: str) -> str:
    value = _clean_history_text(text, "")
    for prefix in ("控制信息：", "控制信息:", "控制信息"):
        if value.startswith(prefix):
            return value[len(prefix):].strip(" ：:")
    return ""


def _split_history_control_logs(frame_log: str, frame_logs: list) -> tuple[str, list[str], list[str]]:
    logic_logs = []
    control_logs = []

    if isinstance(frame_logs, list) and frame_logs:
        for item in frame_logs:
            text = _clean_history_text(item, "")
            if not text:
                continue
            control_text = _strip_control_history_log_prefix(text)
            if control_text:
                control_logs.append(control_text)
            else:
                logic_logs.append(text)
        return "", logic_logs, control_logs

    plain_log = _clean_history_text(frame_log, "")
    if not plain_log:
        return "", [], []
    control_text = _strip_control_history_log_prefix(plain_log)
    if control_text:
        return "", [], [control_text]
    return plain_log, [], []


def _split_history_typed_logs(frame_log_entries: list) -> tuple[list[str], list[str]]:
    logic_logs = []
    control_logs = []
    if not isinstance(frame_log_entries, list):
        return logic_logs, control_logs

    for entry in frame_log_entries:
        if not isinstance(entry, dict):
            continue
        message = _clean_history_text(entry.get("message"), "")
        if not message:
            continue
        if entry.get("category") == LOG_CATEGORY_UI:
            control_logs.append(message)
        else:
            logic_logs.append(message)
    return logic_logs, control_logs


def _format_control_frame_logs(control_logs: list[str]) -> list[str]:
    lines = []
    for item in control_logs:
        text = _clean_history_text(item, "")
        if text:
            lines.append(f"- {text}")
    return lines


def _format_history_control(control_logs: list[str], semantic_actions: list[dict]) -> list[str]:
    control_lines = _format_control_frame_logs(control_logs)
    if control_lines:
        return control_lines
    return _format_semantic_actions(semantic_actions)


def _numeric_param(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _format_action_params(params: dict, duration: str) -> list[str]:
    if not isinstance(params, dict):
        params = {}
    preferred_order = [
        "x_bias",
        "y_bias",
        "dura",
        "wait",
        "duration_ms",
        "duration",
        "finger_id",
        "backend",
    ]
    parts = []
    used = set()
    for key in preferred_order:
        value = _clean_history_text(params.get(key), "")
        if value:
            parts.append(f"{key}={value}")
            used.add(key)
    for key, value in params.items():
        if key in used:
            continue
        text = _clean_history_text(value, "")
        if text:
            parts.append(f"{key}={text}")
    duration_text = _clean_history_text(duration, "")
    if duration_text and not any(item.startswith(("dura=", "duration=", "duration_ms=")) for item in parts):
        parts.append(f"duration={duration_text}")
    return parts


def _history_action_verb(raw_name: str, action_name: str, params: dict, duration: str, start_pos: str, end_pos: str) -> str:
    duration_ms = (
        _numeric_param(params.get("duration_ms") if isinstance(params, dict) else None)
        or _numeric_param(params.get("duration") if isinstance(params, dict) else None)
        or _numeric_param(duration)
    )
    if raw_name == "click" and duration_ms is not None and duration_ms >= 800:
        return "长按了"
    if raw_name == "click" or action_name == "click":
        return "点击了"
    swipe_actions = {"tap_single", "tap_double", "uinput_tap_single", "move_press", "move_to", "move_up"}
    semantic_swipes = {"move_forward", "move_backward", "move_lateral", "turn_view", "move_control"}
    has_bias = isinstance(params, dict) and any(_clean_history_text(params.get(key), "") for key in ("x_bias", "y_bias"))
    if raw_name in swipe_actions and (has_bias or start_pos or end_pos or action_name in semantic_swipes):
        return "滑动了"
    return "执行了"


def _format_semantic_actions(actions: list[dict]) -> list[str]:
    lines = []
    if not isinstance(actions, list) or not actions:
        return ["- 暂无控制动作"]
    for action in actions[-12:]:
        if not isinstance(action, dict):
            continue
        raw_name = _clean_history_text(action.get("name"), "")
        action_name = _clean_history_text(action.get("action"), raw_name)
        target = _clean_history_text(action.get("target"), "")
        control_point = _clean_history_text(action.get("control_point"), target)
        target_text = control_point or target or action_name or raw_name or "控制点"
        actual_pos = _clean_history_text(action.get("actual_pos"), "")
        start_pos = _clean_history_text(action.get("start_pos"), "")
        end_pos = _clean_history_text(action.get("end_pos"), "")
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        duration = _clean_history_text(action.get("duration") or params.get("dura") or params.get("duration"), "")
        reason = _clean_history_text(action.get("reason"), "")
        verb = _history_action_verb(raw_name, action_name, params, duration, start_pos, end_pos)
        detail_parts = []
        if actual_pos:
            detail_parts.append(f"actual_pos={actual_pos}")
        if start_pos:
            detail_parts.append(f"start={start_pos}")
        if end_pos:
            detail_parts.append(f"end={end_pos}")
        detail_parts.extend(_format_action_params(params, duration))
        if reason:
            if detail_parts:
                lines.append(f"- {verb}{target_text}: {', '.join(detail_parts)}; reason={reason}")
            else:
                lines.append(f"- {verb}{target_text}: reason={reason}")
        elif detail_parts:
            lines.append(f"- {verb}{target_text}: {', '.join(detail_parts)}")
        else:
            lines.append(f"- {verb}{target_text}")
    return lines or ["- 暂无控制动作"]


def format_history_frame_details(frame_record: dict) -> str:
    payload = frame_record.get("payload") if isinstance(frame_record, dict) else {}
    payload = payload if isinstance(payload, dict) else {}
    frame_info = payload.get("frame") if isinstance(payload.get("frame"), dict) else {}
    stage_info = payload.get("stage")
    if isinstance(stage_info, dict):
        stage_name = stage_info.get("name") or "-"
        stage_group = stage_info.get("group") or "-"
    else:
        stage_name = payload.get("stage") or "-"
        stage_group = "-"

    seen = payload.get("seen") if isinstance(payload.get("seen"), dict) else {}
    info_payload = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    semantic_log = payload.get("semantic_log") if isinstance(payload.get("semantic_log"), dict) else {}
    decision_payload = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    code_branch = payload.get("code_branch") if isinstance(payload.get("code_branch"), dict) else {}
    next_action = str(payload.get("next_action") or "").strip() or "-"
    seen_summary = _extract_seen_text(
        semantic_log.get("perception") if isinstance(semantic_log.get("perception"), dict) else {},
        seen,
        info_payload,
    )

    lines = [
        f"帧: {frame_info.get('image') or Path(str(frame_record.get('image_path') or '')).name}",
        f"序号: {frame_info.get('index', frame_record.get('index', '-'))}",
    ]

    semantic_stage = semantic_log.get("current_stage") if isinstance(semantic_log.get("current_stage"), dict) else {}
    semantic_judgment = semantic_log.get("judgment") if isinstance(semantic_log.get("judgment"), dict) else {}
    semantic_branch = semantic_log.get("branch") if isinstance(semantic_log.get("branch"), dict) else {}
    semantic_actions = semantic_log.get("actions") if isinstance(semantic_log.get("actions"), list) else []
    frame_log = (
        decision_payload.get("frame_log")
        or semantic_log.get("frame_log")
        or payload.get("frame_log")
        or ""
    )
    frame_logs = (
        decision_payload.get("frame_logs")
        or semantic_log.get("frame_logs")
        or payload.get("frame_logs")
        or []
    )
    frame_log_entries = (
        decision_payload.get("frame_log_entries")
        or semantic_log.get("frame_log_entries")
        or payload.get("frame_log_entries")
        or []
    )
    typed_logic_logs, typed_control_logs = _split_history_typed_logs(
        frame_log_entries
    )
    if typed_logic_logs or typed_control_logs:
        logic_frame_log = ""
        logic_frame_logs = typed_logic_logs
        control_frame_logs = typed_control_logs
    else:
        logic_frame_log, logic_frame_logs, control_frame_logs = _split_history_control_logs(
            frame_log,
            frame_logs,
        )

    lines.extend([
        "",
        "日志信息",
        *_format_history_logic(
            seen_text=seen_summary,
            stage_name=stage_name,
            frame_log=logic_frame_log,
            frame_logs=logic_frame_logs,
            semantic_judgment=semantic_judgment,
            semantic_branch=semantic_branch,
            decision_payload=decision_payload,
            code_branch=code_branch,
            next_action=next_action,
        ),
        "",
        "控制信息",
        *_format_history_control(control_frame_logs, semantic_actions),
        "",
        "当前阶段",
        f"- stage: {semantic_stage.get('stage') or stage_name}",
        f"- group: {semantic_stage.get('group') or stage_group}",
        "",
        "识别信息",
        *_format_history_info(info_payload),
    ])

    return "\n".join(lines)


def discover_project_cases() -> list[str]:
    LOGGER.debug("discover_project_cases: dir=%s exists=%s", CUSTOMS_EXAMPLES_DIR, CUSTOMS_EXAMPLES_DIR.exists())
    if not CUSTOMS_EXAMPLES_DIR.exists():
        return []

    cases = []
    for path in sorted(CUSTOMS_EXAMPLES_DIR.iterdir()):
        if path.is_dir() and (path / "info.py").exists():
            cases.append(path.name)
    return cases


def discover_target_cases(project_case: str) -> list[str]:
    project_dir = CUSTOMS_GAME_EXAMPLES_DIR / project_case
    LOGGER.debug("discover_target_cases: project_case=%s dir=%s exists=%s", project_case, project_dir, project_dir.exists())
    if not project_dir.exists():
        return []

    cases = []
    for path in sorted(project_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        cases.append(path.stem)
    return cases


def _xdevice_summary_exit_code(summary_path: Path) -> int:
    """Translate an xDevice summary report into a process exit code."""
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        LOGGER.error("xDevice summary report is missing: %s", summary_path)
        return 1

    try:
        summary = ElementTree.parse(summary_path).getroot()
        counts = {
            name: int(summary.get(name, "0") or 0)
            for name in ("tests", "failures", "errors", "disabled", "unavailable")
        }
    except (OSError, ValueError, ElementTree.ParseError):
        LOGGER.exception("failed to read xDevice summary report: %s", summary_path)
        return 1

    LOGGER.info("xDevice summary result: path=%s counts=%s", summary_path, counts)
    if counts["tests"] <= 0:
        return 1
    if any(counts[name] > 0 for name in ("failures", "errors", "disabled", "unavailable")):
        return 1
    return 0


def run_testcase_entry(testcase_label: str) -> int:
    install_hidden_subprocess_patch()
    start_hidden_subprocess_window_suppressor()
    LOGGER.info("run_testcase_entry: testcase_label=%s", testcase_label)
    from xdevice.__main__ import main_process

    with hidden_subprocess_context(
        target_executables=("icpm_xdc.exe", "hdc.exe", "hdc"),
        hide_all=True,
    ):
        main_process(f"run -l {testcase_label}")

    from xdevice import Variables

    report_dir = (
        Path(Variables.exec_dir)
        / Variables.report_vars.report_dir
        / Variables.task_name
    )
    return _xdevice_summary_exit_code(report_dir / "summary_report.xml")


def run_direct_entry(project_case: str, target_case: str):
    install_hidden_subprocess_patch()
    start_hidden_subprocess_window_suppressor()
    LOGGER.info(
        "run_direct_entry: project_case=%s target_case=%s",
        project_case,
        target_case,
    )
    os.environ["TARGET_PROJECT_CASE"] = project_case
    os.environ["TARGET_GAME_CASE"] = target_case

    from aw.autogame.tools.GameAutomator import GameAutomator

    automator = GameAutomator(driver=None, logger=None)
    automator.start()


def run_game_recording_entry() -> int:
    """由 Launcher 子进程启动独立的录制回放 Qt 窗口。"""
    from aw.autogame.customs_game_examples.Game_Recording.main import main as game_recording_main

    # ``--run-game-recording`` 是 Launcher 自己的 helper 参数，不能泄漏给
    # Game Recording 的 argparse；后者只应接收其自身的可选启动参数。
    return int(game_recording_main(argv=[]))


def run_hdc_shell(command: str) -> Optional[str]:
    hdc_executable = resolve_hdc_executable()
    cmd = [hdc_executable, "shell", command]
    LOGGER.debug("run_hdc_shell: cmd=%s", cmd)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=HDC_SHELL_TIMEOUT_SECONDS,
            **hidden_subprocess_kwargs(),
        )
    except FileNotFoundError:
        LOGGER.warning(
            "hdc executable not found. Please ensure `hdc` is installed and available in PATH. attempted_cmd=%s",
            cmd,
        )
        return None
    except subprocess.TimeoutExpired:
        LOGGER.warning("hdc shell timeout: %s", cmd)
        return None
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        LOGGER.warning(
            "hdc shell failed: %s | stdout=%s | stderr=%s",
            cmd,
            stdout,
            stderr,
        )
        return None
    output = result.stdout.strip()
    LOGGER.debug("run_hdc_shell success: command=%s output=%s", cmd, output)
    return output


_POWER_SUPPLY_BATTERY_PATHS = (
    "/sys/class/power_supply/Battery",
    "/sys/class/power_supply/battery",
)


def _extract_sysfs_number(raw: str, key: Optional[str] = None) -> Optional[float]:
    text = (raw or "").strip()
    if not text:
        return None

    if key:
        match = re.search(rf"(?:^|\n)\s*{re.escape(key)}\s*=\s*([+-]?\d+(?:\.\d+)?)", text)
        if match:
            return float(match.group(1))

    match = re.search(r"=\s*([+-]?\d+(?:\.\d+)?)\s*(?:\r?\n|$)", text)
    if match:
        return float(match.group(1))

    match = re.search(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:\r?\n|$)", text)
    if match:
        return float(match.group(1))
    return None


def _parse_battery_temperature_c(raw: str) -> Optional[float]:
    value = _extract_sysfs_number(raw, "POWER_SUPPLY_TEMP")
    if value is None:
        return None

    if abs(value) > 1000:
        temperature = value / 1000.0
    elif abs(value) > 100:
        temperature = value / 10.0
    else:
        temperature = value

    if -40.0 <= temperature <= 125.0:
        return temperature
    return None


def _parse_battery_capacity(raw: str) -> Optional[int]:
    value = _extract_sysfs_number(raw, "POWER_SUPPLY_CAPACITY")
    if value is None:
        return None
    capacity = int(float(value))
    if 0 <= capacity <= 100:
        return capacity
    return None


def _read_device_value_with_fallback(label: str, commands: list[str], parser):
    for command in commands:
        raw = run_hdc_shell(command)
        if not raw:
            LOGGER.debug("%s read empty: command=%s", label, command)
            continue
        value = parser(raw)
        if value is not None:
            LOGGER.info("%s read success: command=%s value=%s raw=%s", label, command, value, raw[:200])
            return value
        LOGGER.warning("%s parse failed: command=%s raw=%s", label, command, raw[:200])

    LOGGER.warning("%s unavailable after fallback commands: %s", label, commands)
    return None


def build_restart_device_commands(hdc_executable: Optional[str] = None):
    hdc = hdc_executable or resolve_hdc_executable()
    return [
        [hdc, "shell", "reboot", "-D"],
        [hdc, "wait"],
    ]


def launch_restart_bat_with_system_shell(script_path: Path) -> None:
    LOGGER.info("restart.bat system shell start: script_path=%s", script_path)
    if not hasattr(os, "startfile"):
        raise RuntimeError("os.startfile is only available on Windows")
    os.startfile(script_path)  # type: ignore[attr-defined]


def build_restart_bat_cmd_command(script_path: Path) -> list[str]:
    if os.name == "nt":
        return ["cmd", "/c", f'title {RESTART_BAT_CMD_TITLE} && call "{script_path.name}"']
    return [str(script_path)]


def restart_bat_cmd_window_kwargs() -> Dict[str, int]:
    if os.name != "nt":
        return {}
    create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    return {"creationflags": create_new_console} if create_new_console else {}


def launch_restart_bat_cmd_window(script_path: Path) -> subprocess.Popen:
    command = build_restart_bat_cmd_command(script_path)
    LOGGER.info(
        "restart.bat visible cmd start: command=%s cwd=%s",
        command,
        script_path.parent,
    )
    return subprocess.Popen(
        command,
        cwd=str(script_path.parent),
        **restart_bat_cmd_window_kwargs(),
    )


def install_helper_signal_handlers():
    def _request_graceful_exit(signum, _frame):
        LOGGER.warning("helper process received signal=%s, exiting gracefully", signum)
        raise SystemExit(128 + int(signum))

    for sig in (
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGBREAK", None),
    ):
        if sig is None:
            continue
        try:
            signal.signal(sig, _request_graceful_exit)
        except Exception:
            LOGGER.debug("install signal handler failed: sig=%s", sig, exc_info=True)


def get_battery_temperature_c() -> Optional[float]:
    commands = [f"cat {path}/temp" for path in _POWER_SUPPLY_BATTERY_PATHS]
    commands.extend(f"cat {path}/uevent" for path in _POWER_SUPPLY_BATTERY_PATHS)
    commands.extend(
        [
            'for p in /sys/class/power_supply/*; do [ -f "$p/temp" ] && printf "%s=%s\\n" "$p" "$(cat "$p/temp")"; done; true',
            "for p in /sys/class/power_supply/*; do [ -f \"$p/uevent\" ] && grep -E 'POWER_SUPPLY_TEMP=' \"$p/uevent\"; done; true",
        ]
    )
    return _read_device_value_with_fallback("battery_temperature", commands, _parse_battery_temperature_c)


def get_battery_capacity() -> Optional[int]:
    commands = [f"cat {path}/capacity" for path in _POWER_SUPPLY_BATTERY_PATHS]
    commands.extend(f"cat {path}/uevent" for path in _POWER_SUPPLY_BATTERY_PATHS)
    commands.extend(
        [
            'for p in /sys/class/power_supply/*; do [ -f "$p/capacity" ] && printf "%s=%s\\n" "$p" "$(cat "$p/capacity")"; done; true',
            "for p in /sys/class/power_supply/*; do [ -f \"$p/uevent\" ] && grep -E 'POWER_SUPPLY_CAPACITY=' \"$p/uevent\"; done; true",
        ]
    )
    return _read_device_value_with_fallback("battery_capacity", commands, _parse_battery_capacity)


def set_hiz_mode(active: bool):
    if active:
        run_hdc_shell("echo 1 > /sys/class/hw_power/charger/charge_data/enable_hiz")
        run_hdc_shell("echo stopsink > /sys/class/hw_power/charger/charge_data/plugusb")
    else:
        run_hdc_shell("echo 0 > /sys/class/hw_power/charger/charge_data/enable_hiz")
        run_hdc_shell("echo startsink > /sys/class/hw_power/charger/charge_data/plugusb")


def force_stop_apps(apps: list[str]) -> list[str]:
    stopped = []
    for app in apps:
        if not app:
            continue
        run_hdc_shell(f"aa force-stop {app}")
        stopped.append(app)
    return stopped


def _preview_scene_pool_lookup(scene_pool_info) -> dict[str, dict]:
    if not isinstance(scene_pool_info, dict):
        return {}
    pool_scenes = scene_pool_info.get("scenes", {})
    if not isinstance(pool_scenes, dict):
        return {}

    lookup = {}
    for scene_dir, scene_data in pool_scenes.items():
        if not isinstance(scene_data, dict):
            continue
        scene_dir_text = str(scene_dir or "").strip()
        if scene_dir_text:
            lookup[scene_dir_text] = scene_data
        scene_name = str(scene_data.get("name") or "").strip()
        if scene_name:
            lookup.setdefault(scene_name, scene_data)
    return lookup


def _is_preview_scene_data(value) -> bool:
    if not isinstance(value, dict):
        return False
    if "resolutions" in value:
        return True
    return any(key in value for key in ("areas", "points", "special_areas"))


def _preview_scene_reference_candidates(scene_name: str, scene_ref) -> list[str]:
    candidates = []
    if isinstance(scene_ref, str):
        candidates.append(scene_ref)
    elif isinstance(scene_ref, dict):
        for key in (
            "dir",
            "scene_dir",
            "scene_pool_dir",
            "scene_pool_ref",
            "ref",
            "scene",
            "name",
        ):
            value = str(scene_ref.get(key) or "").strip()
            if value:
                candidates.append(value)
    scene_name = str(scene_name or "").strip()
    if scene_name:
        candidates.append(scene_name)

    result = []
    seen = set()
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def resolve_preview_stage_scenes(stage_entry, scene_pool_info=None) -> dict[str, dict]:
    if not isinstance(stage_entry, dict):
        return {}

    raw_scenes = stage_entry.get("scenes", {})
    if isinstance(raw_scenes, dict):
        raw_entries = list(raw_scenes.items())
    elif isinstance(raw_scenes, list):
        raw_entries = []
        for item in raw_scenes:
            if isinstance(item, dict):
                name = (
                    item.get("name")
                    or item.get("scene")
                    or item.get("dir")
                    or item.get("scene_dir")
                )
                raw_entries.append((str(name or "").strip(), item))
            elif isinstance(item, str):
                raw_entries.append((item, item))
    else:
        return {}

    pool_lookup = _preview_scene_pool_lookup(scene_pool_info)
    resolved = {}
    for raw_name, raw_scene in raw_entries:
        scene_name = str(raw_name or "").strip()
        scene_data = raw_scene if _is_preview_scene_data(raw_scene) else None
        if scene_data is None:
            for candidate in _preview_scene_reference_candidates(scene_name, raw_scene):
                candidate_data = pool_lookup.get(candidate)
                if _is_preview_scene_data(candidate_data):
                    scene_data = candidate_data
                    break
        if not isinstance(scene_data, dict):
            continue

        display_name = scene_name
        if isinstance(raw_scene, str):
            display_name = str(scene_data.get("name") or scene_name or raw_scene).strip()
        elif isinstance(raw_scene, dict):
            display_name = str(raw_scene.get("name") or scene_data.get("name") or scene_name).strip()
        if not display_name:
            display_name = str(scene_data.get("name") or "场景").strip()
        resolved[display_name] = scene_data
    return resolved


def resolve_preview_stage_info_entries(
    stage_entry,
    scene_pool_info=None,
    screen_width: Optional[int] = None,
    screen_height: Optional[int] = None,
) -> list[dict[str, str]]:
    """Build the stable area rows shown while one preview stage is active."""
    entries = []
    seen_keys = set()
    scenes = resolve_preview_stage_scenes(stage_entry, scene_pool_info)
    item_types = (
        ("special_areas", "特殊区域"),
        ("areas", "区域"),
        ("points", "控点"),
    )

    selected_scenes = []
    for scene_name, raw_scene_data in scenes.items():
        if isinstance(raw_scene_data, dict):
            selected_scenes.append(
                (
                    scene_name,
                    select_scene_resolution(
                        raw_scene_data,
                        screen_width,
                        screen_height,
                    ),
                )
            )

    for item_type, type_label in item_types:
        for scene_name, scene_data in selected_scenes:
            items = scene_data.get(item_type, {})
            if not isinstance(items, dict):
                continue
            for item_name, item_data in items.items():
                item_name = str(item_name or "").strip()
                if not item_name:
                    continue
                info_key = f"{scene_name}__{item_name}"
                row_key = f"{item_type}::{info_key}"
                if row_key in seen_keys:
                    continue
                seen_keys.add(row_key)
                entries.append(
                    {
                        "key": row_key,
                        "info_key": info_key,
                        "item_type": item_type,
                        "type": type_label,
                        "name": f"{scene_name}/{item_name}",
                        "template": str(
                            item_data.get("template") or ""
                            if isinstance(item_data, dict)
                            else ""
                        ),
                    }
                )
    return entries


class LauncherWindow(QWidget):
    stream_verification_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        LOGGER.info("LauncherWindow init start")
        self.process: Optional[QProcess] = None
        self.selected_testcase_file: Optional[Path] = None
        self.selected_testcase_description = ""
        self._updating_targets = False
        self.latest_preview_file: Optional[Path] = None
        self.latest_preview_pixmap: Optional[QPixmap] = None
        self.latest_preview_payload: Optional[dict] = None
        self.stage_info_cache: dict[str, dict] = {}
        self.preview_info_cache: dict[str, dict] = {}
        self.preview_info_stage_name: Optional[str] = None
        self.preview_info_items: dict[str, QTreeWidgetItem] = {}
        self.preview_info_item_types: dict[str, str] = {}
        self.preview_info_result_keys: dict[str, str] = {}
        self.preview_info_template_paths: dict[str, str] = {}
        self.preview_info_template_pixmap: Optional[QPixmap] = None
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(150)
        self.stream_verify_timer = QTimer(self)
        self.stream_verify_timer.setInterval(80)
        self.safety_timer = QTimer(self)
        self.safety_timer.setInterval(5000)
        self.run_timeout_timer = QTimer(self)
        self.run_timeout_timer.setSingleShot(True)
        self.stream_disconnect_signal_timer = QTimer(self)
        self.stream_disconnect_signal_timer.setInterval(500)
        self.stream_verification_failed.connect(self._handle_stream_verification_failed)

        self.batch_active = False
        self.stop_requested = False
        self.manual_stop_requested = False
        self.recovery_processes: list[subprocess.Popen] = []
        self._close_after_stop = False
        self.current_run_index = 0
        self.total_runs = 1
        self.current_plan: Optional[dict] = None
        self.current_run_timed_out = False
        self.current_run_output_start = 0
        self.output_log_spool_path: Optional[Path] = None
        self.process_output_buffer = ""
        self.current_run_stream_started = False
        self.current_run_stream_disconnected = False
        self.current_run_stream_disconnect_startup = False
        self.current_run_stream_disconnect_message = ""
        self.current_run_stream_preserved = False
        self.current_run_sp_started = False
        self.current_run_sp_started_monotonic: Optional[float] = None
        self.current_run_sp_save_confirmed = False
        self.current_run_sp_state: dict = {}
        self.sp_save_settle_in_progress = False
        self.pending_process_finished: Optional[tuple[int, object]] = None
        self.current_run_failure_code = ""
        self.current_run_failure_reason = ""
        self.current_run_failure_details: dict = {}
        self.current_run_inactivity_preserved = False
        self.dismiss_reboot_prompt_on_next_case_start = False
        self.current_batch_start_timestamp: Optional[str] = None
        self.current_run_start_timestamp: Optional[str] = None
        self.current_run_archive_dir: Optional[Path] = None
        self.current_hdc_debug_capture: Optional[HdcDebugRunCapture] = None
        self.current_hilog_capture: Optional[HilogRunCapture] = None
        self.current_memory_capture: Optional[MemoryRunCapture] = None
        # 真正启动任务时再校验环境变量，避免无效配置让 Launcher 初始化阶段崩溃。
        self.hdc_debug_level = 5
        self.preview_target_info_height = 64
        self.preview_target_info_width = 460
        self._adjusting_preview_splitter = False
        self.preview_render_screen_size: Optional[tuple[int, int]] = None
        self.stream_verify_active = False
        self.stream_verify_first_frame_seen = False
        self.stream_verify_screen_mode = ""
        self.stream_verify_client = None
        self.stream_verify_buffer = None
        self.stream_verify_thread: Optional[threading.Thread] = None
        self.stream_verify_failure_reported = False
        self.stream_verify_lock = threading.Lock()
        self.preset_buttons: list[QPushButton] = []
        self.theme_mode = "light"
        self.inputs_enabled = True
        self.label_tool = None
        self.label_tool_project_dir: Optional[Path] = None
        self.label_tool_return_page = None
        self.game_recording_window = None
        self.game_recording_run_dir: Optional[Path] = None
        self.game_recording_started_at = None
        self.game_recording_hdc_capture = None
        self.game_recording_hilog_capture = None
        self.game_replay_panel = None
        self.game_replay_window = None
        self.history_records: list[dict] = []
        self.selected_history_record: Optional[dict] = None
        self.selected_history_batch_dir: Optional[Path] = None
        self.history_frame_records: list[dict] = []
        self.history_frame_index = -1
        self.process_launch_tracer = WindowsProcessLaunchTracer()

        self.setWindowTitle("Auto Game 启动器")
        self.resize(1260, 860)
        self.setMinimumSize(1120, 820)
        self._apply_style()

        self.mode_testcase = QRadioButton("通过 testcases 用例启动")
        self.mode_direct = QRadioButton("直接指定 project_case / target_case")
        self.mode_testcase.setChecked(True)

        self.testcase_path_edit = QLineEdit()
        self.testcase_path_edit.setReadOnly(True)
        self.testcase_path_edit.setPlaceholderText("未选择 testcases 用例文件")

        self.browse_button = QPushButton("选择用例")
        self.clear_button = QPushButton("重选")
        self.open_label_tool_button = QPushButton("打开标注工具")
        self.open_game_recording_button = QPushButton("录制")
        self.open_game_recording_button.setFixedWidth(72)
        self.open_game_recording_button.setToolTip(
            "打开当前 Game_Recording 录制界面"
        )
        self.open_game_replay_button = QPushButton("回放")
        self.open_game_replay_button.setFixedWidth(72)
        self.open_game_replay_button.setToolTip("浏览历史录制并开始回放")
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.setToolTip("刷新配置")
        self.refresh_button.setFixedWidth(64)

        self.project_combo = QComboBox()
        self.target_combo = QComboBox()

        self.case_info_label = QLabel("用例信息：未选择用例")
        self.case_info_label.setObjectName("caseInfoLabel")
        self.case_info_label.setWordWrap(True)
        self.case_info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # 保留旧属性名，避免外部扩展直接访问时中断。
        self.status_label = self.case_info_label
        self.runtime_label = QLabel("运行信息：未开始")
        self.runtime_label.setObjectName("runtimeLabel")
        self.runtime_label.setWordWrap(True)
        self.runtime_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.run_count_spin = QSpinBox()
        self.run_count_spin.setRange(1, 9999)
        self.run_count_spin.setValue(1)
        self.run_count_field = self._create_spin_with_presets(
            self.run_count_spin,
            [1, 3, 5, 8, 10],
        )

        self.test_profile_field = QWidget()
        self.test_profile_layout = QHBoxLayout(self.test_profile_field)
        self.test_profile_layout.setContentsMargins(0, 0, 0, 0)
        self.test_profile_layout.setSpacing(4)
        self.power_test_radio = QRadioButton("功耗\n测试")
        self.function_test_radio = QRadioButton("功能\n测试")
        self.marathon_test_radio = QRadioButton("马拉松\n测试")
        self.power_test_radio.setChecked(True)
        self.test_profile_button_group = QButtonGroup(self)
        self.test_profile_button_group.setExclusive(True)
        self.test_profile_button_group.addButton(self.power_test_radio)
        self.test_profile_button_group.addButton(self.function_test_radio)
        self.test_profile_button_group.addButton(self.marathon_test_radio)
        self.test_profile_layout.addWidget(self.power_test_radio)
        self.test_profile_layout.addWidget(self.function_test_radio)
        self.test_profile_layout.addWidget(self.marathon_test_radio)
        self.test_profile_layout.addStretch(1)

        self.case_loop_count_spin = QSpinBox()
        self.case_loop_count_spin.setRange(1, 999)
        self.case_loop_count_spin.setValue(1)
        self.case_loop_count_field = self._create_spin_with_presets(
            self.case_loop_count_spin,
            [1, 2, 3, 5],
        )

        self.safe_temp_spin = QDoubleSpinBox()
        self.safe_temp_spin.setRange(0.0, 100.0)
        self.safe_temp_spin.setDecimals(1)
        self.safe_temp_spin.setSingleStep(0.5)
        self.safe_temp_spin.setValue(36.0)
        self.safe_temp_spin.setSuffix(" °C")
        self.safe_temp_field = self._create_spin_with_presets(
            self.safe_temp_spin,
            [35, 38, 40, 42, 45],
            suffix="°C",
        )

        self.safe_battery_spin = QSpinBox()
        self.safe_battery_spin.setRange(0, 100)
        self.safe_battery_spin.setValue(60)
        self.safe_battery_spin.setSuffix(" %")
        self.safe_battery_field = self._create_spin_with_presets(
            self.safe_battery_spin,
            [10, 20, 30, 40, 50, 60, 70, 80],
            suffix="%",
        )

        self.safe_time_spin = QDoubleSpinBox()
        self.safe_time_spin.setRange(0.0, 10000.0)
        self.safe_time_spin.setDecimals(1)
        self.safe_time_spin.setSingleStep(1.0)
        self.safe_time_spin.setValue(0.0)
        self.safe_time_spin.setSuffix(" 分钟")
        self.safe_time_field = self._create_spin_with_presets(
            self.safe_time_spin,
            [0, 10, 20, 30, 45, 60],
            suffix="分",
        )

        self.inactivity_timeout_spin = QDoubleSpinBox()
        self.inactivity_timeout_spin.setRange(0.0, 10000.0)
        self.inactivity_timeout_spin.setDecimals(1)
        self.inactivity_timeout_spin.setSingleStep(1.0)
        self.inactivity_timeout_spin.setValue(5.0)
        self.inactivity_timeout_spin.setSuffix(" 分钟")
        self.inactivity_timeout_field = self._create_spin_with_presets(
            self.inactivity_timeout_spin,
            [1, 3, 5, 8, 10],
            suffix="分",
        )

        self.power_collection_duration_spin = QDoubleSpinBox()
        self.power_collection_duration_spin.setRange(0.0, 10000.0)
        self.power_collection_duration_spin.setDecimals(1)
        self.power_collection_duration_spin.setSingleStep(1.0)
        self.power_collection_duration_spin.setValue(0.0)
        self.power_collection_duration_spin.setSuffix(" 秒")
        self.power_collection_duration_field = self._create_spin_with_presets(
            self.power_collection_duration_spin,
            [3, 5, 10, 15, 20],
            suffix="秒",
        )

        self.marathon_duration_spin = QDoubleSpinBox()
        self.marathon_duration_spin.setRange(0.0, 10000.0)
        self.marathon_duration_spin.setDecimals(1)
        self.marathon_duration_spin.setSingleStep(10.0)
        self.marathon_duration_spin.setValue(60.0)
        self.marathon_duration_spin.setSuffix(" 分钟")
        self.marathon_duration_spin.setToolTip(
            "马拉松每轮按 SP 有效录制时间运行；每满该时长保存 SP 并重启下一轮"
        )
        self.marathon_duration_field = self._create_spin_with_presets(
            self.marathon_duration_spin,
            [0, 30, 60, 120, 180],
            suffix="分",
        )

        self.marathon_end_battery_spin = QSpinBox()
        self.marathon_end_battery_spin.setRange(0, 100)
        self.marathon_end_battery_spin.setValue(5)
        self.marathon_end_battery_spin.setSuffix(" %")
        self.marathon_end_battery_spin.setToolTip(
            "马拉松中电量小于等于该值时长按 SP 保存，并停止整个自动化"
        )
        self.marathon_end_battery_field = self._create_spin_with_presets(
            self.marathon_end_battery_spin,
            [0, 5, 10, 15, 20],
            suffix="%",
        )

        self.start_button = QPushButton("启动")
        self.start_button.setProperty("primaryButton", True)
        self.stream_verify_button = QPushButton("验证流")
        self.stream_verify_button.setToolTip("按 config.json 的 screen_mode 启动对应抓流验证，并在预览区域显示实时画面")
        self.hos_frame_rate_label = QLabel("帧率")
        self.hos_frame_rate_combo = QComboBox()
        self.hos_frame_rate_combo.setToolTip("选择后立即写入 config.json 的 hoscrcpy_frame_rate，下次验证流时生效")
        self.hos_frame_rate_combo.setFixedWidth(76)
        # 与操作栏按钮的样式后高度保持一致，避免未抛光的默认 sizeHint 把下拉框撑高。
        self.hos_frame_rate_combo.setFixedHeight(32)
        self.hos_frame_rate_combo.setStyleSheet(
            "QComboBox { min-height: 0px; padding: 3px 8px; }"
        )
        for frame_rate in HOSCRCPY_FRAME_RATE_OPTIONS:
            self.hos_frame_rate_combo.addItem(str(frame_rate), frame_rate)
        try:
            configured_frame_rate = read_hoscrcpy_frame_rate_config()
        except Exception as exc:
            configured_frame_rate = DEFAULT_HOSCRCPY_FRAME_RATE
            LOGGER.warning(
                "invalid hoscrcpy_frame_rate in %s; reset to %s: %s",
                AUTOGAME_CONFIG_FILE,
                configured_frame_rate,
                exc,
            )
            write_hoscrcpy_frame_rate_config(configured_frame_rate)
        self.hos_frame_rate_combo.setCurrentIndex(
            self.hos_frame_rate_combo.findData(configured_frame_rate)
        )
        self.current_hos_frame_rate = configured_frame_rate
        self.stop_button = QPushButton("停止")
        self.stop_button.setProperty("dangerButton", True)
        self.stop_button.setEnabled(False)
        self.open_history_button = QPushButton("历史输出")
        self.game_process_policy_button = QPushButton("关闭进程")
        self.game_process_policy_button.setObjectName("gameProcessPolicyButton")
        self.game_process_policy_button.setCheckable(True)
        # 与“显示/隐藏标注”保持一致：选中态代表用户开启了非默认策略。
        # 默认不选中=关闭进程（原色），选中=保留进程（绿色）。
        self.game_process_policy_button.setChecked(False)
        self.game_process_policy_button.setProperty("toggleButton", True)
        self.game_process_policy_button.setToolTip(
            "功耗测试和功能测试可切换进程策略；马拉松每轮强制关闭游戏和 SP 进程"
        )
        self.generate_preview_video_button = QPushButton("生成视频：关")
        self.generate_preview_video_button.setObjectName("generatePreviewVideoButton")
        self.generate_preview_video_button.setCheckable(True)
        self.generate_preview_video_button.setChecked(False)
        self.generate_preview_video_button.setProperty("toggleButton", True)
        self.generate_preview_video_button.setToolTip(
            "关闭时仍保留运行帧图片/JSON、Launcher 日志和 hilog，"
            "但不生成预览视频"
        )
        self.preview_overlay_button = QPushButton("隐藏标注")
        self.preview_overlay_button.setCheckable(True)
        self.preview_overlay_button.setChecked(True)
        self.preview_overlay_button.setProperty("toggleButton", True)
        self.preview_points_button = QPushButton("隐藏控点")
        self.preview_points_button.setCheckable(True)
        self.preview_points_button.setChecked(True)
        self.preview_points_button.setProperty("toggleButton", True)
        self.preview_fullscreen_button = QPushButton("放大预览")
        self.preview_fullscreen_button.setToolTip("让实时预览覆盖整个启动器界面；按 Esc 可退出")

        self.theme_combo = QComboBox()
        self.theme_combo.setObjectName("themeCombo")
        self.theme_combo.addItem("亮白", "light")
        self.theme_combo.addItem("暗黑", "dark")
        self.theme_combo.setCurrentIndex(0)
        self.theme_combo.setFixedWidth(96)

        self.output_edit = QPlainTextEdit()
        self.output_edit.setObjectName("outputConsole")
        self.output_edit.setReadOnly(True)
        self.output_edit.setMinimumHeight(90)
        self.output_edit.setPlaceholderText("运行输出会显示在这里...")
        self.output_edit.document().setMaximumBlockCount(OUTPUT_CONSOLE_MAX_BLOCKS)
        self.output_log_filter = LOG_FILTER_ALL
        self.output_log_entries: list[tuple[str, str]] = []
        self.output_filter_button_group = QButtonGroup(self)
        self.output_filter_button_group.setExclusive(True)
        self.output_filter_buttons: dict[str, QPushButton] = {}
        for filter_name in LOG_FILTERS:
            button = QPushButton(filter_name)
            button.setObjectName("outputLogFilterButton")
            button.setCheckable(True)
            button.setProperty("toggleButton", True)
            button.setChecked(filter_name == self.output_log_filter)
            self.output_filter_button_group.addButton(button)
            self.output_filter_buttons[filter_name] = button

        self.preview_image_label = QLabel("启动后将在这里实时显示可视化帧")
        self.preview_image_label.setObjectName("previewSurface")
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_label.setMinimumWidth(640)
        self.preview_image_label.setMinimumHeight(100)
        self.preview_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.preview_info_panel = QWidget()
        self.preview_info_panel.setObjectName("previewInfoPanel")
        preview_info_layout = QVBoxLayout(self.preview_info_panel)
        preview_info_layout.setContentsMargins(0, 0, 0, 0)
        preview_info_layout.setSpacing(6)
        preview_info_header = QWidget()
        preview_info_header_layout = QHBoxLayout(preview_info_header)
        preview_info_header_layout.setContentsMargins(0, 0, 0, 0)
        preview_info_header_layout.setSpacing(6)
        self.preview_info_stage_label = QLabel("当前阶段：等待运行")
        self.preview_info_stage_label.setObjectName("previewInfoStage")
        self.preview_info_select_all_button = QPushButton("全选")
        self.preview_info_select_none_button = QPushButton("全不选")
        self.preview_info_select_all_button.setFixedWidth(64)
        self.preview_info_select_none_button.setFixedWidth(64)
        preview_info_header_layout.addWidget(self.preview_info_stage_label, 1)
        preview_info_header_layout.addWidget(self.preview_info_select_all_button)
        preview_info_header_layout.addWidget(self.preview_info_select_none_button)
        self.preview_info_tree = QTreeWidget()
        self.preview_info_tree.setObjectName("previewInfo")
        self.preview_info_tree.setHeaderLabels(["类型", "场景/名称", "当前帧识别信息"])
        self.preview_info_tree.setRootIsDecorated(False)
        self.preview_info_tree.setAlternatingRowColors(True)
        self.preview_info_tree.setUniformRowHeights(True)
        self.preview_info_tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.preview_info_tree.setMinimumHeight(120)
        self.preview_info_tree.setMinimumWidth(320)
        self.preview_info_tree.setColumnWidth(0, 82)
        self.preview_info_tree.setColumnWidth(1, 150)
        self.preview_info_tree.header().setStretchLastSection(True)
        preview_template_group = QGroupBox("模板预览")
        preview_template_group.setObjectName("previewTemplateGroup")
        preview_template_layout = QVBoxLayout(preview_template_group)
        preview_template_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_info_template_label = QLabel("点击特殊区域或区域查看模板图片")
        self.preview_info_template_label.setObjectName("previewTemplate")
        self.preview_info_template_label.setFrameShape(QFrame.Shape.NoFrame)
        self.preview_info_template_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_info_template_label.setWordWrap(True)
        self.preview_info_template_label.setMinimumHeight(48)
        self.preview_info_template_label.setMaximumHeight(72)
        self.preview_info_template_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        preview_template_layout.addWidget(self.preview_info_template_label)
        preview_detail_group = QGroupBox("详细信息")
        preview_detail_group.setObjectName("previewDetailGroup")
        preview_template_group.setMaximumWidth(220)
        preview_detail_layout = QVBoxLayout(preview_detail_group)
        preview_detail_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_info_detail_edit = QPlainTextEdit()
        self.preview_info_detail_edit.setObjectName("previewInfoDetail")
        self.preview_info_detail_edit.setFrameShape(QFrame.Shape.NoFrame)
        self.preview_info_detail_edit.setReadOnly(True)
        self.preview_info_detail_edit.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        self.preview_info_detail_edit.setWordWrapMode(
            QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
        )
        self.preview_info_detail_edit.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.preview_info_detail_edit.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.preview_info_detail_edit.setMinimumHeight(64)
        self.preview_info_detail_edit.setMaximumHeight(100)
        self.preview_info_detail_edit.setPlainText("选中区域、特殊区域或控点后查看当前帧完整信息")
        preview_detail_layout.addWidget(self.preview_info_detail_edit)
        preview_info_layout.addWidget(preview_info_header, 0)
        preview_info_layout.addWidget(self.preview_info_tree, 1)
        preview_auxiliary_layout = QHBoxLayout()
        preview_auxiliary_layout.setContentsMargins(0, 0, 0, 0)
        preview_auxiliary_layout.setSpacing(6)
        preview_auxiliary_layout.addWidget(preview_detail_group, 2)
        preview_auxiliary_layout.addWidget(preview_template_group, 1)
        preview_info_layout.addLayout(preview_auxiliary_layout, 0)

        self._build_ui()
        self._bind_signals()
        self._load_project_cases()
        self._sync_mode_ui()
        self._sync_test_profile_ui()
        self._sync_game_process_policy_ui()
        self._log_message(
            "[Launcher] 启动器已初始化，日志将保存到每次运行目录。\n",
            level=logging.INFO,
        )
        LOGGER.info("LauncherWindow init finished")

    def _apply_style(self):
        if getattr(self, "theme_mode", "light") != "dark":
            self.setStyleSheet(
                """
                QWidget {
                    background: #eef3f8;
                    color: #18212f;
                    font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI", sans-serif;
                    font-size: 13px;
                }
                QWidget#headerBar {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #ffffff, stop:0.55 #f4f9ff, stop:1 #e8f2ff);
                    border: 1px solid #d6e3f0;
                    border-radius: 10px;
                }
                QWidget#actionBar {
                    background: #ffffff;
                    border: 1px solid #d9e5f1;
                    border-radius: 8px;
                }
                QWidget#statusStrip {
                    background: #ffffff;
                    border: 1px solid #d9e5f1;
                    border-radius: 8px;
                }
                QLabel#launcherTitle {
                    color: #101828;
                    font-size: 20px;
                    font-weight: 700;
                }
                QLabel#launcherSubtitle {
                    color: #64748b;
                    font-size: 12px;
                }
                QLabel#headerStatusPill,
                QLabel#headerRuntimePill {
                    background: #f7fbff;
                    border: 1px solid #cbddec;
                    border-radius: 13px;
                    color: #334155;
                    padding: 5px 12px;
                    font-weight: 600;
                }
                QLabel#headerStatusPill {
                    color: #087f5b;
                    border-color: #9ce3c8;
                    background: #eafff7;
                }
                QLabel#formLabel {
                    color: #475569;
                    font-weight: 600;
                }
                QWidget#configPanel,
                QWidget#configItem {
                    background: transparent;
                }
                QWidget#configSection {
                    background: #ffffff;
                    border: 1px solid #d9e5f1;
                    border-radius: 8px;
                }
                QLabel#configSectionTitle {
                    color: #2563eb;
                    font-size: 12px;
                    font-weight: 700;
                }
                QWidget#configSection QSpinBox,
                QWidget#configSection QDoubleSpinBox,
                QWidget#configSection QComboBox {
                    min-height: 0px;
                    max-height: 32px;
                    padding: 3px 8px;
                }
                QWidget#configSection QPushButton {
                    min-height: 0px;
                    max-height: 32px;
                    padding: 3px 10px;
                }
                QGroupBox {
                    background: #ffffff;
                    border: 1px solid #d9e5f1;
                    border-radius: 8px;
                    margin-top: 15px;
                    padding: 12px;
                    font-weight: 600;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                    color: #334155;
                    background: #eef3f8;
                    font-size: 12px;
                    letter-spacing: 0px;
                }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                    background: #fbfdff;
                    border: 1px solid #c7d6e5;
                    border-radius: 6px;
                    padding: 5px 8px;
                    color: #18212f;
                    selection-background-color: #2563eb;
                    selection-color: #ffffff;
                }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
                    border-color: #2f80ed;
                    background: #ffffff;
                }
                QSpinBox, QDoubleSpinBox, QComboBox {
                    min-height: 28px;
                }
                QComboBox#themeCombo {
                    background: #ffffff;
                    border-color: #b9cde2;
                    font-weight: 600;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 24px;
                }
                QComboBox QAbstractItemView {
                    background: #ffffff;
                    border: 1px solid #c7d6e5;
                    color: #18212f;
                    selection-background-color: #e8f2ff;
                    selection-color: #0f172a;
                    outline: none;
                }
                QPushButton {
                    background: #f7faff;
                    border: 1px solid #c9d8e8;
                    border-radius: 7px;
                    padding: 7px 13px;
                    color: #1f2937;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #edf6ff;
                    border-color: #8fb9e8;
                }
                QPushButton:pressed {
                    background: #dceeff;
                }
                QPushButton:disabled {
                    color: #94a3b8;
                    background: #f1f5f9;
                    border-color: #dbe4ee;
                }
                QPushButton[presetButton="true"] {
                    background: #f8fbff;
                    padding: 5px 8px;
                    min-width: 30px;
                    border-color: #d5e2ef;
                    font-weight: 600;
                    color: #475569;
                }
                QPushButton[presetButton="true"]:hover {
                    background: #eaf4ff;
                    border-color: #7bb5f4;
                    color: #1d4ed8;
                }
                QPushButton#outputLogFilterButton {
                    padding: 2px 3px;
                    min-height: 0px;
                    max-height: 32px;
                    font-size: 14px;
                }
                QPushButton[primaryButton="true"] {
                    background: #2563eb;
                    color: #ffffff;
                    border-color: #1d4ed8;
                    font-weight: 700;
                }
                QPushButton[primaryButton="true"]:hover {
                    background: #1d4ed8;
                }
                QPushButton[dangerButton="true"] {
                    background: #fff4f5;
                    border-color: #f2a4ad;
                    color: #b4232d;
                }
                QPushButton[dangerButton="true"]:hover {
                    background: #ffe7ea;
                    border-color: #e65f6d;
                }
                QPushButton[toggleButton="true"]:checked {
                    background: #eafff7;
                    border-color: #34c79a;
                    color: #087f5b;
                }
                QPushButton#generatePreviewVideoButton {
                    background: #fff1f2;
                    border-color: #f43f5e;
                    color: #be123c;
                    font-weight: 700;
                }
                QPushButton#generatePreviewVideoButton:hover {
                    background: #ffe4e6;
                    border-color: #e11d48;
                }
                QPushButton#generatePreviewVideoButton:checked {
                    background: #dcfce7;
                    border-color: #22c55e;
                    color: #166534;
                }
                QLabel {
                    background: transparent;
                }
                QLabel#caseInfoLabel,
                QLabel#runtimeLabel {
                    color: #334155;
                    background: #f8fbff;
                    border: 1px solid #d5e2ef;
                    border-radius: 6px;
                    padding: 6px 8px;
                }
                QLabel#previewSurface {
                    background: #05070b;
                    border: 1px solid #cbd8e6;
                    border-radius: 8px;
                    color: #8b96a6;
                    font-size: 14px;
                }
                QLabel#previewTemplate {
                    background: transparent;
                    border: none;
                    color: #64748b;
                }
                QGroupBox#previewDetailGroup,
                QGroupBox#previewTemplateGroup {
                    padding: 4px;
                }
                QPlainTextEdit#previewInfoDetail {
                    background: transparent;
                    border: none;
                    border-radius: 0;
                    padding: 0;
                    color: #1f2937;
                    font-family: "JetBrains Mono", "SF Mono", "Consolas", monospace;
                    font-size: 12px;
                }
                QPlainTextEdit#outputConsole,
                QTreeWidget#previewInfo {
                    background: #fbfdff;
                    border: 1px solid #d3dfec;
                    border-radius: 8px;
                    color: #1f2937;
                    font-family: "JetBrains Mono", "SF Mono", "Consolas", monospace;
                    font-size: 12px;
                }
                QRadioButton {
                    color: #334155;
                    spacing: 8px;
                }
                QRadioButton::indicator {
                    width: 15px;
                    height: 15px;
                    border-radius: 8px;
                    border: 1px solid #9aaebe;
                    background: #ffffff;
                }
                QRadioButton::indicator:checked {
                    border: 4px solid #2f80ed;
                    background: #ffffff;
                }
                QSplitter::handle {
                    background: #eef3f8;
                }
                QSplitter::handle:hover {
                    background: #d4e3f3;
                }
                QScrollArea {
                    background: transparent;
                    border: none;
                }
                QScrollBar:vertical {
                    background: #eef3f8;
                    width: 10px;
                    margin: 2px;
                }
                QScrollBar::handle:vertical {
                    background: #b8c7d8;
                    border-radius: 5px;
                    min-height: 28px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #8fa4bb;
                }
                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                """
            )
            return

        self.setStyleSheet(
            """
            QWidget {
                background: #0e1116;
                color: #eef2f7;
                font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QWidget#headerBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #182130, stop:0.55 #121820, stop:1 #151a21);
                border: 1px solid #293241;
                border-radius: 10px;
            }
            QWidget#actionBar {
                background: #151a21;
                border: 1px solid #293241;
                border-radius: 8px;
            }
            QWidget#statusStrip {
                background: #151a21;
                border: 1px solid #293241;
                border-radius: 8px;
            }
            QLabel#launcherTitle {
                color: #f8fafc;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#launcherSubtitle {
                color: #9aa4b2;
                font-size: 12px;
            }
            QLabel#headerStatusPill,
            QLabel#headerRuntimePill {
                background: #101722;
                border: 1px solid #334155;
                border-radius: 13px;
                color: #cbd5e1;
                padding: 5px 12px;
                font-weight: 600;
            }
            QLabel#headerStatusPill {
                color: #77e4c8;
                border-color: #1f7a65;
                background: #10231f;
            }
            QLabel#formLabel {
                color: #b8c2d0;
                font-weight: 600;
            }
            QWidget#configPanel,
            QWidget#configItem {
                background: transparent;
            }
            QWidget#configSection {
                background: #151a21;
                border: 1px solid #293241;
                border-radius: 8px;
            }
            QLabel#configSectionTitle {
                color: #5eead4;
                font-size: 12px;
                font-weight: 700;
            }
            QWidget#configSection QSpinBox,
            QWidget#configSection QDoubleSpinBox,
            QWidget#configSection QComboBox {
                min-height: 0px;
                max-height: 32px;
                padding: 3px 8px;
            }
            QWidget#configSection QPushButton {
                min-height: 0px;
                max-height: 32px;
                padding: 3px 10px;
            }
            QGroupBox {
                background: #151a21;
                border: 1px solid #293241;
                border-radius: 8px;
                margin-top: 15px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #d6dde8;
                background: #0e1116;
                font-size: 12px;
                letter-spacing: 0px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background: #0f131a;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 5px 8px;
                color: #edf2f7;
                selection-background-color: #2f80ed;
                selection-color: #ffffff;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
                border-color: #2dd4bf;
                background: #111821;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                min-height: 28px;
            }
            QComboBox#themeCombo {
                background: #101722;
                border-color: #334155;
                font-weight: 600;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: #111821;
                border: 1px solid #334155;
                color: #eef2f7;
                selection-background-color: #1f6feb;
                outline: none;
            }
            QPushButton {
                background: #1b2330;
                border: 1px solid #334155;
                border-radius: 7px;
                padding: 7px 13px;
                color: #e5e7eb;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #243044;
                border-color: #4b647f;
            }
            QPushButton:pressed {
                background: #111827;
            }
            QPushButton:disabled {
                color: #687384;
                background: #141820;
                border-color: #202735;
            }
            QPushButton[presetButton="true"] {
                background: #101722;
                padding: 5px 8px;
                min-width: 30px;
                border-color: #293241;
                font-weight: 500;
                color: #b8c2d0;
            }
            QPushButton[presetButton="true"]:hover {
                background: #162334;
                border-color: #2f80ed;
                color: #d9ecff;
            }
            QPushButton#outputLogFilterButton {
                padding: 2px 3px;
                min-height: 0px;
                max-height: 32px;
                font-size: 14px;
            }
            QPushButton[primaryButton="true"] {
                background: #2f80ed;
                color: #ffffff;
                border-color: #4493ff;
                font-weight: 700;
            }
            QPushButton[primaryButton="true"]:hover {
                background: #1f6feb;
            }
            QPushButton[dangerButton="true"] {
                background: #2a1519;
                border-color: #7f3038;
                color: #ffd8dc;
            }
            QPushButton[dangerButton="true"]:hover {
                background: #3a1b22;
                border-color: #ef6a74;
            }
            QPushButton[toggleButton="true"]:checked {
                background: #10231f;
                border-color: #1f9d7a;
                color: #97f5d2;
            }
            QPushButton#generatePreviewVideoButton {
                background: #3a151c;
                border-color: #e5485b;
                color: #ffd6dc;
                font-weight: 700;
            }
            QPushButton#generatePreviewVideoButton:hover {
                background: #4a1a24;
                border-color: #ff6b7c;
            }
            QPushButton#generatePreviewVideoButton:checked {
                background: #12351f;
                border-color: #2fbd6f;
                color: #a7f3c1;
            }
            QLabel {
                background: transparent;
            }
            QLabel#caseInfoLabel,
            QLabel#runtimeLabel {
                color: #cbd5e1;
                background: #101722;
                border: 1px solid #273142;
                border-radius: 6px;
                padding: 6px 8px;
            }
            QLabel#previewSurface {
                background: #05070b;
                border: 1px solid #293241;
                border-radius: 8px;
                color: #8b96a6;
                font-size: 14px;
            }
            QLabel#previewTemplate {
                background: transparent;
                border: none;
                color: #94a3b8;
            }
            QGroupBox#previewDetailGroup,
            QGroupBox#previewTemplateGroup {
                padding: 4px;
            }
            QPlainTextEdit#previewInfoDetail {
                background: transparent;
                border: none;
                border-radius: 0;
                padding: 0;
                color: #d6dde8;
                font-family: "JetBrains Mono", "SF Mono", "Consolas", monospace;
                font-size: 12px;
            }
            QPlainTextEdit#outputConsole,
            QTreeWidget#previewInfo {
                background: #060912;
                border: 1px solid #273142;
                border-radius: 8px;
                color: #d6dde8;
                font-family: "JetBrains Mono", "SF Mono", "Consolas", monospace;
                font-size: 12px;
            }
            QRadioButton {
                color: #d6dde8;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
                border-radius: 8px;
                border: 1px solid #4b5563;
                background: #0f131a;
            }
            QRadioButton::indicator:checked {
                border: 4px solid #2dd4bf;
                background: #071512;
            }
            QSplitter::handle {
                background: #0e1116;
            }
            QSplitter::handle:hover {
                background: #293241;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #0e1116;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 5px;
                min-height: 28px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4b647f;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

    def _create_spin_with_presets(self, spin, values, suffix: str = "") -> QWidget:
        # Values used to be rendered as a row of preset buttons after every
        # numeric input.  They made the configuration panel unnecessarily tall
        # and squeezed the live preview, so a numeric field now stands alone.
        spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        spin.setMinimumWidth(0)
        return spin

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.page_stack = QStackedWidget()
        root_layout.addWidget(self.page_stack)

        self.launcher_page = QWidget()
        main_layout = QVBoxLayout(self.launcher_page)
        main_layout.setContentsMargins(14, 12, 14, 14)
        main_layout.setSpacing(10)

        header_bar = QWidget()
        header_bar.setObjectName("headerBar")
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(18, 13, 18, 13)
        header_layout.setSpacing(14)

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(3)
        title_label = QLabel("Auto Game Launcher")
        title_label.setObjectName("launcherTitle")
        subtitle_label = QLabel("自动化运行控制台")
        subtitle_label.setObjectName("launcherSubtitle")
        title_column.addWidget(title_label)
        title_column.addWidget(subtitle_label)
        header_layout.addLayout(title_column, 1)

        self.header_runtime_label = QLabel("未开始")
        self.header_runtime_label.setObjectName("headerRuntimePill")
        self.header_status_label = QLabel("待命")
        self.header_status_label.setObjectName("headerStatusPill")
        header_layout.addWidget(self.header_runtime_label)
        header_layout.addWidget(self.header_status_label)
        header_layout.addWidget(self.theme_combo)
        main_layout.addWidget(header_bar, 0)

        controls_widget = QWidget()
        controls_widget.setObjectName("controlsPanel")
        controls_widget.setFixedHeight(302)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 4, 0)
        controls_layout.setSpacing(10)

        launch_row = QHBoxLayout()
        launch_row.setContentsMargins(0, 0, 0, 0)
        launch_row.setSpacing(10)

        mode_group = QGroupBox("启动方式")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(12, 8, 12, 10)
        mode_layout.setSpacing(7)
        mode_layout.addWidget(self.mode_testcase)
        mode_layout.addWidget(self.mode_direct)
        launch_row.addWidget(mode_group, 1)

        testcase_group = QGroupBox("testcases 用例")
        testcase_layout = QHBoxLayout(testcase_group)
        testcase_layout.setContentsMargins(12, 8, 12, 10)
        testcase_layout.setSpacing(8)
        testcase_layout.addWidget(self.testcase_path_edit, 1)
        testcase_layout.addWidget(self.browse_button)
        testcase_layout.addWidget(self.clear_button)
        testcase_layout.addWidget(self.open_label_tool_button)
        testcase_layout.addWidget(self.refresh_button)
        launch_row.addWidget(testcase_group, 2)

        recording_group = QGroupBox("录制回放")
        recording_group.setFixedWidth(188)
        recording_layout = QHBoxLayout(recording_group)
        recording_layout.setContentsMargins(12, 8, 12, 10)
        recording_layout.setSpacing(8)
        recording_layout.addWidget(self.open_game_recording_button)
        recording_layout.addWidget(self.open_game_replay_button)
        launch_row.addWidget(recording_group, 0)
        controls_layout.addLayout(launch_row)

        config_panel = QWidget()
        config_panel.setObjectName("configPanel")
        config_panel.setFixedHeight(184)
        config_layout = QHBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(10)

        def create_config_item(
            label_text: str,
            widget: QWidget,
            field_width: int = 150,
            item_height: int = 32,
        ) -> QWidget:
            item = QWidget()
            item.setObjectName("configItem")
            item.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(8)

            label = QLabel(label_text)
            label.setObjectName("formLabel")
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            label.setFixedWidth(96)
            widget.setFixedWidth(field_width)
            widget.setFixedHeight(item_height)

            item_layout.addStretch(1)
            item_layout.addWidget(label)
            item_layout.addWidget(widget)
            item_layout.addStretch(1)
            return item

        def create_config_section(
            title: str,
            items: tuple[tuple[str, QWidget, int], ...],
        ) -> QWidget:
            section = QWidget()
            section.setObjectName("configSection")
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(8, 6, 8, 6)
            section_layout.setSpacing(4)

            title_label = QLabel(title)
            title_label.setObjectName("configSectionTitle")
            title_label.setFixedHeight(18)
            section_layout.addWidget(title_label)
            item_height = 26 if len(items) > 4 else 32
            for label_text, widget, field_width in items:
                if len(items) > 4:
                    current_item_height = 34 if widget is self.test_profile_field else 24
                else:
                    current_item_height = item_height
                config_item = create_config_item(
                    label_text,
                    widget,
                    field_width,
                    item_height=current_item_height,
                )
                if widget is self.marathon_duration_field:
                    self.marathon_duration_item = config_item
                elif widget is self.marathon_end_battery_field:
                    self.marathon_end_battery_item = config_item
                section_layout.addWidget(config_item)
            section_layout.addStretch(1)
            return section

        # 按用途分成三个等宽区块，每项始终是“标签 + 短控件”。
        config_sections = (
            create_config_section(
                "用例与运行",
                (
                    ("project_case", self.project_combo, 150),
                    ("target_case", self.target_combo, 150),
                    ("运行次数", self.run_count_field, 150),
                    ("单次循环", self.case_loop_count_field, 150),
                ),
            ),
            create_config_section(
                "安全限制",
                (
                    ("安全温度", self.safe_temp_field, 150),
                    ("安全电量", self.safe_battery_field, 150),
                    ("安全时间", self.safe_time_field, 150),
                    ("无操控超时", self.inactivity_timeout_field, 150),
                ),
            ),
            create_config_section(
                "测试与归档",
                (
                    ("测试类型", self.test_profile_field, 170),
                    ("SP运行时长", self.marathon_duration_field, 150),
                    ("结束电量", self.marathon_end_battery_field, 150),
                    ("视频归档", self.generate_preview_video_button, 150),
                    ("回放录像时长", self.power_collection_duration_field, 150),
                ),
            ),
        )
        for section in config_sections:
            config_layout.addWidget(section, 1)
        controls_layout.addWidget(config_panel)

        main_layout.addWidget(controls_widget, 0)

        action_bar = QWidget()
        action_bar.setObjectName("actionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(12, 9, 12, 9)
        action_layout.setSpacing(8)
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.stop_button)
        action_layout.addWidget(self.open_history_button)
        action_layout.addWidget(self.game_process_policy_button)
        action_layout.addWidget(self.preview_overlay_button)
        action_layout.addWidget(self.preview_points_button)
        action_layout.addWidget(self.preview_fullscreen_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self.stream_verify_button)
        action_layout.addWidget(self.hos_frame_rate_label)
        action_layout.addWidget(self.hos_frame_rate_combo)
        main_layout.addWidget(action_bar, 0)

        status_strip = QWidget()
        status_strip.setObjectName("statusStrip")
        status_layout = QHBoxLayout(status_strip)
        status_layout.setContentsMargins(12, 9, 12, 9)
        status_layout.setSpacing(10)
        status_layout.addWidget(self.case_info_label, 1)
        status_layout.addWidget(self.runtime_label, 1)
        main_layout.addWidget(status_strip, 0)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setHandleWidth(8)

        preview_group = QGroupBox("实时可视化")
        preview_group.setObjectName("previewPanel")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(12, 10, 12, 12)
        preview_layout.setSpacing(8)
        self.preview_splitter = QSplitter(Qt.Orientation.Vertical)
        self.preview_splitter.setChildrenCollapsible(False)
        self.preview_splitter.setHandleWidth(8)
        self.preview_splitter.addWidget(self.preview_image_label)
        self.preview_splitter.addWidget(self.preview_info_panel)
        self.preview_splitter.setStretchFactor(0, 4)
        self.preview_splitter.setStretchFactor(1, 1)
        self.preview_splitter.setSizes([620, self.preview_target_info_height])
        preview_layout.addWidget(self.preview_splitter)

        log_group = QGroupBox("运行输出")
        log_group.setObjectName("logPanel")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(12, 10, 12, 12)
        log_filter_layout = QHBoxLayout()
        log_filter_layout.setContentsMargins(0, 0, 0, 0)
        log_filter_layout.setSpacing(6)
        for filter_name in LOG_FILTERS:
            log_filter_layout.addWidget(self.output_filter_buttons[filter_name])
        log_filter_layout.addStretch(1)
        log_layout.addLayout(log_filter_layout)
        log_layout.addWidget(self.output_edit)
        content_splitter.addWidget(preview_group)
        content_splitter.addWidget(log_group)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([860, 340])
        QTimer.singleShot(0, lambda: content_splitter.setSizes([860, 340]))

        main_layout.addWidget(content_splitter, 1)
        self.page_stack.addWidget(self.launcher_page)
        self.preview_fullscreen_page = self._build_preview_fullscreen_page()
        self.page_stack.addWidget(self.preview_fullscreen_page)
        self.label_tool_page = self._build_label_tool_page()
        self.page_stack.addWidget(self.label_tool_page)
        self.game_recording_page = self._build_game_recording_page()
        self.page_stack.addWidget(self.game_recording_page)
        self.game_replay_page = self._build_game_replay_page()
        self.page_stack.addWidget(self.game_replay_page)
        self.history_page = self._build_history_page()
        self.page_stack.addWidget(self.history_page)
        self._update_header_badges()

    def _build_preview_fullscreen_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        title = QLabel("实时预览（放大）")
        title.setObjectName("launcherTitle")
        self.preview_fullscreen_exit_button = QPushButton("退出放大（Esc）")
        toolbar_layout.addWidget(title)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.preview_fullscreen_exit_button)
        layout.addWidget(toolbar, 0)

        self.preview_fullscreen_image_label = QLabel("启动后将在这里实时显示可视化帧")
        self.preview_fullscreen_image_label.setObjectName("previewSurface")
        self.preview_fullscreen_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_fullscreen_image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self.preview_fullscreen_image_label, 1)

        self.preview_fullscreen_escape_shortcut = QShortcut(QKeySequence("Escape"), page)
        self.preview_fullscreen_escape_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.preview_fullscreen_escape_shortcut.activated.connect(self._exit_preview_fullscreen)
        return page

    def _build_label_tool_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        header_bar = QWidget()
        header_bar.setObjectName("headerBar")
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(18, 13, 18, 13)
        header_layout.setSpacing(14)

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(3)
        title_label = QLabel("Auto Game Label Tool")
        title_label.setObjectName("launcherTitle")
        self.label_tool_project_label = QLabel("未加载标注项目")
        self.label_tool_project_label.setObjectName("launcherSubtitle")
        self.label_tool_project_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        title_column.addWidget(title_label)
        title_column.addWidget(self.label_tool_project_label)
        header_layout.addLayout(title_column, 1)

        self.back_to_launcher_button = QPushButton("返回主界面")
        header_layout.addWidget(self.back_to_launcher_button)
        layout.addWidget(header_bar, 0)

        self.label_tool_host = QWidget()
        self.label_tool_host_layout = QVBoxLayout(self.label_tool_host)
        self.label_tool_host_layout.setContentsMargins(0, 0, 0, 0)
        self.label_tool_host_layout.setSpacing(0)
        self.label_tool_empty_label = QLabel("请选择 testcases 用例后打开标注工具")
        self.label_tool_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_tool_empty_label.setObjectName("previewSurface")
        self.label_tool_host_layout.addWidget(self.label_tool_empty_label)
        layout.addWidget(self.label_tool_host, 1)
        return page

    def _build_game_recording_page(self) -> QWidget:
        """与标注工具相同：把录制回放作为 Launcher 内的专用页面。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        header_bar = QWidget()
        header_bar.setObjectName("headerBar")
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(18, 13, 18, 13)
        title_column = QVBoxLayout()
        title = QLabel("Game Recording")
        title.setObjectName("launcherTitle")
        self.game_recording_page_status = QLabel(
            "录制、回放与对比；标注入口位于页面内。"
        )
        self.game_recording_page_status.setObjectName("launcherSubtitle")
        title_column.addWidget(title)
        title_column.addWidget(self.game_recording_page_status)
        header_layout.addLayout(title_column, 1)
        self.game_recording_back_button = QPushButton("返回启动器")
        self.game_recording_back_button.clicked.connect(
            self._close_embedded_game_recording
        )
        header_layout.addWidget(self.game_recording_back_button)
        layout.addWidget(header_bar, 0)

        self.game_recording_host = QWidget()
        self.game_recording_host_layout = QVBoxLayout(self.game_recording_host)
        self.game_recording_host_layout.setContentsMargins(0, 0, 0, 0)
        self.game_recording_host_layout.setSpacing(0)
        self.game_recording_empty_label = QLabel("正在准备录制回放界面……")
        self.game_recording_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.game_recording_empty_label.setObjectName("previewSurface")
        self.game_recording_host_layout.addWidget(self.game_recording_empty_label)
        layout.addWidget(self.game_recording_host, 1)
        return page

    def _build_game_replay_page(self) -> QWidget:
        """独立回放页：记录浏览、视频预览、记录级控点校准和回放入口。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        header_bar = QWidget()
        header_bar.setObjectName("headerBar")
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(18, 13, 18, 13)
        header_layout.setSpacing(14)
        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(3)
        title = QLabel("Game Replay")
        title.setObjectName("launcherTitle")
        self.game_replay_page_status = QLabel(
            "选择历史记录，查看视频、校准该记录的控点后开始回放。"
        )
        self.game_replay_page_status.setObjectName("launcherSubtitle")
        title_column.addWidget(title)
        title_column.addWidget(self.game_replay_page_status)
        header_layout.addLayout(title_column, 1)
        self.game_replay_back_button = QPushButton("返回启动器")
        self.game_replay_back_button.clicked.connect(self._close_game_replay_page)
        header_layout.addWidget(self.game_replay_back_button)
        layout.addWidget(header_bar, 0)

        self.game_replay_host = QWidget()
        self.game_replay_host_layout = QVBoxLayout(self.game_replay_host)
        self.game_replay_host_layout.setContentsMargins(0, 0, 0, 0)
        self.game_replay_host_layout.setSpacing(0)
        self.game_replay_empty_label = QLabel("正在加载回放记录……")
        self.game_replay_empty_label.setObjectName("previewSurface")
        self.game_replay_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.game_replay_host_layout.addWidget(self.game_replay_empty_label)
        layout.addWidget(self.game_replay_host, 1)
        return page

    def _build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        header_bar = QWidget()
        header_bar.setObjectName("headerBar")
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(18, 13, 18, 13)
        header_layout.setSpacing(14)

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(3)
        title_label = QLabel("历史输出管理")
        title_label.setObjectName("launcherTitle")
        self.history_status_label = QLabel("读取 aw/autogame/temp 下的历史运行归档")
        self.history_status_label.setObjectName("launcherSubtitle")
        self.history_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        title_column.addWidget(title_label)
        title_column.addWidget(self.history_status_label)
        header_layout.addLayout(title_column, 1)

        self.history_refresh_button = QPushButton("刷新")
        self.history_open_dir_button = QPushButton("打开目录")
        self.history_open_dir_button.setEnabled(False)
        self.history_delete_button = QPushButton("删除选中")
        self.history_delete_button.setProperty("dangerButton", True)
        self.history_delete_button.setEnabled(False)
        self.history_back_button = QPushButton("返回启动器")
        header_layout.addWidget(self.history_refresh_button)
        header_layout.addWidget(self.history_open_dir_button)
        header_layout.addWidget(self.history_delete_button)
        header_layout.addWidget(self.history_back_button)
        layout.addWidget(header_bar, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        left_group = QGroupBox("历史记录")
        left_layout = QVBoxLayout(left_group)
        left_layout.setContentsMargins(12, 10, 12, 12)
        self.history_tree = QTreeWidget()
        self.history_tree.setHeaderLabels(["批次 / 轮次", "状态", "路径"])
        self.history_tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        left_layout.addWidget(self.history_tree)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        summary_group = QGroupBox("摘要")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setContentsMargins(12, 10, 12, 12)
        self.history_summary_edit = QPlainTextEdit()
        self.history_summary_edit.setReadOnly(True)
        self.history_summary_edit.setMinimumHeight(190)
        self.history_summary_edit.setPlaceholderText("选择一条历史输出后显示摘要...")
        summary_layout.addWidget(self.history_summary_edit)

        frame_group = QGroupBox("逐帧场景日志")
        frame_layout = QVBoxLayout(frame_group)
        frame_layout.setContentsMargins(12, 10, 12, 12)
        frame_layout.setSpacing(8)
        frame_nav_layout = QHBoxLayout()
        frame_nav_layout.setContentsMargins(0, 0, 0, 0)
        frame_nav_layout.setSpacing(8)
        self.history_prev_frame_button = QPushButton("上一帧")
        self.history_next_frame_button = QPushButton("下一帧")
        self.history_prev_frame_shortcut = QShortcut(QKeySequence("A"), page)
        self.history_prev_frame_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.history_prev_frame_shortcut.activated.connect(self._show_previous_history_frame)
        self.history_next_frame_shortcut = QShortcut(QKeySequence("D"), page)
        self.history_next_frame_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.history_next_frame_shortcut.activated.connect(self._show_next_history_frame)
        self.history_frame_counter_label = QLabel("未加载帧")
        self.history_frame_counter_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        frame_nav_layout.addWidget(self.history_prev_frame_button)
        frame_nav_layout.addWidget(self.history_next_frame_button)
        frame_nav_layout.addWidget(self.history_frame_counter_label, 1)
        frame_layout.addLayout(frame_nav_layout)

        frame_splitter = QSplitter(Qt.Orientation.Horizontal)
        frame_splitter.setChildrenCollapsible(False)
        frame_splitter.setHandleWidth(8)
        self.history_frame_image_label = QLabel("选择历史输出后显示帧画面")
        self.history_frame_image_label.setObjectName("previewSurface")
        self.history_frame_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_frame_image_label.setMinimumSize(420, 260)
        self.history_frame_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.history_frame_log_edit = QPlainTextEdit()
        self.history_frame_log_edit.setReadOnly(True)
        self.history_frame_log_edit.setPlaceholderText("选择历史输出后显示这一帧的阶段、info、日志信息和控制信息...")
        frame_splitter.addWidget(self.history_frame_image_label)
        frame_splitter.addWidget(self.history_frame_log_edit)
        frame_splitter.setStretchFactor(0, 3)
        frame_splitter.setStretchFactor(1, 2)
        frame_layout.addWidget(frame_splitter, 1)

        output_group = QGroupBox("launcher 输出")
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(12, 10, 12, 12)
        self.history_output_edit = QPlainTextEdit()
        self.history_output_edit.setReadOnly(True)
        self.history_output_edit.setPlaceholderText("选择一条历史输出后显示 logs/launcher_output.txt...")
        output_layout.addWidget(self.history_output_edit)

        right_layout.addWidget(summary_group, 0)
        right_layout.addWidget(frame_group, 2)
        right_layout.addWidget(output_group, 1)

        splitter.addWidget(left_group)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)
        return page

    def _bind_signals(self):
        self.mode_testcase.toggled.connect(self._sync_mode_ui)
        self.marathon_test_radio.toggled.connect(self._sync_test_profile_ui)
        self.browse_button.clicked.connect(self._choose_testcase_file)
        self.clear_button.clicked.connect(self._reselect_testcase_file)
        self.open_label_tool_button.clicked.connect(self._open_label_tool_for_selected_case)
        self.open_game_recording_button.clicked.connect(self._open_game_recording)
        self.open_game_replay_button.clicked.connect(self._open_game_replay)
        self.back_to_launcher_button.clicked.connect(self._return_from_label_tool)
        self.refresh_button.clicked.connect(self._refresh_config_choices)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        self.start_button.clicked.connect(self._start_run)
        self.stream_verify_button.clicked.connect(self._toggle_stream_verification)
        self.hos_frame_rate_combo.currentIndexChanged.connect(self._on_hos_frame_rate_changed)
        self.stop_button.clicked.connect(self._stop_run)
        self.open_history_button.clicked.connect(self._show_history_page)
        self.history_refresh_button.clicked.connect(self._refresh_history_outputs)
        self.history_open_dir_button.clicked.connect(self._open_selected_history_dir)
        self.history_delete_button.clicked.connect(self._delete_selected_history_output)
        self.history_back_button.clicked.connect(self._show_launcher_page)
        self.history_tree.itemSelectionChanged.connect(self._on_history_selection_changed)
        self.history_prev_frame_button.clicked.connect(self._show_previous_history_frame)
        self.history_next_frame_button.clicked.connect(self._show_next_history_frame)
        self.game_process_policy_button.toggled.connect(self._toggle_game_process_policy)
        self.generate_preview_video_button.toggled.connect(self._toggle_generate_preview_video)
        self.preview_overlay_button.toggled.connect(self._toggle_preview_overlay)
        self.preview_points_button.toggled.connect(self._toggle_preview_points)
        self.preview_info_select_all_button.clicked.connect(self._select_all_preview_info)
        self.preview_info_select_none_button.clicked.connect(self._select_no_preview_info)
        self.preview_info_tree.itemChanged.connect(self._on_preview_info_item_changed)
        self.preview_info_tree.itemSelectionChanged.connect(
            self._on_preview_info_selection_changed
        )
        self.preview_fullscreen_button.clicked.connect(self._toggle_preview_fullscreen)
        self.preview_fullscreen_exit_button.clicked.connect(self._exit_preview_fullscreen)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        for filter_name, button in self.output_filter_buttons.items():
            button.clicked.connect(lambda checked=False, name=filter_name: self._set_output_log_filter(name))
        self.preview_timer.timeout.connect(self._poll_preview_frame)
        self.stream_verify_timer.timeout.connect(self._poll_stream_verification_frame)
        self.safety_timer.timeout.connect(self._check_and_start_if_safe)
        self.run_timeout_timer.timeout.connect(self._handle_run_timeout)
        self.stream_disconnect_signal_timer.timeout.connect(self._poll_stream_disconnect_signal)
        LOGGER.debug("signals bound")

    def _insert_output_text(self, text: str):
        scrollbar = self.output_edit.verticalScrollBar()
        old_scroll_value = scrollbar.value()
        should_follow = scrollbar.value() >= max(0, scrollbar.maximum() - 4)

        cursor = QTextCursor(self.output_edit.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)

        if should_follow:
            self.output_edit.moveCursor(QTextCursor.MoveOperation.End)
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(old_scroll_value)

    def _start_output_log_spool(self, archive_dir: Path):
        initial_text = ""
        if self.output_log_spool_path is None:
            initial_text = "".join(line for _, line in self.output_log_entries)
        self.output_log_spool_path = Path(archive_dir) / "launcher_output.txt"
        self.output_log_spool_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_log_spool_path.write_text(initial_text, encoding="utf-8")

    def _append_output_log_spool(self, entries):
        path = self.output_log_spool_path
        if path is None or not entries:
            return
        text = "".join(line for _, line in entries)
        if not text:
            return
        with path.open("a", encoding="utf-8", newline="") as spool:
            spool.write(text)

    def _current_output_offset(self) -> int:
        path = self.output_log_spool_path
        if path is None or not path.exists():
            return len(self._all_output_text().encode("utf-8"))
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0

    def _output_text_since(self, offset: int) -> str:
        path = self.output_log_spool_path
        if path is None or not path.exists():
            return self._all_output_text()
        try:
            with path.open("rb") as spool:
                spool.seek(max(0, int(offset or 0)))
                return spool.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _record_output_text(self, text: str):
        entries = decode_output_text(text)
        self._append_output_log_spool(entries)
        self.output_log_entries.extend(entries)
        overflow = len(self.output_log_entries) - OUTPUT_MEMORY_MAX_ENTRIES
        if overflow > 0:
            del self.output_log_entries[:overflow]
        return entries

    def _all_output_text(self) -> str:
        path = self.output_log_spool_path
        if path is not None and path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        return "".join(line for _, line in self.output_log_entries)

    def _filtered_output_text(self) -> str:
        if self.output_log_filter == LOG_FILTER_ALL:
            return "".join(line for _, line in self.output_log_entries)
        return "".join(
            line
            for category, line in self.output_log_entries
            if category == self.output_log_filter
        )

    def _render_output_filter(self):
        self.output_edit.setPlainText(self._filtered_output_text())
        self.output_edit.moveCursor(QTextCursor.MoveOperation.End)

    def _set_output_log_filter(self, filter_name: str):
        if filter_name not in LOG_FILTERS:
            return

        self.output_log_filter = filter_name
        for name, button in self.output_filter_buttons.items():
            button.setChecked(name == filter_name)
        self._render_output_filter()

    def _append_output(self, text: str):
        if not text:
            return

        entries = self._record_output_text(text)
        visible_text = "".join(
            line
            for category, line in entries
            if self.output_log_filter == LOG_FILTER_ALL
            or category == self.output_log_filter
        )
        if visible_text:
            self._insert_output_text(visible_text)

    def _log_message(self, text: str, level: int = logging.INFO):
        self._append_output(text)
        message = text.rstrip()
        if message:
            LOGGER.log(level, message)

    def _update_header_badges(self):
        if not hasattr(self, "header_status_label") or not hasattr(self, "header_runtime_label"):
            return

        if self.stop_requested:
            state_text = "停止中"
        elif self.process is not None:
            state_text = "运行中"
        elif self.batch_active:
            state_text = "等待中"
        else:
            state_text = "待命"

        if self.current_plan is not None:
            total_runs = int(self.current_plan.get("run_count", self.total_runs))
            progress = min(self.current_run_index + (1 if self.process is not None else 0), total_runs)
            runtime_text = f"{progress}/{total_runs}"
        else:
            runtime_text = "未开始"

        self.header_status_label.setText(state_text)
        self.header_runtime_label.setText(runtime_text)

    @staticmethod
    def _format_information_text(prefix: str, text: str) -> str:
        value = str(text or "").strip()
        normalized_prefix = f"{prefix}："
        if value.startswith(normalized_prefix) or value.startswith(f"{prefix}:"):
            return value
        return f"{normalized_prefix}{value}"

    def _set_case_info(self, text: str):
        self.case_info_label.setText(
            self._format_information_text("用例信息", text)
        )
        self._update_header_badges()

    def _set_runtime(self, text: str):
        self.runtime_label.setText(
            self._format_information_text("运行信息", text)
        )
        self._update_header_badges()

    def _set_status(self, text: str):
        """兼容旧的运行状态调用；运行过程不再覆盖用例信息。"""
        self._set_runtime(text)

    def _toggle_preview_overlay(self, checked: bool):
        self.preview_overlay_button.setText("隐藏标注" if checked else "显示标注")
        LOGGER.info("preview overlay toggled: %s", checked)
        self._refresh_preview_pixmap()

    def _toggle_preview_points(self, checked: bool):
        self.preview_points_button.setText("隐藏控点" if checked else "显示控点")
        LOGGER.info("preview points toggled: %s", checked)
        self._refresh_preview_pixmap()

    def _set_all_preview_info_checked(self, checked: bool):
        previous_signals_blocked = self.preview_info_tree.blockSignals(True)
        try:
            check_state = (
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
            for item in self.preview_info_items.values():
                item.setCheckState(0, check_state)
        finally:
            self.preview_info_tree.blockSignals(previous_signals_blocked)
        self._sync_preview_master_buttons_from_checks()
        self._refresh_preview_pixmap()

    def _select_all_preview_info(self):
        self._set_all_preview_info_checked(True)

    def _select_no_preview_info(self):
        self._set_all_preview_info_checked(False)

    def _on_preview_info_item_changed(self, _item, column: int):
        if column == 0:
            self._sync_preview_master_buttons_from_checks()
            self._refresh_preview_pixmap()

    def _sync_preview_master_buttons_from_checks(self):
        overlay_checked = False
        points_checked = False
        for info_key, item in self.preview_info_items.items():
            if item.checkState(0) != Qt.CheckState.Checked:
                continue
            item_type = self.preview_info_item_types.get(info_key)
            if item_type in {"areas", "special_areas"}:
                overlay_checked = True
            elif item_type == "points":
                points_checked = True

        button_states = (
            (self.preview_overlay_button, overlay_checked, "隐藏标注", "显示标注"),
            (self.preview_points_button, points_checked, "隐藏控点", "显示控点"),
        )
        for button, checked, checked_text, unchecked_text in button_states:
            previous_signals_blocked = button.blockSignals(True)
            button.setChecked(checked)
            button.setText(checked_text if checked else unchecked_text)
            button.blockSignals(previous_signals_blocked)

    def _reset_preview_info_template(self, message: str):
        self.preview_info_template_pixmap = None
        self.preview_info_template_label.setPixmap(QPixmap())
        self.preview_info_template_label.setText(message)
        self.preview_info_template_label.setToolTip("")

    def _reset_preview_info_detail(self, message: str):
        self.preview_info_detail_edit.setPlainText(message)

    def _refresh_preview_info_detail(self, payload=None):
        selected_items = self.preview_info_tree.selectedItems()
        if not selected_items:
            self._reset_preview_info_detail(
                "选中区域、特殊区域或控点后查看当前帧完整信息"
            )
            return

        item = selected_items[0]
        row_key = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
        info_key = self.preview_info_result_keys.get(row_key, "")
        item_type = item.text(0).strip() or "条目"
        item_name = item.text(1).strip() or row_key or "未命名"
        payload = payload if isinstance(payload, dict) else self.latest_preview_payload
        info_payload = (
            payload.get("info")
            if isinstance(payload, dict) and isinstance(payload.get("info"), dict)
            else {}
        )
        if not info_key or info_key not in info_payload:
            detail_text = f"{item_type}：{item_name}\n\n当前帧无该项信息"
        else:
            full_value = format_preview_info_detail(info_payload[info_key])
            detail_text = f"{item_type}：{item_name}\n帧信息：\n{full_value}"

        if self.preview_info_detail_edit.toPlainText() != detail_text:
            self.preview_info_detail_edit.setPlainText(detail_text)
            self.preview_info_detail_edit.moveCursor(QTextCursor.MoveOperation.Start)

    def _render_preview_info_template(self):
        label = getattr(self, "preview_info_template_label", None)
        pixmap = getattr(self, "preview_info_template_pixmap", None)
        if label is None or pixmap is None or pixmap.isNull():
            return
        target_width = max(1, label.width() - 8)
        target_height = max(1, label.height() - 8)
        scaled = pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setText("")
        label.setPixmap(scaled)

    def _on_preview_info_selection_changed(self):
        self._refresh_preview_info_detail()
        selected_items = self.preview_info_tree.selectedItems()
        if not selected_items:
            self._reset_preview_info_template("点击特殊区域或区域查看模板图片")
            return

        item = selected_items[0]
        row_key = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
        item_type = self.preview_info_item_types.get(row_key)
        if item_type not in {"special_areas", "areas"}:
            message = "控点没有模板图片" if item_type == "points" else "该条目没有模板图片"
            self._reset_preview_info_template(message)
            return

        template_path = self.preview_info_template_paths.get(row_key, "").strip()
        if not template_path:
            self._reset_preview_info_template("该区域未配置模板图片")
            return

        path = Path(template_path)
        if not path.is_absolute():
            project_case = self._get_preview_project_case()
            path = (
                ROOT_DIR
                / "aw"
                / "autogame"
                / "customs_examples"
                / project_case
                / path
            )
        path = path.resolve()
        if not path.is_file():
            self._reset_preview_info_template(f"模板图片不存在：{path.name}")
            return

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._reset_preview_info_template(f"模板图片读取失败：{path.name}")
            return
        self.preview_info_template_pixmap = pixmap
        self.preview_info_template_label.setToolTip(str(path))
        self._render_preview_info_template()

    def _toggle_preview_fullscreen(self):
        if self.page_stack.currentWidget() is self.preview_fullscreen_page:
            self._exit_preview_fullscreen()
            return
        LOGGER.info("open fullscreen preview")
        self.page_stack.setCurrentWidget(self.preview_fullscreen_page)
        self._refresh_preview_pixmap()

    def _exit_preview_fullscreen(self):
        if self.page_stack.currentWidget() is not self.preview_fullscreen_page:
            return
        LOGGER.info("close fullscreen preview")
        self.page_stack.setCurrentWidget(self.launcher_page)
        self._refresh_preview_pixmap()

    def _on_theme_changed(self):
        theme_mode = self.theme_combo.currentData() or "light"
        if theme_mode == self.theme_mode:
            return
        self.theme_mode = str(theme_mode)
        LOGGER.info("launcher theme changed: %s", self.theme_mode)
        self._apply_style()
        self._refresh_preview_pixmap()

    def _toggle_game_process_policy(self, preserve_process: bool):
        self._sync_game_process_policy_ui()
        LOGGER.info(
            "game process policy toggled: %s",
            "preserve" if preserve_process else "close",
        )

    def _sync_test_profile_ui(self, _checked: bool = False):
        marathon_selected = self.marathon_test_radio.isChecked()
        editable = self.inputs_enabled
        self.marathon_duration_item.setVisible(marathon_selected)
        self.marathon_end_battery_item.setVisible(marathon_selected)
        self.marathon_duration_spin.setEnabled(editable and marathon_selected)
        self.marathon_end_battery_spin.setEnabled(editable and marathon_selected)
        self.run_count_spin.setEnabled(editable and not marathon_selected)
        self.safe_temp_spin.setEnabled(editable and not marathon_selected)
        self.safe_battery_spin.setEnabled(editable and not marathon_selected)
        self.inactivity_timeout_spin.setEnabled(editable and not marathon_selected)
        if marathon_selected:
            self.game_process_policy_button.setChecked(False)
        self._sync_game_process_policy_ui()

    def _sync_game_process_policy_ui(self):
        marathon_selected = self.marathon_test_radio.isChecked()
        preserve_process = self.game_process_policy_button.isChecked() and not marathon_selected
        self.game_process_policy_button.setText(
            "保留进程" if preserve_process else "关闭进程"
        )
        self.game_process_policy_button.setEnabled(
            self.inputs_enabled and not marathon_selected
        )
        self.game_process_policy_button.setToolTip(
            "马拉松每轮结束后固定关闭游戏和 SP 进程"
            if marathon_selected
            else "当前为保留进程：功耗测试和功能测试在启动、手动停止、自动结束时都不关闭相关应用"
            if preserve_process
            else "当前为关闭进程：功耗测试和功能测试在启动、手动停止、自动结束时都会关闭相关应用"
        )

    def _toggle_generate_preview_video(self, checked: bool):
        self.generate_preview_video_button.setText("生成视频：开" if checked else "生成视频：关")
        LOGGER.info("generate preview video toggled: %s", checked)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adjust_preview_splitter_sizes()
        self._refresh_preview_pixmap()
        self._render_preview_info_template()

    def closeEvent(self, event):
        self._stop_stream_verification("")
        if self.batch_active or self.process is not None:
            self._close_after_stop = True
            self._stop_run()
            event.ignore()
            return
        if self._game_recording_is_running():
            self._close_embedded_game_recording()
        self._close_game_replay_runtime()
        if self.game_replay_panel is not None:
            self.game_replay_panel.stop()
        self.process_launch_tracer.stop()
        super().closeEvent(event)

    def _sync_mode_ui(self):
        LOGGER.debug("sync_mode_ui: testcase_mode=%s", self.mode_testcase.isChecked())
        self._sync_testcase_controls_state()
        self._refresh_case_info_display()

    def _refresh_case_info_display(self):
        if self.mode_testcase.isChecked():
            if self.selected_testcase_file is None:
                self._set_case_info("未选择 testcases 用例")
                return

            description = self.selected_testcase_description.strip()
            if description:
                self._set_case_info(description)
                return

            try:
                case_path = self.selected_testcase_file.relative_to(APP_DIR).as_posix()
            except ValueError:
                case_path = str(self.selected_testcase_file)
            self._set_case_info(f"{case_path} 未提供 testcase_description")
            return

        project_case = self.project_combo.currentText().strip()
        target_case = self.target_combo.currentText().strip()
        if project_case and target_case:
            self._set_case_info(
                f"直接启动 project_case={project_case}，target_case={target_case}"
            )
        else:
            self._set_case_info("直接启动模式：请选择 project_case 和 target_case")

    def _can_open_label_tool_for_selection(self) -> bool:
        if self.selected_testcase_file is None:
            return False
        project_case = self.project_combo.currentText().strip()
        return resolve_label_project_dir(project_case) is not None

    def _game_recording_is_running(self) -> bool:
        return self.game_recording_window is not None

    def _sync_testcase_controls_state(self):
        testcase_mode = self.mode_testcase.isChecked()
        has_selection = self.selected_testcase_file is not None
        choose_text, reselect_text = get_testcase_button_texts(has_selection)
        can_use_testcase_controls = self.inputs_enabled and testcase_mode

        self.testcase_path_edit.setEnabled(can_use_testcase_controls)
        self.browse_button.setText(choose_text)
        self.browse_button.setEnabled(can_use_testcase_controls and not has_selection)
        self.clear_button.setText(reselect_text)
        self.clear_button.setEnabled(can_use_testcase_controls and has_selection)
        self.open_label_tool_button.setEnabled(
            can_use_testcase_controls and self._can_open_label_tool_for_selection()
        )
        game_recording_ready = (
            self.inputs_enabled
            and not self._game_recording_is_running()
            and (
                bool(getattr(sys, "frozen", False))
                or (
                    GAME_RECORDING_PROJECT_DIR.is_dir()
                    and (GAME_RECORDING_PROJECT_DIR / "info.py").is_file()
                )
            )
        )
        self.open_game_recording_button.setEnabled(game_recording_ready)
        self.open_game_replay_button.setEnabled(self.inputs_enabled)

    def _set_combo_value(self, combo: QComboBox, value: str):
        if not value:
            return
        index = combo.findText(value)
        if index < 0:
            combo.addItem(value)
            index = combo.findText(value)
        combo.setCurrentIndex(index)

    def _load_project_cases(self, preferred: Optional[str] = None):
        current = preferred or self.project_combo.currentText()
        LOGGER.debug("load_project_cases: preferred=%s current=%s", preferred, current)
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        self.project_combo.addItems(discover_project_cases())
        self.project_combo.blockSignals(False)

        if current:
            self._set_combo_value(self.project_combo, current)
        elif self.project_combo.count() > 0:
            self.project_combo.setCurrentIndex(0)

        self._load_target_cases(preferred=None)

    def _load_target_cases(self, preferred: Optional[str]):
        project_case = self.project_combo.currentText().strip()
        current = preferred or self.target_combo.currentText()
        LOGGER.debug(
            "load_target_cases: project_case=%s preferred=%s current=%s",
            project_case,
            preferred,
            current,
        )
        self._updating_targets = True
        self.target_combo.clear()
        self.target_combo.addItems(discover_target_cases(project_case))
        if current:
            self._set_combo_value(self.target_combo, current)
        elif self.target_combo.count() > 0:
            self.target_combo.setCurrentIndex(0)
        self._updating_targets = False

    def _refresh_config_choices(self):
        project = self.project_combo.currentText().strip()
        target = self.target_combo.currentText().strip()
        LOGGER.info("refresh_config_choices: project=%s target=%s", project, target)
        self.preview_info_cache.clear()
        self.stage_info_cache.clear()
        self._load_project_cases(preferred=project)
        self._load_target_cases(preferred=target)
        self._sync_testcase_controls_state()
        self._set_status("已刷新 project_case 和 target_case 列表。")

    def _on_project_changed(self, project_case: str):
        if self._updating_targets:
            return
        LOGGER.info("project changed: %s", project_case)
        self._load_target_cases(preferred=None)
        if project_case:
            self._set_status(f"已选择 project_case={project_case}，请确认 target_case。")
        self._sync_testcase_controls_state()
        self._refresh_case_info_display()
        self._refresh_preview_pixmap()

    def _on_target_changed(self, _target_case: str):
        if self._updating_targets:
            return
        self._refresh_case_info_display()

    def _show_validation_issues(self, dialog_title: str, issues: ValidationIssues) -> bool:
        for warning_title, warning_message in issues.warnings:
            self._log_message(
                f"[Launcher] 非阻断校验提示：{warning_title}：{warning_message}\n",
                level=logging.WARNING,
            )

        if not issues.has_errors():
            return False

        lines = ["以下问题需要处理：", ""]
        for index, (error_title, error_message) in enumerate(issues.errors, start=1):
            lines.append(f"{index}. {error_title}：{error_message}")
        message = "\n".join(lines)
        self._log_message(f"[Launcher] {dialog_title}：{message}\n", level=logging.ERROR)
        QMessageBox.warning(self, dialog_title, message)
        return True

    def _choose_testcase_file(self) -> bool:
        LOGGER.info("choose_testcase_file dialog open")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 testcases 用例",
            str(TESTCASES_DIR),
            "Python Files (*.py)",
        )
        if not file_path:
            LOGGER.info("choose_testcase_file canceled")
            return False

        py_file = Path(file_path).resolve()
        try:
            py_file.relative_to(TESTCASES_DIR)
            rel_path = py_file.relative_to(APP_DIR)
        except ValueError:
            LOGGER.warning("choose_testcase_file invalid path: %s", py_file)
            issues = ValidationIssues()
            issues.add_error("路径错误", "请选择当前项目 testcases 目录下的用例文件。")
            self._show_validation_issues("无法选择用例", issues)
            return False

        self.selected_testcase_file = py_file
        LOGGER.info("choose_testcase_file selected: %s", py_file)
        self.testcase_path_edit.setText(rel_path.as_posix())
        self.mode_testcase.setChecked(True)
        self._apply_parsed_testcase(py_file)
        self._sync_testcase_controls_state()
        return True

    def _reselect_testcase_file(self):
        LOGGER.info("reselect_testcase_file")
        self._choose_testcase_file()

    def _clear_testcase_file(self):
        LOGGER.info("clear_testcase_file")
        self.selected_testcase_file = None
        self.selected_testcase_description = ""
        self.testcase_path_edit.clear()
        self._sync_testcase_controls_state()
        self._refresh_case_info_display()
        self._set_status("已清空 testcases 选择。可以直接指定 project_case / target_case 启动。")

    def _ensure_label_tool(self):
        if self.label_tool is not None:
            return

        LOGGER.info("initializing embedded label tool")
        from aw.autogame.tools.Label import AutoStudioWindow

        self.label_tool_empty_label.hide()
        self.label_tool = AutoStudioWindow()
        self.label_tool.setWindowFlags(Qt.WindowType.Widget)
        self.label_tool_host_layout.addWidget(self.label_tool)

    def _open_game_recording(self):
        if self._game_recording_is_running():
            self.page_stack.setCurrentWidget(self.game_recording_page)
            return
        if (
            not getattr(sys, "frozen", False)
            and (
                not (GAME_RECORDING_PROJECT_DIR / "info.py").is_file()
            )
        ):
            QMessageBox.warning(
                self,
                "无法打开录制回放",
                "未找到 Game_Recording/info.py，请检查工程文件。",
            )
            return
        self.page_stack.setCurrentWidget(self.game_recording_page)
        self.game_recording_empty_label.setText("正在启动录制回放和日志采集……")
        QApplication.processEvents()

        from aw.autogame.customs_examples.Game_Recording.resource.main_app import (
            create_main_window,
        )
        from aw.autogame.customs_examples.Game_Recording.resource.runtime_log import (
            HdcDebugLogCapture,
            HilogCapture,
            create_run_directory,
            save_run_summary,
        )

        started_at = datetime.now().astimezone()
        records_root = ROOT_DIR / "aw" / "autogame" / "records" / "Game_Recording"
        run_dir = None
        hdc_capture = None
        hilog_capture = None
        try:
            run_dir = create_run_directory(records_root, now=started_at)
            runtime_log_path = run_dir / "start_record.log"
            runtime_log_path.write_text(
                "[Game Recording] Launcher 内嵌页面启动\n",
                encoding="utf-8",
            )
            hdc_capture = HdcDebugLogCapture(run_dir)
            hdc_capture.__enter__()
            hilog_capture = HilogCapture(run_dir)
            hilog_capture.__enter__()
            window = create_main_window(
                output_root=run_dir,
                runtime_log_path=runtime_log_path,
                hilog_capture=hilog_capture,
                open_label_tool_callback=self._open_game_recording_label_tool,
                parent=self,
            )
            if window is None:
                hilog_capture.stop()
                hdc_capture.stop()
                save_run_summary(run_dir, started_at, "cancelled", 0)
                self._show_launcher_page()
                return
        except Exception as exc:
            LOGGER.exception("无法启动 Launcher 内嵌 Game Recording 页面")
            if hilog_capture is not None:
                hilog_capture.stop()
            if hdc_capture is not None:
                hdc_capture.stop()
            if run_dir is not None:
                save_run_summary(run_dir, started_at, "failed", 1, str(exc))
            self._show_launcher_page()
            QMessageBox.critical(self, "无法打开录制回放", str(exc))
            return

        window.setWindowFlags(Qt.WindowType.Widget)
        self.game_recording_empty_label.hide()
        self.game_recording_host_layout.addWidget(window)
        window.show()
        window.recorder_window.setFocus()
        self.game_recording_window = window
        self.game_recording_run_dir = run_dir
        self.game_recording_started_at = started_at
        self.game_recording_hdc_capture = hdc_capture
        self.game_recording_hilog_capture = hilog_capture
        self.game_recording_page_status.setText(
            f"记录目录：{run_dir}"
        )
        self._set_status("已进入 Game Recording 专用页面。")
        self._sync_testcase_controls_state()

    def _open_game_replay(self):
        self.page_stack.setCurrentWidget(self.game_replay_page)
        QApplication.processEvents()
        if self.game_replay_panel is None:
            try:
                from aw.autogame.customs_examples.Game_Recording.resource.launcher_replay import (
                    LauncherReplayPanel,
                )

                records_root = (
                    ROOT_DIR / "aw" / "autogame" / "records" / "Game_Recording"
                )
                panel = LauncherReplayPanel(records_root, parent=self.game_replay_host)
                panel.replayRequested.connect(self._start_selected_game_replay)
                self.game_replay_empty_label.hide()
                self.game_replay_host_layout.addWidget(panel)
                self.game_replay_panel = panel
            except Exception as exc:
                LOGGER.exception("无法加载 Launcher 回放页面")
                self.game_replay_empty_label.setText(f"回放页面加载失败：{exc}")
                QMessageBox.critical(self, "无法打开回放页面", str(exc))
                return
        else:
            self.game_replay_panel.refresh_records()
        self.game_replay_panel.set_replay_active(self.game_replay_window is not None)
        self.game_replay_page_status.setText(
            f"记录目录：{self.game_replay_panel.records_root}"
        )
        self._set_status("已进入独立回放页面。")

    def _start_selected_game_replay(self, record):
        active_window = self.game_replay_window
        if active_window is not None:
            self._set_status("当前记录正在回放。")
            return
        try:
            from aw.autogame.customs_examples.Game_Recording.resource.replay_app import (
                ReplayWindow,
            )

            window = ReplayWindow(record=record, background=True)
            window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            window.replayProgress.connect(self._on_game_replay_progress)
            window.replayFinished.connect(self._on_game_replay_finished)
            window.destroyed.connect(self._on_game_replay_window_destroyed)
            self.game_replay_window = window
            if self.game_replay_panel is not None:
                self.game_replay_panel.set_replay_active(True)
            self._set_status(f"正在后台回放记录：{record.directory.name}")
        except Exception as exc:
            LOGGER.exception("无法启动所选回放记录")
            self.game_replay_window = None
            if self.game_replay_panel is not None:
                self.game_replay_panel.set_replay_active(False)
            QMessageBox.critical(self, "无法开始回放", str(exc))

    def _on_game_replay_window_destroyed(self, *_args):
        self.game_replay_window = None
        if self.game_replay_panel is not None:
            self.game_replay_panel.set_replay_active(False, reset_status=False)
        self._sync_testcase_controls_state()

    def _on_game_replay_progress(self, progress: float, action: str):
        if self.game_replay_panel is not None:
            self.game_replay_panel.set_replay_progress(progress, action)

    def _on_game_replay_finished(self, success: bool, message: str):
        if self.game_replay_panel is not None:
            self.game_replay_panel.set_replay_result(success, message)
        self._set_status(
            f"回放{'完成' if success else '失败'}：{message}"
        )

    def _close_game_replay_runtime(self):
        window, self.game_replay_window = self.game_replay_window, None
        if window is not None:
            window.close()
        if self.game_replay_panel is not None:
            self.game_replay_panel.set_replay_active(False)

    def _close_game_replay_page(self):
        if self.game_replay_window is not None:
            self.game_replay_page_status.setText("回放正在后台执行，完成后可再返回。")
            return
        if self.game_replay_panel is not None:
            self.game_replay_panel.stop()
        self._show_launcher_page()

    def _close_embedded_game_recording(self):
        """返回 Launcher 时同步停止抓流、hilog 和本次 HDC DEBUG 归档。"""
        window = self.game_recording_window
        run_dir = self.game_recording_run_dir
        started_at = self.game_recording_started_at
        hdc_capture = self.game_recording_hdc_capture
        hilog_capture = self.game_recording_hilog_capture
        error = ""
        try:
            if window is not None:
                window.shutdown()
                self.game_recording_host_layout.removeWidget(window)
                window.setParent(None)
                window.deleteLater()
        except Exception as exc:
            error = str(exc)
            LOGGER.exception("关闭 Launcher 内嵌 Game Recording 失败")
        finally:
            if hilog_capture is not None:
                hilog_capture.stop()
            if hdc_capture is not None:
                hdc_capture.stop()
            if run_dir is not None and started_at is not None:
                from aw.autogame.customs_examples.Game_Recording.resource.runtime_log import (
                    save_run_summary,
                )
                save_run_summary(
                    run_dir,
                    started_at,
                    "success" if not error else "failed",
                    0 if not error else 1,
                    error,
                )
            self.game_recording_window = None
            self.game_recording_run_dir = None
            self.game_recording_started_at = None
            self.game_recording_hdc_capture = None
            self.game_recording_hilog_capture = None
            self.game_recording_empty_label.show()
            self.game_recording_empty_label.setText("录制回放已关闭。")
            self._show_launcher_page()

    def _open_game_recording_label_tool(self):
        """从录制回放页进入标注页，返回时恢复到原录制回放页。"""
        recording_window = self.game_recording_window
        if (
            recording_window is not None
            and recording_window.recorder_window.recorder.is_recording
        ):
            QMessageBox.information(
                self,
                "请先结束录制",
                "请先点击“关闭录制”保存当前记录，再修改控点和确认新的键位绑定。",
            )
            return False
        if not (GAME_RECORDING_PROJECT_DIR / "info.py").is_file():
            QMessageBox.warning(
                self,
                "无法打开标注工具",
                "未找到 Game_Recording/info.py，请检查工程文件。",
            )
            return False
        return self._open_label_project(
            "Game_Recording",
            GAME_RECORDING_PROJECT_DIR,
            return_page=self.game_recording_page,
        )

    def _open_label_project(self, project_case: str, project_dir: Path, return_page=None):
        try:
            self._ensure_label_tool()
            self.label_tool.load_project_from_dir(str(project_dir))
        except Exception as exc:
            log_exception(f"open label tool failed: project_dir={project_dir}")
            issues = ValidationIssues()
            issues.add_error("打开失败", f"无法打开标注工具：{exc}")
            self._show_validation_issues("无法打开标注工具", issues)
            return False

        self.label_tool_project_dir = project_dir
        self.label_tool_project_label.setText(
            f"当前标注项目：{project_case}    {project_dir}"
        )
        self.label_tool_return_page = return_page
        self.back_to_launcher_button.setText(
            "返回录制回放" if return_page is self.game_recording_page else "返回主界面"
        )
        self.page_stack.setCurrentWidget(self.label_tool_page)
        self._set_status(f"已打开标注工具：{project_case}")
        return True

    def _open_label_tool_for_selected_case(self):
        LOGGER.info(
            "open_label_tool_for_selected_case: testcase=%s project=%s",
            self.selected_testcase_file,
            self.project_combo.currentText().strip(),
        )
        issues = ValidationIssues()
        if self.selected_testcase_file is None:
            issues.add_error("缺少用例", "请先选择一个 testcases 用例。")
        project_case = self.project_combo.currentText().strip()
        project_dir = resolve_label_project_dir(project_case)
        if not project_case:
            issues.add_error("缺少配置", "请选择 project_case。")
        elif project_dir is None:
            issues.add_error(
                "缺少标注资源",
                f"未找到 project_case={project_case} 对应的标注资源目录或 info.py。",
            )
        if self._show_validation_issues("无法打开标注工具", issues):
            self._sync_testcase_controls_state()
            return

        self._open_label_project(project_case, project_dir)

    def _return_from_label_tool(self):
        return_page = self.label_tool_return_page
        if (
            return_page is self.game_recording_page
            and self._game_recording_is_running()
        ):
            try:
                if not self.game_recording_window.refresh_bindings(parent=self):
                    return
            except Exception as exc:
                LOGGER.exception("重新扫描 Game Recording 控点或确认绑定失败")
                QMessageBox.critical(
                    self,
                    "无法返回录制回放",
                    f"重新扫描控点或确认键位绑定失败：{exc}",
                )
                return
            self.label_tool_return_page = None
            self.back_to_launcher_button.setText("返回主界面")
            self.page_stack.setCurrentWidget(self.game_recording_page)
            self.game_recording_window.recorder_window.setFocus()
            self._set_status("已返回录制回放。")
            return
        self.label_tool_return_page = None
        self.back_to_launcher_button.setText("返回主界面")
        self._show_launcher_page()

    def _show_launcher_page(self):
        LOGGER.info("show launcher page")
        project = self.project_combo.currentText().strip()
        target = self.target_combo.currentText().strip()
        self.preview_info_cache.clear()
        self.stage_info_cache.clear()
        self.page_stack.setCurrentWidget(self.launcher_page)
        self._load_project_cases(preferred=project)
        self._load_target_cases(preferred=target)
        self._sync_testcase_controls_state()
        self._set_status("已返回启动器。标注项目如已导出，可直接继续运行或刷新配置。")

    def _show_history_page(self):
        LOGGER.info("show history page")
        self.page_stack.setCurrentWidget(self.history_page)
        self._refresh_history_outputs()

    def _history_record_title(self, record: dict) -> str:
        run_index = record.get("run_index")
        archive_time = str(record.get("archive_time") or "").strip()
        target_case = str(record.get("target_case") or "").strip()
        dir_name = Path(record.get("archive_dir")).name if record.get("archive_dir") else "历史输出"
        run_text = f"第{run_index}次" if run_index not in (None, "") else dir_name
        if target_case:
            run_text = f"{run_text} {target_case}"
        if archive_time:
            run_text = f"{run_text}  {archive_time}"
        return run_text

    def _history_record_status(self, record: dict) -> str:
        parts = []
        exit_code = record.get("exit_code")
        if exit_code not in (None, ""):
            parts.append(f"exit={exit_code}")
        if record.get("timed_out") is True:
            parts.append("超时")
        if record.get("stream_disconnected") is True:
            parts.append("断流")
        if not parts:
            parts.append("已归档")
        return " / ".join(parts)

    def _set_selected_history_record(self, record: Optional[dict]):
        self.selected_history_record = record
        has_archive = record is not None
        has_batch = self.selected_history_batch_dir is not None
        self.history_open_dir_button.setEnabled(has_archive or has_batch)
        self.history_delete_button.setEnabled(has_batch)
        if not has_archive:
            self.history_summary_edit.clear()
            self.history_output_edit.clear()
            self.history_frame_records = []
            self.history_frame_index = -1
            self._render_history_frame()
            return

        self.history_summary_edit.setPlainText(format_history_record_summary(record))
        self.history_frame_records = load_history_frame_records(record)
        self.history_frame_index = 0 if self.history_frame_records else -1
        self._render_history_frame()
        launcher_output = str(record.get("launcher_output") or "").strip()
        if launcher_output:
            self.history_output_edit.setPlainText(launcher_output)
        else:
            self.history_output_edit.setPlainText("未找到 logs/launcher_output.txt。")

    def _render_history_frame(self):
        frame_count = len(self.history_frame_records)
        has_frame = frame_count > 0 and 0 <= self.history_frame_index < frame_count
        self.history_prev_frame_button.setEnabled(has_frame and self.history_frame_index > 0)
        self.history_next_frame_button.setEnabled(has_frame and self.history_frame_index < frame_count - 1)

        if not has_frame:
            self.history_frame_counter_label.setText("未找到逐帧日志")
            self.history_frame_image_label.setPixmap(QPixmap())
            self.history_frame_image_label.setText("未找到 process_temp_logs/frame_*.jpg")
            self.history_frame_log_edit.setPlainText(
                "未找到逐帧 JSON。请检查本轮 process_temp_logs 是否已正常生成。"
            )
            return

        frame_record = self.history_frame_records[self.history_frame_index]
        image_path = Path(frame_record.get("image_path"))
        self.history_frame_counter_label.setText(
            f"{self.history_frame_index + 1}/{frame_count}  {image_path.name}"
        )
        self.history_frame_log_edit.setPlainText(format_history_frame_details(frame_record))

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.history_frame_image_label.setPixmap(QPixmap())
            self.history_frame_image_label.setText(f"帧图片读取失败：\n{image_path}")
            return

        available_w = max(1, self.history_frame_image_label.width() - 12)
        available_h = max(1, self.history_frame_image_label.height() - 12)
        scaled = pixmap.scaled(
            available_w,
            available_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.history_frame_image_label.setText("")
        self.history_frame_image_label.setPixmap(scaled)

    def _show_previous_history_frame(self):
        if self.history_frame_index <= 0:
            return
        self.history_frame_index -= 1
        self._render_history_frame()

    def _show_next_history_frame(self):
        if self.history_frame_index >= len(self.history_frame_records) - 1:
            return
        self.history_frame_index += 1
        self._render_history_frame()

    def _refresh_history_outputs(self):
        LOGGER.info("refresh_history_outputs")
        history_temp_dir = resolve_history_temp_dir()
        self.history_records = discover_history_outputs(history_temp_dir)
        self.history_tree.clear()
        self.selected_history_batch_dir = None
        self._set_selected_history_record(None)

        if not self.history_records:
            self.history_status_label.setText(f"未发现历史归档：{history_temp_dir}")
            self.history_summary_edit.setPlainText(
                f"未发现历史输出归档。\n\n运行完成后，launcher 会把每轮产物归档到：\n{history_temp_dir}"
            )
            return

        group_items: dict[str, QTreeWidgetItem] = {}
        for index, record in enumerate(self.history_records):
            batch_dir = Path(record["batch_dir"])
            batch_key = str(batch_dir)
            group_item = group_items.get(batch_key)
            if group_item is None:
                group_item = QTreeWidgetItem(self.history_tree)
                group_item.setText(0, batch_dir.name)
                group_item.setText(1, "批次")
                group_item.setText(2, str(batch_dir))
                group_item.setData(0, Qt.ItemDataRole.UserRole, None)
                group_items[batch_key] = group_item

            child = QTreeWidgetItem(group_item)
            child.setText(0, self._history_record_title(record))
            child.setText(1, self._history_record_status(record))
            child.setText(2, str(record["archive_dir"]))
            child.setData(0, Qt.ItemDataRole.UserRole, index)

        self.history_tree.expandAll()
        for column in range(3):
            self.history_tree.resizeColumnToContents(column)

        self.history_status_label.setText(f"发现 {len(self.history_records)} 条历史输出：{history_temp_dir}")
        first_group = self.history_tree.topLevelItem(0)
        if first_group and first_group.childCount() > 0:
            self.history_tree.setCurrentItem(first_group.child(0))

    def _on_history_selection_changed(self):
        items = self.history_tree.selectedItems()
        if not items:
            self.selected_history_batch_dir = None
            self._set_selected_history_record(None)
            return

        index = items[0].data(0, Qt.ItemDataRole.UserRole)
        if index is None:
            self.selected_history_batch_dir = Path(items[0].text(2))
            self._set_selected_history_record(None)
            return
        try:
            record = self.history_records[int(index)]
        except (IndexError, TypeError, ValueError):
            record = None
        self.selected_history_batch_dir = (
            Path(record["batch_dir"]) if record is not None else None
        )
        self._set_selected_history_record(record)

    def _open_selected_history_dir(self):
        if self.selected_history_record:
            history_dir = Path(self.selected_history_record["archive_dir"])
        elif self.selected_history_batch_dir:
            history_dir = self.selected_history_batch_dir
        else:
            return
        if not history_dir.exists():
            issues = ValidationIssues()
            issues.add_error("目录不存在", f"历史输出目录不存在：{history_dir}")
            self._show_validation_issues("无法打开历史输出", issues)
            self._refresh_history_outputs()
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(history_dir.resolve())))

    def _delete_selected_history_output(self):
        if not self.selected_history_batch_dir:
            return
        batch_dir = self.selected_history_batch_dir.resolve()
        history_temp_dir = resolve_history_temp_dir().resolve()
        try:
            batch_dir.relative_to(history_temp_dir)
        except ValueError:
            QMessageBox.warning(self, "拒绝删除", f"只能删除 temp 目录下的历史输出：\n{batch_dir}")
            return
        if batch_dir == history_temp_dir:
            QMessageBox.warning(self, "拒绝删除", "不能删除历史输出的根目录。")
            return

        reply = QMessageBox.question(
            self,
            "确认删除",
            "确定删除这个日期_用例名目录及其中所有用例信息吗？\n\n"
            f"{batch_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            shutil.rmtree(batch_dir)
        except Exception as exc:
            log_exception(f"delete history output failed: batch_dir={batch_dir}")
            QMessageBox.critical(self, "删除失败", f"删除失败：\n{exc}")
            return

        self._set_status(f"已删除历史输出批次：{batch_dir.name}")
        self._refresh_history_outputs()

    def _apply_parsed_testcase(self, py_file: Path):
        LOGGER.info("apply_parsed_testcase: %s", py_file)
        try:
            parsed = parse_case_vars(py_file)
        except Exception as exc:
            log_exception(f"apply_parsed_testcase failed: file={py_file}")
            self.selected_testcase_description = ""
            self._set_case_info(f"用例解析失败：{exc}")
            self._set_status(f"解析失败：{exc}")
            return

        project_case = parsed.get("project_case")
        target_case = parsed.get("target_case")
        self.selected_testcase_description = str(
            parsed.get("testcase_description") or ""
        ).strip()

        messages = []
        if project_case:
            self._load_project_cases(preferred=project_case)
            messages.append(f"解析到 project_case={project_case}")
        else:
            messages.append("未解析到 project_case，请手动选择")

        if project_case:
            self._load_target_cases(preferred=target_case)
        elif target_case:
            self._set_combo_value(self.target_combo, target_case)

        if target_case:
            self._set_combo_value(self.target_combo, target_case)
            messages.append(f"解析到 target_case={target_case}")
        else:
            messages.append("未解析到 target_case，请手动选择")

        if is_pubg_testcase_file(py_file, parsed):
            self.case_loop_count_spin.setValue(PUBG_CASE_DEFAULT_LOOP_COUNT)

        self._sync_testcase_controls_state()
        self._refresh_case_info_display()
        self._set_status("；".join(messages))

    def _build_process_environment(self, project_case: str, target_case: str, run_no: int) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        apply_pyinstaller_splash_suppression(env)
        env.insert("TARGET_PROJECT_CASE", project_case)
        env.insert("TARGET_GAME_CASE", target_case)
        env.insert("AUTOGAME_VIS_MODE", "launcher")
        env.insert("AUTOGAME_RUN_SOURCE", "launcher")
        env.insert("AUTOGAME_RUN_INDEX", str(int(run_no)))
        env.insert("AUTOGAME_HDC_DEBUG_LEVEL", str(self.hdc_debug_level))
        env.insert(
            "AUTOGAME_EXIT_ON_STREAM_DISCONNECT",
            "1" if self._stream_disconnect_recovery_enabled() else "0",
        )
        for key, value in build_launcher_plan_env_values(self.current_plan).items():
            env.insert(key, value)
        if self.dismiss_reboot_prompt_on_next_case_start:
            env.insert(DISMISS_REBOOT_PROMPT_ENV, "1")
        if self.current_batch_start_timestamp:
            env.insert("AUTOGAME_BATCH_START_TIMESTAMP", self.current_batch_start_timestamp)
        if self.current_run_start_timestamp:
            env.insert("AUTOGAME_RUN_START_TIMESTAMP", self.current_run_start_timestamp)
        archive_metadata = {}
        if target_case:
            archive_metadata["target_case"] = target_case
        if self.current_batch_start_timestamp:
            archive_metadata["batch_start_timestamp"] = self.current_batch_start_timestamp
        if self.current_run_start_timestamp:
            archive_metadata["run_start_timestamp"] = self.current_run_start_timestamp
        try:
            run_archive_dir = resolve_run_archive_dir(
                int(run_no),
                extra_metadata=archive_metadata,
                create=True,
            )
            self.current_run_archive_dir = run_archive_dir
            run_preview_dir = run_archive_dir / "process_temp_logs"
            run_preview_dir.mkdir(parents=True, exist_ok=True)
            env.insert("AUTOGAME_LOG_DIR", str(run_archive_dir))
            env.insert("AUTOGAME_PREVIEW_DIR", str(run_preview_dir))
            env.insert("AUTOGAME_RUN_ARCHIVE_DIR", str(run_archive_dir))
            env.insert("AUTOGAME_BATCH_ARCHIVE_DIR", str(run_archive_dir.parent))
            env.insert(
                "AUTOGAME_DEVICE_LOG_PATH",
                str(run_archive_dir / "hilog.txt"),
            )
            env.insert(
                "AUTOGAME_MEMORY_LOG_PATH",
                str(run_archive_dir / "memory.log"),
            )
            if (
                self.current_hilog_capture is not None
                and not self.current_hilog_capture.start_error
            ):
                env.insert("AUTOGAME_DEVICE_LOG_OWNER", "launcher")
        except Exception:
            log_exception(f"resolve run archive dir failed: run_no={run_no}")
        if self.current_plan is not None:
            env.insert(
                "AUTOGAME_LAUNCHER_INACTIVITY_TIMEOUT_MINUTES",
                str(float(self.current_plan.get("inactivity_timeout_minutes", 5.0))),
            )
            env.insert(
                "POWER_COLLECTION_DURATION_SECONDS",
                str(float(self.current_plan.get("power_collection_duration_seconds"))),
            )
        LOGGER.debug(
            "build_process_environment: project_case=%s target_case=%s run_no=%s batch_start=%s run_start=%s inactivity_timeout=%s power_collection_duration=%s",
            project_case,
            target_case,
            run_no,
            self.current_batch_start_timestamp,
            self.current_run_start_timestamp,
            self.current_plan.get("inactivity_timeout_minutes") if self.current_plan else None,
            self.current_plan.get("power_collection_duration_seconds") if self.current_plan else None,
        )
        return env

    def _set_preview_render_screen_size(self, width, height, source: str):
        width = _positive_int(width)
        height = _positive_int(height)
        if not width or not height:
            return False
        self.preview_render_screen_size = (width, height)
        LOGGER.info(
            "preview render screen size locked: %sx%s source=%s",
            width,
            height,
            source,
        )
        return True

    def _lock_preview_render_screen_size_for_plan(self, plan: dict) -> tuple[Optional[int], Optional[int]]:
        plan = plan if isinstance(plan, dict) else {}
        width = _positive_int(plan.get("screen_width"))
        height = _positive_int(plan.get("screen_height"))
        if width and height:
            self._set_preview_render_screen_size(width, height, "plan")
            return width, height

        try:
            width, height = get_resolution()
        except Exception:
            log_exception("lock preview render screen size failed")
            width, height = None, None

        if self._set_preview_render_screen_size(width, height, "startup"):
            plan["screen_width"] = int(width)
            plan["screen_height"] = int(height)
            return int(width), int(height)

        self.preview_render_screen_size = None
        return None, None

    def _get_preview_render_screen_size(self, payload) -> tuple[Optional[int], Optional[int]]:
        return resolve_preview_render_screen_size(
            payload,
            self.latest_preview_pixmap,
            self.preview_render_screen_size,
        )

    def _set_inputs_enabled(self, enabled: bool):
        self.inputs_enabled = enabled
        self.mode_testcase.setEnabled(enabled)
        self.mode_direct.setEnabled(enabled)
        self.refresh_button.setEnabled(enabled)
        self.open_history_button.setEnabled(enabled)
        self.stream_verify_button.setEnabled(enabled or self.stream_verify_active)
        self.hos_frame_rate_combo.setEnabled(enabled and not self.stream_verify_active)
        self.project_combo.setEnabled(enabled)
        self.target_combo.setEnabled(enabled)
        self.run_count_spin.setEnabled(enabled)
        self.test_profile_field.setEnabled(enabled)
        self.case_loop_count_spin.setEnabled(enabled)
        self.safe_temp_spin.setEnabled(enabled)
        self.safe_battery_spin.setEnabled(enabled)
        self.safe_time_spin.setEnabled(enabled)
        self.inactivity_timeout_spin.setEnabled(enabled)
        self.power_collection_duration_spin.setEnabled(enabled)
        self.marathon_duration_spin.setEnabled(enabled)
        self.marathon_end_battery_spin.setEnabled(enabled)
        self.generate_preview_video_button.setEnabled(enabled)
        self._sync_test_profile_ui()
        for button in self.preset_buttons:
            button.setEnabled(enabled)
        self._sync_testcase_controls_state()

    def _current_preview_dir(self) -> Optional[Path]:
        if self.current_run_archive_dir is not None:
            return self.current_run_archive_dir / "process_temp_logs"
        return None

    def _clear_preview_files(self):
        preview_dir = self._current_preview_dir()
        LOGGER.info("preview frame dir=%s", preview_dir)
        LOGGER.debug("clear_preview_files: dir=%s", preview_dir)
        self.latest_preview_file = None
        self.latest_preview_pixmap = None
        self.latest_preview_payload = None
        self.preview_info_stage_name = None
        self.preview_info_items.clear()
        self.preview_info_item_types.clear()
        self.preview_info_result_keys.clear()
        self.preview_info_template_paths.clear()
        self.preview_image_label.setText("启动后将在这里实时显示可视化帧")
        self.preview_image_label.setPixmap(QPixmap())
        self.preview_info_stage_label.setText("当前阶段：等待运行")
        self.preview_info_tree.clear()
        self._reset_preview_info_template("点击特殊区域或区域查看模板图片")
        self._reset_preview_info_detail(
            "选中区域、特殊区域或控点后查看当前帧完整信息"
        )

        if preview_dir is None:
            LOGGER.debug("skip preview directory cleanup: run archive is not ready")
            return

        preview_dir.mkdir(parents=True, exist_ok=True)
        for path in preview_dir.iterdir():
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    LOGGER.warning("failed to unlink preview file: %s", path, exc_info=True)
                    pass
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    def _adjust_preview_splitter_sizes(self, force: bool = False):
        if self._adjusting_preview_splitter:
            return

        total_width = self.preview_splitter.size().width()
        total_height = self.preview_splitter.size().height()
        if total_width <= 0 or total_height <= 0:
            return

        desired_orientation = (
            Qt.Orientation.Horizontal
            if total_width >= total_height * 1.35
            else Qt.Orientation.Vertical
        )
        if self.preview_splitter.orientation() != desired_orientation:
            self.preview_splitter.setOrientation(desired_orientation)
            force = True

        handle_height = self.preview_splitter.handleWidth()
        current_sizes = self.preview_splitter.sizes()
        if len(current_sizes) != 2:
            current_sizes = [max(0, total_width - self.preview_target_info_width), self.preview_target_info_width]

        if desired_orientation == Qt.Orientation.Horizontal:
            available_width = max(0, total_width - handle_height)
            if available_width <= 0:
                return

            min_info_width = max(self.preview_info_tree.minimumWidth(), 320)
            preferred_info_width = max(min_info_width, self.preview_target_info_width)
            current_preview_width = max(0, current_sizes[0])
            current_info_width = max(0, current_sizes[1])
            max_preview_width = max(0, available_width - preferred_info_width)

            if force:
                target_preview_width = max_preview_width
            else:
                target_preview_width = current_preview_width
                if current_info_width < preferred_info_width:
                    target_preview_width = max_preview_width
                elif current_preview_width > max_preview_width + 40:
                    target_preview_width = max_preview_width
                else:
                    return

            target_info_width = max(min_info_width, available_width - target_preview_width)
            target_preview_width = max(0, available_width - target_info_width)
            target_sizes = [target_preview_width, target_info_width]
        else:
            available_height = max(0, total_height - handle_height)
            if available_height <= 0:
                return

            min_info_height = max(self.preview_info_tree.minimumHeight(), 150)
            preferred_info_height = max(min_info_height, self.preview_target_info_height)
            current_preview_height = max(0, current_sizes[0])
            current_info_height = max(0, current_sizes[1])
            max_preview_height = max(0, available_height - preferred_info_height)

            if force:
                target_preview_height = max_preview_height
            else:
                target_preview_height = current_preview_height
                if current_info_height < preferred_info_height:
                    target_preview_height = max_preview_height
                elif current_preview_height > max_preview_height + 40:
                    target_preview_height = max_preview_height
                else:
                    return

            target_info_height = max(min_info_height, available_height - target_preview_height)
            target_preview_height = max(0, available_height - target_info_height)
            target_sizes = [target_preview_height, target_info_height]

        self._adjusting_preview_splitter = True
        try:
            self.preview_splitter.setSizes(target_sizes)
        finally:
            self._adjusting_preview_splitter = False

    def _refresh_preview_pixmap(self) -> bool:
        if self.latest_preview_pixmap is None:
            LOGGER.debug("render preview fail: no latest preview pixmap")
            return False
        try:
            display_pixmap = self._build_preview_display_pixmap()
        except Exception:
            log_exception("build preview overlay failed; fallback to raw pixmap")
            display_pixmap = self.latest_preview_pixmap
        if display_pixmap.isNull():
            LOGGER.warning(
                "render preview fail: display pixmap is null latest_frame=%s",
                self.latest_preview_file,
            )
            return False
        self._adjust_preview_splitter_sizes()
        preview_targets = [self.preview_image_label]
        if hasattr(self, "preview_fullscreen_image_label"):
            preview_targets.append(self.preview_fullscreen_image_label)
        rendered = False
        for target in preview_targets:
            rendered = self._render_preview_pixmap_for_label(display_pixmap, target) or rendered
        return rendered

    def _render_preview_pixmap_for_label(self, display_pixmap: QPixmap, target: QLabel) -> bool:
        scaled = display_pixmap.scaled(
            target.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if scaled.isNull():
            LOGGER.warning(
                "render preview fail: scaled pixmap is null latest_frame=%s label=%s label_size=%s",
                self.latest_preview_file,
                target.objectName(),
                target.size(),
            )
            return False
        target.setText("")
        target.setPixmap(scaled)
        LOGGER.debug(
            "render preview success: latest_frame=%s label=%s label_size=%s scaled_size=%s",
            self.latest_preview_file,
            target.objectName(),
            target.size(),
            scaled.size(),
        )
        return True

    def _get_preview_project_case(self) -> str:
        if self.current_plan is not None:
            return str(self.current_plan.get("project_case") or "").strip()
        return self.project_combo.currentText().strip()

    def _load_preview_project_info(self, project_case: str) -> dict:
        if not project_case:
            return {}
        if project_case in self.preview_info_cache:
            return self.preview_info_cache[project_case]

        try:
            importlib.invalidate_caches()
            module_name = f"aw.autogame.customs_examples.{project_case}.info"
            if module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
            else:
                module = importlib.import_module(module_name)
            stage_info = getattr(module, "STAGE_INFO", {})
            scene_pool = getattr(module, "SCENE_POOL", {})
        except Exception:
            log_exception(f"load preview project info failed: project_case={project_case}")
            stage_info = {}
            scene_pool = {}

        if not isinstance(stage_info, dict):
            stage_info = {}
        if not isinstance(scene_pool, dict):
            scene_pool = {}
        project_info = {
            "stage_info": stage_info,
            "scene_pool": scene_pool,
        }
        self.preview_info_cache[project_case] = project_info
        self.stage_info_cache[project_case] = stage_info
        return project_info

    def _load_stage_info(self, project_case: str) -> dict:
        return self._load_preview_project_info(project_case).get("stage_info", {})

    def _rebuild_preview_info_list(self, stage_name: str, payload):
        self.preview_info_tree.clear()
        self.preview_info_items.clear()
        self.preview_info_item_types.clear()
        self.preview_info_result_keys.clear()
        self.preview_info_template_paths.clear()
        self._reset_preview_info_template("点击特殊区域或区域查看模板图片")
        self._reset_preview_info_detail(
            "选中区域、特殊区域或控点后查看当前帧完整信息"
        )
        self.preview_info_stage_name = stage_name

        project_info = self._load_preview_project_info(self._get_preview_project_case())
        stage_info = project_info.get("stage_info", {})
        scene_pool = project_info.get("scene_pool", {})
        stage_entry = stage_info.get(stage_name, {}) if isinstance(stage_info, dict) else {}
        screen_width, screen_height = self._get_preview_render_screen_size(payload)
        entries = resolve_preview_stage_info_entries(
            stage_entry,
            scene_pool,
            screen_width,
            screen_height,
        )

        info_payload = payload.get("info") if isinstance(payload, dict) else None
        info_payload = info_payload if isinstance(info_payload, dict) else {}
        known_info_keys = {
            entry["info_key"] for entry in entries if entry["info_key"]
        }
        unknown_info_keys = [
            str(info_key)
            for info_key in info_payload
            if str(info_key) not in known_info_keys
        ]
        if unknown_info_keys:
            LOGGER.warning(
                "preview info keys absent from current stage config; stage=%s keys=%s",
                stage_name,
                unknown_info_keys,
            )

        for entry in entries:
            item = QTreeWidgetItem(self.preview_info_tree)
            item.setText(0, entry["type"])
            item.setText(1, entry["name"])
            item.setText(2, "—")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
            item.setData(0, Qt.ItemDataRole.UserRole, entry["key"])
            self.preview_info_items[entry["key"]] = item
            self.preview_info_item_types[entry["key"]] = entry["item_type"]
            self.preview_info_result_keys[entry["key"]] = entry["info_key"]
            self.preview_info_template_paths[entry["key"]] = entry["template"]

        self._sync_preview_master_buttons_from_checks()

        if not entries:
            empty_item = QTreeWidgetItem(self.preview_info_tree)
            empty_item.setText(0, "阶段信息")
            empty_item.setText(1, stage_name or "未识别阶段")
            empty_item.setText(2, "该阶段未配置区域或特殊区域")

    def _update_preview_info_list(self, payload):
        payload = payload if isinstance(payload, dict) else {"raw": payload}
        stage_name = resolve_preview_payload_stage_name(payload)
        group_name = resolve_preview_payload_group_name(payload)
        stage_changed = stage_name != self.preview_info_stage_name

        vertical_scroll = self.preview_info_tree.verticalScrollBar()
        horizontal_scroll = self.preview_info_tree.horizontalScrollBar()
        vertical_value = vertical_scroll.value()
        horizontal_value = horizontal_scroll.value()

        previous_signals_blocked = self.preview_info_tree.blockSignals(True)
        self.preview_info_tree.setUpdatesEnabled(False)
        try:
            if stage_changed:
                self._rebuild_preview_info_list(stage_name, payload)

            stage_text = stage_name or "未识别"
            group_text = group_name or "默认"
            self.preview_info_stage_label.setText(
                f"当前阶段：{stage_text}    分组：{group_text}"
            )

            info_payload = payload.get("info")
            info_payload = info_payload if isinstance(info_payload, dict) else {}
            for row_key, item in self.preview_info_items.items():
                info_key = self.preview_info_result_keys.get(row_key, row_key)
                if info_key in info_payload:
                    value = info_payload[info_key]
                    display_text = format_preview_info_item_value(value)
                    tooltip_text = format_preview_info_item_value(value, max_length=2000)
                else:
                    display_text = "—"
                    tooltip_text = "当前帧无该项信息"
                if item.text(2) != display_text:
                    item.setText(2, display_text)
                item.setToolTip(2, tooltip_text)
        finally:
            self.preview_info_tree.setUpdatesEnabled(True)
            self.preview_info_tree.blockSignals(previous_signals_blocked)

        if not stage_changed:
            vertical_scroll.setValue(vertical_value)
            horizontal_scroll.setValue(horizontal_value)
        self._refresh_preview_info_detail(payload)

    def _is_preview_info_item_checked(
        self,
        scene_name: str,
        item_type: str,
        item_name: str,
    ) -> bool:
        info_key = f"{scene_name}__{item_name}"
        row_key = f"{item_type}::{info_key}"
        item = self.preview_info_items.get(row_key)
        if item is None:
            return True
        if self.preview_info_item_types.get(row_key) != item_type:
            return True
        return item.checkState(0) == Qt.CheckState.Checked

    def _draw_stage_rect(
        self,
        painter: QPainter,
        area_config,
        pixmap_width: int,
        pixmap_height: int,
        origin_width: int,
        origin_height: int,
        screen_width: Optional[int],
        screen_height: Optional[int],
        color: QColor,
        label: str,
        pen_style=None,
        fill_alpha: int = 35,
        label_offset_y: int = 16,
    ):
        try:
            if isinstance(area_config, dict):
                x1, y1, x2, y2 = resolve_area_rect_for_frame(
                    pixmap_width,
                    pixmap_height,
                    area_config,
                    screen_width,
                    screen_height,
                    origin_width,
                    origin_height,
                )
            elif isinstance(area_config, (list, tuple)) and len(area_config) == 4:
                x1, y1, x2, y2 = resolve_area_rect_for_frame(
                    pixmap_width,
                    pixmap_height,
                    {"rect": area_config},
                    None,
                    None,
                    origin_width,
                    origin_height,
                )
            else:
                return
        except Exception:
            return

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        pen = QPen(color, 2)
        if pen_style is not None:
            pen.setStyle(pen_style)
        painter.setPen(pen)
        painter.drawRect(x1, y1, width, height)
        if fill_alpha > 0:
            painter.fillRect(
                x1,
                y1,
                width,
                height,
                QColor(color.red(), color.green(), color.blue(), fill_alpha),
            )
        painter.drawText(x1 + 4, max(14, y1 + label_offset_y), label)

    def _build_preview_display_pixmap(self) -> QPixmap:
        if self.latest_preview_pixmap is None:
            return QPixmap()
        show_overlay = self.preview_overlay_button.isChecked()
        show_points = self.preview_points_button.isChecked()
        if not show_overlay and not show_points:
            return self.latest_preview_pixmap

        payload = self.latest_preview_payload or {}
        stage = resolve_preview_payload_stage_name(payload)
        project_case = self._get_preview_project_case()
        project_info = self._load_preview_project_info(project_case)
        stage_info = project_info.get("stage_info", {})
        scene_pool = project_info.get("scene_pool", {})
        stage_entry = stage_info.get(stage, {}) if isinstance(stage_info, dict) else {}
        scenes = resolve_preview_stage_scenes(stage_entry, scene_pool)
        if not scenes:
            return self.latest_preview_pixmap

        pixmap = self.latest_preview_pixmap.copy()
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            screen_width, screen_height = self._get_preview_render_screen_size(payload)

            colors = {}
            if show_overlay:
                colors["areas"] = QColor(80, 220, 120)
                colors["special_areas"] = QColor(255, 140, 80)
            if show_points:
                colors["points"] = QColor(80, 190, 255)

            for scene_name, scene_data in scenes.items():
                if not isinstance(scene_data, dict):
                    continue
                scene_data = select_scene_resolution(scene_data, screen_width, screen_height)
                for item_type, color in colors.items():
                    items = scene_data.get(item_type, {})
                    if not isinstance(items, dict):
                        continue
                    for item_name, item_data in items.items():
                        if not isinstance(item_data, dict):
                            continue
                        if not self._is_preview_info_item_checked(
                            scene_name,
                            item_type,
                            item_name,
                        ):
                            continue
                        label = f"{scene_name}/{item_name}"
                        self._draw_stage_rect(
                            painter,
                            item_data,
                            pixmap.width(),
                            pixmap.height(),
                            int(scene_data.get("width") or pixmap.width()),
                            int(scene_data.get("height") or pixmap.height()),
                            screen_width,
                            screen_height,
                            color,
                            label,
                        )
                        if item_type == "areas":
                            search_scope = resolve_preview_area_search_scope(item_data)
                            if search_scope is not None:
                                self._draw_stage_rect(
                                    painter,
                                    search_scope,
                                    pixmap.width(),
                                    pixmap.height(),
                                    int(scene_data.get("width") or pixmap.width()),
                                    int(scene_data.get("height") or pixmap.height()),
                                    screen_width,
                                    screen_height,
                                    QColor(255, 220, 80),
                                    f"搜索范围:{label}",
                                    pen_style=Qt.PenStyle.DashLine,
                                    fill_alpha=8,
                                    label_offset_y=34,
                                )
        finally:
            painter.end()
        return pixmap

    def _poll_preview_frame(self):
        preview_dir = self._current_preview_dir()
        if preview_dir is None or not preview_dir.exists():
            return

        latest_image = find_latest_preview_frame(preview_dir)
        if latest_image is None or latest_image == self.latest_preview_file:
            return

        json_path = latest_image.with_suffix(".json")
        pixmap = QPixmap(str(latest_image))
        if pixmap.isNull():
            LOGGER.warning("QPixmap load fail: latest_frame=%s", latest_image)
            return

        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                log_exception(f"preview json load failed: {json_path}")
                payload = {"error": "json 读取失败", "frame": latest_image.name}
        else:
            payload = {"frame": latest_image.name, "preview_json": "missing"}

        self.latest_preview_file = latest_image
        self.latest_preview_pixmap = pixmap
        self.latest_preview_payload = payload if isinstance(payload, dict) else {"raw": payload}
        self.preview_image_label.setText("")
        self._update_preview_info_list(payload)
        self._adjust_preview_splitter_sizes()
        render_success = self._refresh_preview_pixmap()
        LOGGER.info(
            "render preview %s: latest_frame=%s",
            "success" if render_success else "fail",
            latest_image,
        )

    def _validate_selection(self, issues: ValidationIssues) -> Optional[tuple[str, str]]:
        project_case = self.project_combo.currentText().strip()
        target_case = self.target_combo.currentText().strip()
        LOGGER.info(
            "validate_selection: mode=%s project_case=%s target_case=%s testcase=%s",
            "testcase" if self.mode_testcase.isChecked() else "direct",
            project_case,
            target_case,
            self.selected_testcase_file,
        )

        if not project_case:
            issues.add_error("缺少配置", "请选择 project_case。")
        if not target_case:
            issues.add_error("缺少配置", "请选择 target_case。")
        if issues.has_errors():
            return None

        return project_case, target_case

    def _collect_plan(self, issues: ValidationIssues) -> Optional[dict]:
        config = self._validate_selection(issues)
        testcase_label = None
        mode = "direct"

        if self.mode_testcase.isChecked():
            if self.selected_testcase_file is None:
                issues.add_error("缺少用例", "testcases 模式下请先选择一个用例文件。")
            else:
                try:
                    testcase_label = self.selected_testcase_file.relative_to(APP_DIR).with_suffix("").as_posix()
                    mode = "testcase"
                except ValueError:
                    issues.add_error("路径错误", "所选用例不在当前项目目录内，请重新选择。")

        if config is None or issues.has_errors():
            return None

        project_case, target_case = config

        cleanup_apps = set()
        if self.selected_testcase_file is not None:
            cleanup_apps.update(extract_package_names(self.selected_testcase_file))

        target_logic_file = (
            CUSTOMS_GAME_EXAMPLES_DIR / project_case / f"{target_case}.py"
        )
        cleanup_apps.update(extract_package_names(target_logic_file))

        test_profile = resolve_test_profile_from_radio_selection(
            self.power_test_radio.isChecked(),
            self.function_test_radio.isChecked(),
            self.marathon_test_radio.isChecked(),
        )
        marathon_selected = test_profile == TEST_PROFILE_MARATHON
        marathon_duration_minutes = (
            float(self.marathon_duration_spin.value()) if marathon_selected else 0.0
        )
        marathon_end_battery_percent = (
            int(self.marathon_end_battery_spin.value()) if marathon_selected else 0
        )
        if marathon_selected and marathon_duration_minutes <= 0:
            issues.add_error("马拉松时长无效", "马拉松模式的 SP 运行时长必须大于 0 分钟。")
            return None
        if marathon_selected:
            cleanup_apps.update((DEFAULT_PUBG_GAME_PACKAGE, DEFAULT_SP_PACKAGE))
        if not should_use_sp_recording_for_profile(test_profile):
            cleanup_apps.discard(DEFAULT_SP_PACKAGE)
        try:
            screen_mode = resolve_screen_mode_for_test_profile(test_profile, target_case)
        except Exception as exc:
            issues.add_error("截图模式配置错误", f"读取 config.json 的 screen_mode 失败：{exc}")
            return None
        runtime_description = ""
        if is_pubg_testcase_target_case(target_case):
            runtime_description = PUBG_CASE_RUNTIME_DESCRIPTION
        plan = {
            "mode": mode,
            "project_case": project_case,
            "target_case": target_case,
            "testcase_label": testcase_label,
            "run_count": int(self.run_count_spin.value()),
            "test_profile": test_profile,
            "screen_mode": screen_mode,
            "case_loop_count": int(self.case_loop_count_spin.value()),
            "safe_temp": float(self.safe_temp_spin.value()),
            "safe_battery": int(self.safe_battery_spin.value()),
            "safe_minutes": float(self.safe_time_spin.value()),
            "inactivity_timeout_minutes": (
                0.0
                if marathon_selected
                else float(self.inactivity_timeout_spin.value())
            ),
            "power_collection_duration_seconds": float(self.power_collection_duration_spin.value()),
            "marathon_duration_minutes": marathon_duration_minutes,
            "marathon_end_battery_percent": marathon_end_battery_percent,
            "generate_preview_video": bool(self.generate_preview_video_button.isChecked()),
            "preserve_game_process": (
                False
                if marathon_selected
                else self.game_process_policy_button.isChecked()
            ),
            "cleanup_apps": sorted(cleanup_apps),
            "runtime_description": runtime_description,
            "testcase_description": (
                self.selected_testcase_description
                if mode == "testcase"
                else f"直接启动 project_case={project_case}，target_case={target_case}"
            ),
        }
        LOGGER.info("collect_plan result: %s", plan)
        return plan

    def _format_runtime_text(
        self,
        run_index: int,
        total_runs: int,
        temperature: Optional[float],
        battery: Optional[int],
        extra: str,
    ) -> str:
        temp_text = "未知" if temperature is None else f"{temperature:.1f}°C"
        battery_text = "未知" if battery is None else f"{battery}%"
        return f"运行信息：第 {run_index}/{total_runs} 次，温度 {temp_text}，电量 {battery_text}。{extra}"

    def _begin_batch(self, plan: dict):
        LOGGER.info("begin_batch: %s", plan)
        self.current_plan = plan
        screen_width, screen_height = self._lock_preview_render_screen_size_for_plan(plan)
        self.current_batch_start_timestamp = time.strftime("%Y%m%d%H%M%S")
        self.output_log_spool_path = None
        self.current_run_start_timestamp = None
        self.batch_active = True
        self.stop_requested = False
        self.manual_stop_requested = False
        self.current_run_index = 0
        self.current_run_timed_out = False
        self.output_log_entries.clear()
        self.output_edit.clear()
        self._clear_preview_files()
        self.start_button.setEnabled(False)
        self.stream_verify_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_inputs_enabled(False)
        marathon_minutes = float(plan.get("marathon_duration_minutes") or 0.0)
        marathon_end_battery = int(plan.get("marathon_end_battery_percent") or 0)
        if marathon_minutes > 0:
            self._set_status("已开始连续马拉松；每轮保存后关闭游戏/SP，再启动下一轮。")
            self._set_runtime(
                f"运行信息：每轮 SP 有效时长 {marathon_minutes:g} 分钟，等待第 1 轮启动。"
            )
        else:
            self._set_status("已开始批量执行，准备进行安全检查。")
            self._set_runtime(f"运行信息：共 {plan['run_count']} 次，等待第 1 次启动。")
        run_limit_text = "continuous_until_battery_cutoff" if marathon_minutes > 0 else plan["run_count"]
        self._log_message(
            f"[Launcher] 批量运行开始，mode={plan['mode']}, runs={run_limit_text}, "
            f"test_profile={plan['test_profile']}, screen_mode={plan['screen_mode']}, "
            f"screen_size={screen_width}x{screen_height}, "
            f"case_loops={plan['case_loop_count']}, "
            f"generate_preview_video={plan['generate_preview_video']}, "
            f"preserve_game_process={plan['preserve_game_process']}, "
            f"safe_temp={plan['safe_temp']}°C, safe_battery={plan['safe_battery']}%, "
            f"safe_time={plan['safe_minutes']}分钟, inactivity_timeout={plan['inactivity_timeout_minutes']}分钟, "
            f"marathon_duration={marathon_minutes}分钟, "
            f"marathon_end_battery={marathon_end_battery}%, "
            f"power_collection_duration={plan['power_collection_duration_seconds']}秒, "
            f"cleanup_apps={plan['cleanup_apps']}\n"
        )
        if plan.get("runtime_description"):
            self._log_message(f"[Launcher] {plan['runtime_description']}\n")
        if plan.get("testcase_description"):
            self._log_message(
                f"[Launcher] 用例信息：{plan['testcase_description']}\n"
            )
        if marathon_minutes > 0:
            self._log_message(
                "[Launcher] 马拉松模式已启用：运行期间不做温度/电量门禁检查，"
                "每轮仅在 SP 有效时间达标或电量到达结束阈值时长按保存；"
                "正常达标后关闭游戏/SP 进程并自动开始下一轮。\n"
            )
            self._log_message(
                f"[Launcher] 马拉松电量监控：结束电量="
                f"{marathon_end_battery if marathon_end_battery > 0 else '关闭'}，"
                "battery.log 写入本批次各次运行目录的外层。\n"
            )
        if plan.get("capture_preflight_message"):
            self._log_message(f"[Launcher] 截图流预检：{plan['capture_preflight_message']}\n")
        self._cleanup_apps_between_runs("批次启动前预清理")
        self._check_and_start_if_safe()

    def _finish_batch(self, message: str):
        LOGGER.info("finish_batch: %s", message)
        self._stop_recovery_processes()
        self._stop_current_memory_capture()
        self._stop_current_hilog_capture()
        self._stop_current_hdc_debug_capture()
        restore_hiz = bool(
            self.stop_requested
            or self.current_run_timed_out
            or getattr(self, "_close_after_stop", False)
        )
        trace_path = self.process_launch_tracer.stop()
        if trace_path is not None:
            LOGGER.info("process launch trace log available: %s", trace_path)
            self._log_message(f"[Launcher] 进程创建追踪已停止：{trace_path}\n")
        self.batch_active = False
        self.stop_requested = False
        self.current_plan = None
        self.current_run_timed_out = False
        self.current_run_output_start = 0
        self.current_run_stream_started = False
        self.current_run_stream_disconnected = False
        self.current_run_stream_disconnect_startup = False
        self.current_run_stream_disconnect_message = ""
        self.current_run_stream_preserved = False
        self.current_run_sp_started = False
        self.current_run_sp_started_monotonic = None
        self.current_run_sp_save_confirmed = False
        self.current_run_sp_state = {}
        self.sp_save_settle_in_progress = False
        self.pending_process_finished = None
        self.current_run_failure_code = ""
        self.current_run_failure_reason = ""
        self.current_run_failure_details = {}
        self.current_run_inactivity_preserved = False
        self.dismiss_reboot_prompt_on_next_case_start = False
        self.current_batch_start_timestamp = None
        self.current_run_start_timestamp = None
        self.current_run_archive_dir = None
        self.start_button.setEnabled(True)
        self.stream_verify_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_inputs_enabled(True)
        self.preview_timer.stop()
        self.safety_timer.stop()
        self.run_timeout_timer.stop()
        self.stream_disconnect_signal_timer.stop()
        self._set_status(message)
        self._set_runtime(message)
        if restore_hiz:
            self._log_message("[Launcher] 异常/手动停止后由 Launcher 兜底关闭 HIZ 并恢复充电。\n")
            set_hiz_mode(False)
        set_launcher_log_file(None)
        self.output_log_spool_path = None
        if getattr(self, "_close_after_stop", False):
            self._close_after_stop = False
            QTimer.singleShot(0, self.close)

    def _cleanup_apps_between_runs(self, reason: str, force: bool = False):
        if self.current_plan is None:
            return

        if should_preserve_game_process_for_plan(self.current_plan) and not force:
            self._log_message(
                f"[Launcher] {reason}：当前测试处于保留进程模式，跳过应用进程清理。\n"
            )
            return

        apps = list(self.current_plan.get("cleanup_apps", []))
        if not apps:
            self._log_message(f"[Launcher] {reason}：未识别到需要强杀的应用，跳过设备清理。\n")
            return

        self._log_message(f"[Launcher] {reason}：开始强制停止残留应用 {apps}\n")
        stopped = force_stop_apps(apps)
        if stopped:
            time.sleep(1.0)
            self._log_message(f"[Launcher] 已执行 force-stop: {stopped}\n")
        else:
            self._log_message("[Launcher] 未成功执行 force-stop，请检查 hdc 环境或设备连接状态。\n", level=logging.WARNING)

    def _check_and_start_if_safe(self):
        LOGGER.info(
            "check_and_start_if_safe: batch_active=%s process_exists=%s stop_requested=%s current_run_index=%s current_plan=%s",
            self.batch_active,
            self.process is not None,
            self.stop_requested,
            self.current_run_index,
            self.current_plan,
        )
        if not self.batch_active or self.current_plan is None:
            return
        if self.process is not None:
            return
        if self.stop_requested:
            self._finish_batch("任务已停止。")
            return
        if has_reached_plan_run_limit(self.current_plan, self.current_run_index):
            self._finish_batch("所有运行次数已完成。")
            return

        run_no = self.current_run_index + 1
        if is_marathon_plan(self.current_plan):
            self.safety_timer.stop()
            self._log_message(
                f"[Launcher] 马拉松第 {run_no} 轮跳过启动前温度和电量检查。\n"
            )
            self._cleanup_apps_between_runs("马拉松启动前清理")
            self._launch_iteration(run_no, None, None)
            return

        temperature = get_battery_temperature_c()
        battery = get_battery_capacity()
        LOGGER.info(
            "safety_check_result: run_no=%s temperature=%s battery=%s thresholds=(temp=%s,battery=%s)",
            run_no,
            temperature,
            battery,
            self.current_plan["safe_temp"],
            self.current_plan["safe_battery"],
        )

        if battery is None or temperature is None:
            retry_message = "无法读取手机温度或电量，稍后重试。"
            self._log_message(
                f"[Launcher] 安全检查：{retry_message} temperature={temperature}, battery={battery}\n",
                level=logging.WARNING,
            )
            self._set_status(retry_message)
            self._set_runtime(
                self._format_runtime_text(run_no, self.current_plan["run_count"], temperature, battery, "等待重试。")
            )
            if not self.safety_timer.isActive():
                self.safety_timer.start()
            return

        if battery < self.current_plan["safe_battery"]:
            set_hiz_mode(False)
            self._set_status(
                f"当前电量 {battery}% 低于安全电量 {self.current_plan['safe_battery']}%，已开启充电并关闭 HIZ，等待后再运行。"
            )
            self._set_runtime(
                self._format_runtime_text(run_no, self.current_plan["run_count"], temperature, battery, "电量不足，等待充电。")
            )
            if not self.safety_timer.isActive():
                self.safety_timer.start()
            return

        if temperature > self.current_plan["safe_temp"]:
            self._set_status(
                f"当前温度 {temperature:.1f}°C 高于安全温度 {self.current_plan['safe_temp']:.1f}°C，等待降温后再运行。"
            )
            self._set_runtime(
                self._format_runtime_text(run_no, self.current_plan["run_count"], temperature, battery, "温度过高，等待降温。")
            )
            if not self.safety_timer.isActive():
                self.safety_timer.start()
            return

        self.safety_timer.stop()
        self._cleanup_apps_between_runs("启动前清理")
        self._launch_iteration(run_no, temperature, battery)

    def _launch_iteration(
        self,
        run_no: int,
        temperature: Optional[float],
        battery: Optional[int],
    ):
        if self.current_plan is None:
            return

        self.current_run_timed_out = False
        self.current_run_stream_started = False
        self.current_run_stream_disconnected = False
        self.current_run_stream_disconnect_startup = False
        self.current_run_stream_disconnect_message = ""
        self.current_run_stream_preserved = False
        self.current_run_sp_started = False
        self.current_run_sp_started_monotonic = None
        self.current_run_sp_save_confirmed = False
        self.current_run_sp_state = {}
        self.sp_save_settle_in_progress = False
        self.pending_process_finished = None
        self.current_run_failure_code = ""
        self.current_run_failure_reason = ""
        self.current_run_failure_details = {}
        self.current_run_inactivity_preserved = False
        self.current_run_start_timestamp = time.strftime("%Y%m%d%H%M%S")
        self.current_run_archive_dir = None
        self.process_output_buffer = ""
        archive_dir = self._resolve_current_run_archive_dir()
        if archive_dir is not None:
            set_launcher_log_file(archive_dir / "launcher_debug.log")
            self._start_output_log_spool(archive_dir)
            self.current_run_output_start = 0
        LOGGER.info(
            "launch_iteration start: run_no=%s temperature=%s battery=%s plan=%s archive_dir=%s",
            run_no,
            temperature,
            battery,
            self.current_plan,
            archive_dir,
        )
        self._stop_current_memory_capture()
        self._stop_current_hilog_capture()
        self._stop_current_hdc_debug_capture()
        if archive_dir is not None:
            self.process_launch_tracer.stop()
            self.process_launch_tracer = WindowsProcessLaunchTracer(archive_dir)
            trace_label = (
                f"{self.current_plan['mode']}:"
                f"{self.current_plan['project_case']}:"
                f"{self.current_plan['target_case']}"
            )
            trace_path = self.process_launch_tracer.start(trace_label)
            if trace_path is not None:
                self._log_message(f"[Launcher] 进程创建追踪日志：{trace_path}\n")
            else:
                self._log_message("[Launcher] 当前环境未启用 Windows 进程创建追踪。\n")
            self.current_memory_capture = MemoryRunCapture(
                archive_dir / "memory.log",
                root_pid=os.getpid(),
            ).start()
            if self.current_memory_capture.start_error:
                self._log_message(
                    "[Launcher] 内存监控启动失败：%s\n"
                    % self.current_memory_capture.start_error,
                    level=logging.ERROR,
                )
            else:
                self._log_message(
                    "[Launcher] 内存监控日志：%s（每5秒采样，每60秒批量写盘）\n"
                    % self.current_memory_capture.path
                )
            self.current_hdc_debug_capture = HdcDebugRunCapture(
                archive_dir / "hdc_debug.log"
            ).start()
            if self.current_hdc_debug_capture.start_error:
                self._log_message(
                    "[Launcher] HDC DEBUG 分轮采集启动失败：%s\n"
                    % self.current_hdc_debug_capture.start_error,
                    level=logging.ERROR,
                )
            else:
                self._log_message(
                    "[Launcher] HDC DEBUG 分轮日志：%s\n"
                    % self.current_hdc_debug_capture.path
                )
            self.current_hilog_capture = HilogRunCapture(
                archive_dir / "hilog.txt"
            ).start()
            if self.current_hilog_capture.start_error:
                self._log_message(
                    "[Launcher] hilog 分轮采集启动失败：%s\n"
                    % self.current_hilog_capture.start_error,
                    level=logging.ERROR,
                )
            else:
                self._log_message(
                    "[Launcher] hilog 分轮日志：%s\n"
                    % self.current_hilog_capture.path
                )
        self._clear_preview_files()
        if archive_dir is None:
            self.current_run_output_start = self._current_output_offset()

        project_case = self.current_plan["project_case"]
        target_case = self.current_plan["target_case"]
        self.process = HiddenSubprocess(self)
        self.process.setProgram(sys.executable)
        self.process.setWorkingDirectory(str(APP_DIR))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.setProcessEnvironment(self._build_process_environment(project_case, target_case, run_no))
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)

        if self.current_plan["mode"] == "testcase":
            testcase_label = self.current_plan["testcase_label"]
            args = build_launcher_process_args("--run-testcase", testcase_label)
            run_label = (
                f"马拉松第 {run_no} 轮"
                if is_marathon_plan(self.current_plan)
                else f"第 {run_no}/{self.current_plan['run_count']} 次"
            )
            self._set_status(f"{run_label}启动：{testcase_label}")
            self._log_message(
                f"\n[Launcher] {run_label}：通过 testcase 启动 {testcase_label}\n"
            )
        else:
            args = build_launcher_process_args("--run-direct", project_case, target_case)
            run_label = (
                f"马拉松第 {run_no} 轮"
                if is_marathon_plan(self.current_plan)
                else f"第 {run_no}/{self.current_plan['run_count']} 次"
            )
            self._set_status(f"{run_label}启动：project_case={project_case}, target_case={target_case}")
            self._log_message(
                f"\n[Launcher] {run_label}：直接启动 "
                f"project_case={project_case}, target_case={target_case}\n"
            )

        if is_marathon_plan(self.current_plan):
            marathon_minutes = float(self.current_plan["marathon_duration_minutes"])
            self._set_runtime(
                f"运行信息：马拉松第 {run_no} 轮，"
                f"SP 有效时长目标 {marathon_minutes:g} 分钟，正在启动。"
            )
        else:
            self._set_runtime(
                self._format_runtime_text(
                    run_no,
                    self.current_plan["run_count"],
                    temperature,
                    battery,
                    "安全检查通过，正在启动。",
                )
            )

        self.process.setArguments(args)
        LOGGER.info(
            "starting child process: program=%s args=%s workdir=%s",
            sys.executable,
            args,
            APP_DIR,
        )
        self.process.start()
        started = self.process.waitForStarted(3000)
        LOGGER.info(
            "child process start result: started=%s state=%s pid=%s error=%s error_string=%s",
            started,
            self.process.state() if self.process is not None else None,
            int(self.process.processId()) if self.process is not None else None,
            self.process.error() if self.process is not None else None,
            self.process.errorString() if self.process is not None else None,
        )
        if not started:
            self._log_message(
                "[Launcher] 子进程启动失败，请检查日志中的 program/args/error 信息。\n",
                level=logging.ERROR,
            )
            issues = ValidationIssues()
            issues.add_error(
                "子进程启动失败",
                "请检查 Python 环境，并查看 launcher 日志中的 program、args 和 error 信息。",
            )
            self._show_validation_issues("任务未启动", issues)
            self.process.deleteLater()
            self.process = None
            self._finish_batch("启动失败，批量任务已终止。")
            return

        if self.dismiss_reboot_prompt_on_next_case_start:
            self._log_message("[Launcher] 已通知本次用例在打开 sp 后关闭重启弹窗。\n")
            self.dismiss_reboot_prompt_on_next_case_start = False

        self.preview_timer.start()
        safe_minutes = self.current_plan["safe_minutes"]
        if safe_minutes > 0 and not is_marathon_plan(self.current_plan):
            self.run_timeout_timer.start(int(safe_minutes * 60 * 1000))
        if self._stream_disconnect_recovery_enabled():
            self.stream_disconnect_signal_timer.start()
        else:
            self.stream_disconnect_signal_timer.stop()

    def _resolve_current_device_log_path(self) -> Optional[Path]:
        if self.current_run_archive_dir is not None:
            return self.current_run_archive_dir / "hilog.txt"
        return None

    def _wait_for_device_log_stable(self, log_path: Path) -> bool:
        deadline = time.time() + DEVICE_LOG_SETTLE_TIMEOUT_SECONDS
        last_size = None
        stable_since = None

        while time.time() < deadline:
            QApplication.processEvents()
            if log_path.exists() and log_path.is_file():
                try:
                    size = log_path.stat().st_size
                except OSError:
                    size = None

                now = time.time()
                if size is not None and size == last_size:
                    if stable_since is None:
                        stable_since = now
                    if now - stable_since >= DEVICE_LOG_SETTLE_INTERVAL_SECONDS:
                        return True
                else:
                    last_size = size
                    stable_since = None

            time.sleep(DEVICE_LOG_SETTLE_INTERVAL_SECONDS)

        return log_path.exists() and log_path.is_file()

    def _append_stream_disconnect_notice_to_device_log(self, log_path: Path, exit_code: int) -> bool:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            phase_text = "启动阶段" if self.current_run_stream_disconnect_startup else "用例中途"
            notice = (
                "\n"
                f"[AutoGame][StreamDisconnect] {timestamp} launcher 检测到 gRPC 流断连({phase_text})，"
                f"exit_code={exit_code}，message={self.current_run_stream_disconnect_message}\n"
            )
            with log_path.open("a", encoding="utf-8", errors="ignore") as f:
                f.write(notice)
            return True
        except Exception:
            log_exception(f"append stream disconnect notice failed: log_path={log_path}")
            return False

    def _resolve_current_run_archive_dir(self) -> Optional[Path]:
        if self.current_run_archive_dir is not None:
            self.current_run_archive_dir.mkdir(parents=True, exist_ok=True)
            return self.current_run_archive_dir

        if self.current_plan is None:
            return None

        archive_metadata = {}
        target_case = str(self.current_plan.get("target_case") or "").strip()
        if target_case:
            archive_metadata["target_case"] = target_case
        if self.current_batch_start_timestamp:
            archive_metadata["batch_start_timestamp"] = self.current_batch_start_timestamp
        if self.current_run_start_timestamp:
            archive_metadata["run_start_timestamp"] = self.current_run_start_timestamp

        try:
            archive_dir = resolve_run_archive_dir(
                self.current_run_index + 1,
                extra_metadata=archive_metadata,
                create=True,
            )
            self.current_run_archive_dir = archive_dir
            return archive_dir
        except Exception:
            log_exception("resolve current run archive dir failed")
            return None

    def _write_stream_disconnect_immediate_artifacts(self):
        archive_dir = self._resolve_current_run_archive_dir()
        if archive_dir is None:
            return

        try:
            run_output_text = self._output_text_since(self.current_run_output_start)
            (archive_dir / "launcher_output_partial.txt").write_text(
                run_output_text,
                encoding="utf-8",
            )

            marker = {
                "event": "stream_disconnected",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_index": self.current_run_index + 1,
                "stream_started": self.current_run_stream_started,
                "stream_disconnect_startup": self.current_run_stream_disconnect_startup,
                "stream_disconnect_message": self.current_run_stream_disconnect_message,
                "batch_start_timestamp": self.current_batch_start_timestamp,
                "run_start_timestamp": self.current_run_start_timestamp,
                "note": "launcher 检测到断流后立即写入，完整归档会在子进程退出后继续执行。",
            }
            (archive_dir / "stream_disconnect_immediate.json").write_text(
                json.dumps(marker, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            log_exception(f"write stream disconnect immediate artifacts failed: archive_dir={archive_dir}")

    def _mark_current_run_sp_started(self, source: str, state: Optional[dict] = None):
        if state:
            self.current_run_sp_state = state
        if self.current_run_sp_started:
            return
        self.current_run_sp_started = True
        self.current_run_sp_started_monotonic = time.monotonic()
        LOGGER.info("sp recording started detected: source=%s state=%s", source, state)

    def _refresh_current_run_sp_state(self):
        archive_dir = self.current_run_archive_dir
        if archive_dir is None:
            return

        state_path = archive_dir / "sp_recording_state.json"
        controller_state_path = archive_dir / "sp_controller_state.json"
        state = {}
        if state_path.exists():
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    state.update(payload)
            except Exception:
                log_exception(f"read sp recording state failed: state_path={state_path}")
        if controller_state_path.exists():
            try:
                controller_state = json.loads(
                    controller_state_path.read_text(encoding="utf-8")
                )
                if isinstance(controller_state, dict):
                    state["controller"] = controller_state
                    for flag_name in ("sp_started_ever", "sp_recording", "sp_saved"):
                        if flag_name in controller_state:
                            state[flag_name] = bool(controller_state.get(flag_name))
            except Exception:
                log_exception(
                    f"read sp controller state failed: state_path={controller_state_path}"
                )
        if not state:
            return

        if self.current_run_sp_save_confirmed:
            state["sp_saved"] = True

        self.current_run_sp_state = state
        if (
            state.get("sp_started_ever")
            or state.get("sp_recording")
            or state.get("sp_saved")
        ):
            self._mark_current_run_sp_started("state_file", state)

    def _current_sp_actual_runtime_seconds(self) -> float:
        self._refresh_current_run_sp_state()
        state = self.current_run_sp_state
        controller_state = state.get("controller") if isinstance(state, dict) else None
        if not isinstance(controller_state, dict):
            controller_state = state if isinstance(state, dict) else {}

        try:
            effective_seconds = max(
                0.0,
                float(controller_state.get("effective_time_seconds", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            effective_seconds = 0.0

        if controller_state.get("sp_recording") and not controller_state.get("sp_saved"):
            try:
                state_written_at = float(
                    controller_state.get("state_written_at_epoch", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                state_written_at = 0.0
            if state_written_at <= 0 and self.current_run_archive_dir is not None:
                controller_state_path = (
                    self.current_run_archive_dir / SP_CONTROLLER_STATE_FILE
                )
                try:
                    state_written_at = controller_state_path.stat().st_mtime
                except OSError:
                    state_written_at = 0.0
            if state_written_at > 0:
                effective_seconds += max(0.0, time.time() - state_written_at)
        if effective_seconds <= 0 and self.current_run_sp_started_monotonic is not None:
            effective_seconds = max(
                0.0,
                time.monotonic() - self.current_run_sp_started_monotonic,
            )

        return effective_seconds

    def _wait_for_sp_save_settle(
        self,
        reason_label: str,
        actual_runtime_seconds: float,
    ) -> int:
        wait_seconds = calculate_sp_save_settle_seconds(actual_runtime_seconds)
        label = str(reason_label or "SP保全").strip() or "SP保全"
        actual_minutes = max(0.0, float(actual_runtime_seconds)) / 60.0
        self._log_message(
            f"[Launcher] {label}：SP 实际运行 {actual_minutes:.2f} 分钟，"
            f"按 max(60秒, 实际运行分钟×2) 需要等待 {wait_seconds} 秒；"
            "等待完成前不会清理游戏/SP 进程。\n"
        )
        self._set_status(f"{label}：SP 后台保存中，剩余 {wait_seconds} 秒。")

        previous_wait_state = self.sp_save_settle_in_progress
        self.sp_save_settle_in_progress = True
        deadline = time.monotonic() + wait_seconds
        last_reported = wait_seconds
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                QApplication.processEvents()
                time.sleep(min(0.1, remaining))
                remaining_seconds = max(0, int(math.ceil(deadline - time.monotonic())))
                if (
                    0 < remaining_seconds < last_reported
                    and (
                        last_reported - remaining_seconds >= 10
                        or remaining_seconds <= 5
                    )
                ):
                    self._log_message(
                        f"[Launcher] {label}：SP 后台保存中，"
                        f"剩余 {remaining_seconds} 秒。\n"
                    )
                    self._set_status(
                        f"{label}：SP 后台保存中，剩余 {remaining_seconds} 秒。"
                    )
                    last_reported = remaining_seconds
        finally:
            self.sp_save_settle_in_progress = previous_wait_state

        self._log_message(
            f"[Launcher] {label}：SP 后台保存等待 {wait_seconds} 秒已完成，"
            "现在可以继续归档和清理进程。\n"
        )

        if not previous_wait_state and self.pending_process_finished is not None:
            exit_code, exit_status = self.pending_process_finished
            self.pending_process_finished = None
            QTimer.singleShot(
                0,
                lambda code=exit_code, status=exit_status: self._on_process_finished(
                    code,
                    status,
                ),
            )
        return wait_seconds

    def _refresh_current_run_failure_signal(self):
        archive_dir = self.current_run_archive_dir
        if archive_dir is None:
            return

        signal_path = archive_dir / LAUNCHER_FAILURE_SIGNAL_FILE
        if not signal_path.exists():
            return

        try:
            payload = json.loads(signal_path.read_text(encoding="utf-8"))
        except Exception:
            log_exception(f"read launcher failure signal failed: signal_path={signal_path}")
            return

        if not isinstance(payload, dict):
            return

        failure_code = str(payload.get("failure_code") or "").strip()
        if not failure_code:
            return

        previous_code = self.current_run_failure_code
        self.current_run_failure_code = failure_code
        self.current_run_failure_reason = str(payload.get("failure_reason") or "").strip()
        details = payload.get("details")
        self.current_run_failure_details = details if isinstance(details, dict) else {}
        if previous_code != failure_code:
            self._log_message(
                f"[Launcher] 检测到子进程失败信号：code={self.current_run_failure_code}, "
                f"reason={self.current_run_failure_reason or '未提供'}。\n"
            )

    def _current_run_failed_by_inactivity_timeout(self) -> bool:
        return self.current_run_failure_code == "launcher_inactivity_timeout"

    def _current_run_stopped_by_marathon_battery(self) -> bool:
        controller_state = self.current_run_sp_state.get("controller")
        if not isinstance(controller_state, dict):
            return False
        return bool(controller_state.get("battery_stop_requested"))

    def _current_plan_uses_sp_recording(self) -> bool:
        if self.current_plan is None:
            return True
        return should_use_sp_recording_for_profile(self.current_plan.get("test_profile"))

    def _current_plan_uses_hdc_capture(self) -> bool:
        if self.current_plan is None:
            return False
        screen_mode = str(self.current_plan.get("screen_mode") or "").strip()
        return stream_disconnect_policy_for_screen_mode(screen_mode) == STREAM_DISCONNECT_POLICY_DISABLED

    def _stream_disconnect_recovery_enabled(self) -> bool:
        if self.current_plan is None:
            return True
        screen_mode = str(self.current_plan.get("screen_mode") or "").strip()
        return stream_disconnect_policy_for_screen_mode(screen_mode) != STREAM_DISCONNECT_POLICY_DISABLED

    def _current_plan_stops_on_stream_disconnect(self) -> bool:
        if self.current_plan is None:
            return False
        screen_mode = str(self.current_plan.get("screen_mode") or "").strip()
        return stream_disconnect_policy_for_screen_mode(screen_mode) == STREAM_DISCONNECT_POLICY_STOP_ONLY

    def _current_plan_recovers_stream_only_on_disconnect(self) -> bool:
        if self.current_plan is None:
            return False
        screen_mode = str(self.current_plan.get("screen_mode") or "").strip()
        return stream_disconnect_policy_for_screen_mode(screen_mode) == STREAM_DISCONNECT_POLICY_STREAM_ONLY

    def _capture_stream_disconnect_screenshot(self, archive_dir: Path) -> Optional[Path]:
        screenshot_dir = archive_dir / "stream_disconnect_screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        remote_path = f"/data/local/tmp/stream_disconnect_{timestamp}.jpeg"
        local_path = screenshot_dir / f"stream_disconnect_{timestamp}.jpeg"
        need_remote_rm = False

        try:
            snap_result = subprocess.run(
                ["hdc", "shell", "snapshot_display", "-f", remote_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
                **hidden_subprocess_kwargs(),
            )
            if snap_result.returncode != 0:
                raise RuntimeError(snap_result.stderr.strip() or snap_result.stdout.strip())
            need_remote_rm = True

            recv_result = subprocess.run(
                ["hdc", "file", "recv", remote_path, str(local_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
                **hidden_subprocess_kwargs(),
            )
            if recv_result.returncode != 0:
                raise RuntimeError(recv_result.stderr.strip() or recv_result.stdout.strip())

            return local_path
        except Exception:
            log_exception("capture stream disconnect screenshot failed")
            return None
        finally:
            if need_remote_rm:
                try:
                    subprocess.run(
                        ["hdc", "shell", "rm", remote_path],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=5,
                        **hidden_subprocess_kwargs(),
                    )
                except Exception:
                    pass

    def _save_sp_for_preserve(self, reason_label: str) -> dict:
        self.run_timeout_timer.stop()
        try:
            resolution = get_resolution()
        except Exception:
            resolution = None

        if resolution:
            screen_w, screen_h = int(resolution[0]), int(resolution[1])
        else:
            screen_w, screen_h = 2832, 1316

        command, x, y, duration_ms = build_sp_save_shell_command(screen_w, screen_h)
        label = str(reason_label or "SP保全").strip() or "SP保全"

        self._log_message(
            f"[Launcher] {label}：尝试长按 SP 保存，pos=({x},{y}), duration={duration_ms}ms。\n"
        )
        result = run_hdc_shell(command)
        ok = result is not None
        if not ok:
            self._log_message(
                f"[Launcher] {label}：SP 保存指令执行失败，请检查 hdc/uinput 状态。\n",
                level=logging.WARNING,
            )
            return {
                "ok": False,
                "actual_runtime_seconds": self._current_sp_actual_runtime_seconds(),
                "settle_seconds": 0,
            }

        actual_runtime_seconds = self._current_sp_actual_runtime_seconds()
        self.current_run_sp_save_confirmed = True
        self.current_run_sp_state["sp_saved"] = True
        settle_seconds = self._wait_for_sp_save_settle(
            label,
            actual_runtime_seconds,
        )
        return {
            "ok": True,
            "actual_runtime_seconds": actual_runtime_seconds,
            "settle_seconds": settle_seconds,
        }

    def _save_sp_on_stream_disconnect(self) -> dict:
        return self._save_sp_for_preserve("断流保全")

    def _preserve_stream_disconnect_run_state(self):
        if self.current_run_stream_preserved:
            return
        self.current_run_stream_preserved = True

        archive_dir = self._resolve_current_run_archive_dir()
        preserve_result = {
            "event": "stream_disconnect_preserve",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_index": self.current_run_index + 1,
            "stream_disconnect_startup": self.current_run_stream_disconnect_startup,
            "stream_disconnect_message": self.current_run_stream_disconnect_message,
            "sp_recording_enabled": self._current_plan_uses_sp_recording(),
            "sp_started": self.current_run_sp_started,
            "sp_state": self.current_run_sp_state,
            "screenshot_path": None,
            "sp_save_attempted": False,
            "sp_save_ok": False,
            "sp_actual_runtime_seconds": 0.0,
            "sp_save_settle_seconds": 0,
            "sp_save_skipped_reason": "",
        }

        if archive_dir is not None:
            screenshot_path = self._capture_stream_disconnect_screenshot(archive_dir)
            preserve_result["screenshot_path"] = str(screenshot_path) if screenshot_path else None

        if not self.current_run_stream_disconnect_startup and self._current_plan_uses_sp_recording():
            preserve_result["sp_save_attempted"] = True
            save_result = self._save_sp_on_stream_disconnect()
            preserve_result["sp_save_ok"] = save_result["ok"]
            preserve_result["sp_actual_runtime_seconds"] = save_result[
                "actual_runtime_seconds"
            ]
            preserve_result["sp_save_settle_seconds"] = save_result["settle_seconds"]

        if archive_dir is not None:
            try:
                (archive_dir / "stream_disconnect_preserve.json").write_text(
                    json.dumps(preserve_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                log_exception(f"write stream disconnect preserve result failed: archive_dir={archive_dir}")

    def _preserve_inactivity_timeout_run_state(self):
        if self.current_run_inactivity_preserved:
            return
        self.current_run_inactivity_preserved = True

        self._refresh_current_run_sp_state()
        self._refresh_current_run_failure_signal()
        archive_dir = self._resolve_current_run_archive_dir()
        details = dict(self.current_run_failure_details or {})
        preserve_result = {
            "event": "inactivity_timeout_preserve",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_index": self.current_run_index + 1,
            "failure_code": self.current_run_failure_code,
            "failure_reason": self.current_run_failure_reason,
            "failure_details": details,
            "sp_recording_enabled": self._current_plan_uses_sp_recording(),
            "sp_started": self.current_run_sp_started,
            "sp_state": self.current_run_sp_state,
            "screenshot_path": details.get("screenshot_path"),
            "sp_save_attempted": False,
            "sp_save_ok": False,
            "sp_actual_runtime_seconds": 0.0,
            "sp_save_settle_seconds": 0,
            "sp_save_skipped_reason": "",
        }

        if not self._current_plan_uses_sp_recording():
            preserve_result["sp_save_skipped_reason"] = "sp_recording_disabled"
        elif self.current_run_sp_state.get("sp_saved"):
            preserve_result["sp_save_skipped_reason"] = "sp_already_saved"
            self._log_message("[Launcher] 无操作保全：SP 状态显示已经保存，跳过重复长按。\n")
        else:
            preserve_result["sp_save_attempted"] = True
            save_result = self._save_sp_for_preserve("无操作保全")
            preserve_result["sp_save_ok"] = save_result["ok"]
            preserve_result["sp_actual_runtime_seconds"] = save_result[
                "actual_runtime_seconds"
            ]
            preserve_result["sp_save_settle_seconds"] = save_result["settle_seconds"]

        if archive_dir is not None:
            try:
                (archive_dir / "inactivity_timeout_preserve.json").write_text(
                    json.dumps(preserve_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                log_exception(f"write inactivity timeout preserve result failed: archive_dir={archive_dir}")

    def _preserve_manual_stop_run_state(self):
        """Best-effort SP save before manual-stop cleanup, while keeping the stop bounded."""
        self._refresh_current_run_sp_state()
        archive_dir = self._resolve_current_run_archive_dir()
        preserve_result = {
            "event": "manual_stop_preserve",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_index": self.current_run_index + 1,
            "sp_recording_enabled": self._current_plan_uses_sp_recording(),
            "sp_started": self.current_run_sp_started,
            "sp_state": self.current_run_sp_state,
            "sp_save_attempted": False,
            "sp_save_ok": False,
            "sp_actual_runtime_seconds": 0.0,
            "sp_save_settle_seconds": 0,
            "sp_save_skipped_reason": "",
        }

        if not self._current_plan_uses_sp_recording():
            preserve_result["sp_save_skipped_reason"] = "sp_recording_disabled"
        elif self.current_run_sp_state.get("sp_saved"):
            preserve_result["sp_save_skipped_reason"] = "sp_already_saved"
        elif not self.current_run_sp_started:
            preserve_result["sp_save_skipped_reason"] = "sp_not_started"
        else:
            preserve_result["sp_save_attempted"] = True
            save_result = self._save_sp_for_preserve("手动停止保全")
            preserve_result["sp_save_ok"] = save_result["ok"]
            preserve_result["sp_actual_runtime_seconds"] = save_result[
                "actual_runtime_seconds"
            ]
            preserve_result["sp_save_settle_seconds"] = save_result["settle_seconds"]

        if archive_dir is not None:
            try:
                (archive_dir / "manual_stop_preserve.json").write_text(
                    json.dumps(preserve_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                log_exception(
                    f"write manual stop preserve result failed: archive_dir={archive_dir}"
                )

    def _mark_stream_disconnected(self, message: str, source: str):
        if self.current_run_stream_disconnected:
            return

        self._refresh_current_run_sp_state()
        self.current_run_stream_disconnected = True
        uses_sp_recording = self._current_plan_uses_sp_recording()
        stop_only_disconnect = self._current_plan_stops_on_stream_disconnect()
        stream_only_disconnect = self._current_plan_recovers_stream_only_on_disconnect()
        if stop_only_disconnect:
            self.current_run_stream_disconnect_startup = True
        elif uses_sp_recording:
            self.current_run_stream_disconnect_startup = not self.current_run_sp_started
        else:
            self.current_run_stream_disconnect_startup = not self.current_run_stream_started
        self.current_run_stream_disconnect_message = str(message or source or "stream disconnected")
        if stream_only_disconnect and self.current_run_stream_disconnect_startup:
            phase_text = "HOScrcpy流恢复"
        elif stop_only_disconnect:
            phase_text = "流停止"
        elif self.current_run_stream_disconnect_startup:
            phase_text = "启动阶段" if uses_sp_recording else "启动阶段(首帧未到达)"
        else:
            phase_text = "SP记录后" if uses_sp_recording else "功能测试首帧后"
        self._log_message(
            f"\n[Launcher] 检测到 gRPC 流断连({phase_text}, source={source})："
            f"{self.current_run_stream_disconnect_message}\n"
        )
        if self.current_run_stream_disconnect_startup:
            if stream_only_disconnect:
                self._log_message(
                    "[Launcher] HOScrcpy 抓图流断开，正在停止当前子进程；随后只恢复流服务并重跑当前用例。\n"
                )
            elif stop_only_disconnect:
                self._log_message(
                    "[Launcher] HOScrcpy 抓图流断开，当前任务已停止。"
                    "不截图、不保存 SP、不归档本轮日志，直接停止当前子进程。\n"
                )
            elif uses_sp_recording:
                self._log_message(
                    "[Launcher] SP 记录尚未开始，本次按启动阶段断流处理："
                    "不截图、不保存 SP、不归档本轮日志，直接停止当前子进程并准备重启。\n"
                )
            else:
                self._log_message(
                    "[Launcher] 功能测试首帧尚未到达，本次按启动阶段断流处理："
                    "不截图、不归档本轮日志，直接停止当前子进程并准备重启。\n"
                )
            status_text = (
                "HOScrcpy 流断连，正在停止当前子进程；随后只恢复流服务并重跑当前用例。"
                if stream_only_disconnect
                else (
                    "HOScrcpy 流断连，正在停止当前子进程。"
                    if stop_only_disconnect
                    else "启动阶段 gRPC 流断连，正在停止当前子进程并准备重启手机。"
                )
            )
            self._set_status(status_text)
            self._request_stream_disconnect_process_exit(immediate=True)
            return

        self._write_stream_disconnect_immediate_artifacts()
        self._preserve_stream_disconnect_run_state()
        self._log_message(
            "[Launcher] 归档时会把本次 testcases 设备日志复制到对应运行目录；"
            "子进程退出前会先触发 testcases 的 stop_device_log()。\n"
        )
        if uses_sp_recording:
            self._set_status("SP记录后检测到 gRPC 流断连，正在保存本轮状态并等待用例收尾。")
        else:
            self._set_status("功能测试中检测到 gRPC 流断连，正在保存本轮状态并等待用例收尾。")
        self._request_stream_disconnect_process_exit(immediate=False)

    def _request_stream_disconnect_process_exit(self, immediate: bool = False):
        if self.process is None:
            return
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._log_message("[Launcher] 断流保全：子进程已退出，继续归档本轮状态。\n")
            return

        if immediate:
            if self._current_plan_recovers_stream_only_on_disconnect():
                self._log_message("[Launcher] HOScrcpy 断流：立即终止当前子进程，随后只恢复流服务。\n")
            else:
                self._log_message("[Launcher] 启动阶段断流：立即终止当前子进程，随后执行重启恢复。\n")
            self.process.terminate()
            QTimer.singleShot(
                STREAM_DISCONNECT_FORCE_KILL_TIMEOUT_MS,
                self._force_kill_stream_disconnect_process_if_running,
            )
            return

        self._log_message(
            "[Launcher] 断流保全：等待子进程正常退出，"
            f"{STREAM_DISCONNECT_GRACEFUL_STOP_TIMEOUT_MS // 1000}s 后仍未退出则发送终止信号。\n"
        )
        QTimer.singleShot(
            STREAM_DISCONNECT_GRACEFUL_STOP_TIMEOUT_MS,
            self._terminate_stream_disconnect_process_if_running,
        )

    def _terminate_stream_disconnect_process_if_running(self):
        if self.manual_stop_requested:
            return
        if (
            self.process is not None
            and self.current_run_stream_disconnected
            and self.process.state() != QProcess.ProcessState.NotRunning
        ):
            self._log_message(
                "[Launcher] 断流保全：子进程未在宽限时间内退出，发送终止信号。\n",
                level=logging.WARNING,
            )
            self.process.terminate()
            QTimer.singleShot(
                STREAM_DISCONNECT_FORCE_KILL_TIMEOUT_MS,
                self._force_kill_stream_disconnect_process_if_running,
            )

    def _force_kill_stream_disconnect_process_if_running(self):
        if self.manual_stop_requested:
            return
        if (
            self.process is not None
            and self.current_run_stream_disconnected
            and self.process.state() != QProcess.ProcessState.NotRunning
        ):
            self._log_message(
                "[Launcher] 断流保全：子进程收到终止信号后仍未退出，执行强制结束。\n",
                level=logging.WARNING,
            )
            self.process.kill()

    def _poll_stream_disconnect_signal(self):
        if self.manual_stop_requested:
            return
        if self.process is None or self.current_run_stream_disconnected:
            return
        if not self._stream_disconnect_recovery_enabled():
            return

        archive_dir = self._resolve_current_run_archive_dir()
        if archive_dir is None:
            return

        signal_path = archive_dir / "stream_disconnect_signal.json"
        if not signal_path.exists():
            return

        try:
            payload = json.loads(signal_path.read_text(encoding="utf-8"))
            message = payload.get("message") or payload.get("reason") or str(signal_path)
            reason = str(payload.get("reason") or "")
        except Exception:
            message = str(signal_path)
            reason = ""

        if not self.current_run_stream_started and reason != "channel_ready_timeout":
            self.current_run_stream_started = True

        self._mark_stream_disconnected(message, "signal_file")

    def _stop_current_hdc_debug_capture(self):
        capture, self.current_hdc_debug_capture = (
            self.current_hdc_debug_capture,
            None,
        )
        if capture is None:
            return
        try:
            capture.stop()
            self._log_message(
                "[Launcher] HDC DEBUG 分轮采集已收尾："
                "bytes=%s rotations=%s path=%s\n"
                % (capture.bytes_captured, capture.rotation_count, capture.path)
            )
        except Exception:
            log_exception("stop current HDC DEBUG capture failed")

    def _stop_current_memory_capture(self):
        capture, self.current_memory_capture = self.current_memory_capture, None
        if capture is None:
            return
        try:
            capture.stop()
            self._log_message(
                "[Launcher] 内存监控已收尾：samples=%s path=%s\n"
                % (capture.sample_count, capture.path)
            )
        except Exception:
            log_exception("stop current memory capture failed")

    def _stop_current_hilog_capture(self):
        capture, self.current_hilog_capture = self.current_hilog_capture, None
        if capture is None:
            return
        try:
            capture.stop()
            self._log_message(
                "[Launcher] hilog 分轮采集已收尾：returncode=%s path=%s\n"
                % (capture.returncode, capture.path)
            )
        except Exception:
            log_exception("stop current hilog capture failed")

    def _archive_run_outputs(self, run_no: int, exit_code: int):
        if self.current_plan is None:
            return

        extra_log_files = {}
        device_log_path = self._resolve_current_device_log_path()
        if device_log_path is not None:
            if self.current_run_stream_disconnected:
                self._append_stream_disconnect_notice_to_device_log(
                    device_log_path,
                    exit_code,
                )
            self._wait_for_device_log_stable(device_log_path)
            if device_log_path.exists() and device_log_path.is_file():
                direct_hilog_path = (
                    self.current_run_archive_dir / "hilog.txt"
                    if self.current_run_archive_dir is not None
                    else None
                )
                if direct_hilog_path is None or device_log_path != direct_hilog_path:
                    extra_log_files["hilog.txt"] = str(device_log_path)
            else:
                self._log_message(
                    "[Launcher] 本次运行未找到 hilog 日志文件。\n",
                    level=logging.WARNING,
                )

        try:
            generate_preview_video = bool(
                self.current_plan.get("generate_preview_video")
            )
            archive_dir = archive_run_artifacts(
                run_index=run_no,
                source="launcher",
                extra_log_files=extra_log_files or None,
                generate_preview_video=generate_preview_video,
                extra_metadata={
                    "target_case": self.current_plan.get("target_case"),
                    "batch_start_timestamp": self.current_batch_start_timestamp,
                    "run_start_timestamp": self.current_run_start_timestamp,
                },
                reuse_existing=True,
                process_temp_logs_source_dir=self._current_preview_dir(),
            )
            self._log_message(f"[Launcher] 本次运行产物已归档到：{archive_dir}\n")
            (archive_dir / "launcher_output.txt").write_text(
                self._output_text_since(self.current_run_output_start),
                encoding="utf-8",
            )
            prune_run_archive_artifacts(
                archive_dir,
                keep_preview_video=generate_preview_video,
            )
        except Exception:
            log_exception(f"archive_run_outputs failed: run_no={run_no}")
            self._log_message("[Launcher] 运行产物归档失败，请查看 launcher_debug.log。\n", level=logging.ERROR)

    def _pull_current_run_sp_artifacts(self, run_no: int) -> bool:
        if not self._current_plan_uses_sp_recording():
            return False

        controller_state = self.current_run_sp_state.get("controller")
        controller_saved = (
            bool(controller_state.get("sp_saved"))
            if isinstance(controller_state, dict)
            else False
        )
        if not self.current_run_sp_state.get("sp_saved") and not controller_saved:
            self._log_message(
                f"[Launcher] 第 {run_no} 次未确认 SP 已保存，跳过手机端 SP 文件拉取。\n"
            )
            return False

        archive_dir = self._resolve_current_run_archive_dir()
        if archive_dir is None:
            self._log_message(
                f"[Launcher] 第 {run_no} 次 SP 已保存，但无法确定 game_cases 目录。\n",
                level=logging.ERROR,
            )
            return False

        self._set_status(f"SP 已保存，正在拉取第{run_no}次sp记录。")
        QApplication.processEvents()
        record_dir, results = pull_saved_sp_artifacts(archive_dir, run_no)
        failed = [item for item in results if not item["ok"]]
        for item in results:
            level = logging.INFO if item["ok"] else logging.ERROR
            result_text = "成功" if item["ok"] else "失败"
            detail = f"，detail={item['detail']}" if item["detail"] else ""
            self._log_message(
                f"[Launcher] SP文件拉取{result_text}："
                f"{item['remote_path']} -> {item['local_path']}{detail}\n",
                level=level,
            )

        if failed:
            self._set_status(
                f"第{run_no}次 SP 记录部分拉取失败，请查看 Launcher 日志。"
            )
            return False

        self._log_message(f"[Launcher] 第 {run_no} 次 SP 记录已保存到：{record_dir}\n")
        return True

    def _discard_startup_disconnect_archive_dir(self):
        archive_dir = self.current_run_archive_dir
        if archive_dir is None or not archive_dir.exists():
            return

        try:
            shutil.rmtree(archive_dir)
            self._log_message(
                f"[Launcher] 启动阶段断流未归档本轮产物，已清理预创建目录：{archive_dir}\n"
            )
        except Exception:
            log_exception(f"discard startup disconnect archive dir failed: archive_dir={archive_dir}")
            self._log_message(
                f"[Launcher] 启动阶段断流目录清理失败，请手动检查：{archive_dir}\n",
                level=logging.WARNING,
            )

    def _handle_run_timeout(self):
        if self.process is None or self.current_plan is None:
            return
        self.current_run_timed_out = True
        LOGGER.warning(
            "run timeout: run_index=%s safe_minutes=%s pid=%s",
            self.current_run_index + 1,
            self.current_plan["safe_minutes"],
            int(self.process.processId()),
        )
        self._log_message(
            f"\n[Launcher] 第 {self.current_run_index + 1}/{self.current_plan['run_count']} 次运行已超过 "
            f"{self.current_plan['safe_minutes']} 分钟，正在停止本次用例。\n"
        )
        self._set_status("当前用例超过安全时间，正在停止本次运行。")
        self._request_current_process_shutdown("超过安全时间")

    def _request_current_process_shutdown(self, reason: str):
        process = self.process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        self._log_message(
            f"[Launcher] {reason}：先发送终止信号，"
            f"{RUN_STOP_FORCE_KILL_TIMEOUT_MS // 1000}s 后仍未退出再强制结束。\n"
        )
        process.terminate()
        QTimer.singleShot(
            RUN_STOP_FORCE_KILL_TIMEOUT_MS,
            lambda target=process: self._force_kill_requested_process_if_running(target),
        )

    def _force_kill_requested_process_if_running(self, target_process):
        if (
            self.process is target_process
            and target_process.state() != QProcess.ProcessState.NotRunning
        ):
            self._log_message(
                "[Launcher] 子进程未在宽限时间内退出，执行强制结束。\n",
                level=logging.WARNING,
            )
            target_process.kill()

    def _prepare_capture_mode_for_plan(self, plan: dict, issues: ValidationIssues) -> bool:
        screen_mode = normalize_launcher_screen_mode(plan.get("screen_mode") or "2")
        plan["screen_mode"] = screen_mode
        test_profile = str(plan.get("test_profile") or "power")
        try:
            write_screen_mode_config(screen_mode)
        except Exception as exc:
            log_exception("write screen_mode config failed")
            issues.add_error("截图模式配置失败", f"写入 screen_mode={screen_mode} 失败：{exc}")
            return False

        profile_text = {
            "power": "功耗测试",
            "function": "功能测试",
            TEST_PROFILE_MARATHON: "马拉松",
        }.get(test_profile, test_profile)
        self._log_message(
            f"[Launcher] 已切换为{profile_text}，screen_mode={screen_mode}。\n"
        )
        check_result = check_capture_stream_for_screen_mode(screen_mode)
        plan["capture_preflight_message"] = check_result.message
        self._log_message(f"[Launcher] 截图流预检：{check_result.message}\n")
        if not check_result.ok:
            issues.add_error("截图流预检失败", check_result.message)
            return False
        return True

    def _toggle_stream_verification(self):
        if self.stream_verify_active:
            self._stop_stream_verification("验证流已关闭。")
            return
        self._start_stream_verification()

    def _on_hos_frame_rate_changed(self, _index: int):
        frame_rate = self.hos_frame_rate_combo.currentData()
        try:
            frame_rate = int(frame_rate)
            write_hoscrcpy_frame_rate_config(frame_rate)
        except Exception as exc:
            log_exception("write hoscrcpy frame rate config failed")
            self.hos_frame_rate_combo.blockSignals(True)
            self.hos_frame_rate_combo.setCurrentIndex(
                self.hos_frame_rate_combo.findData(self.current_hos_frame_rate)
            )
            self.hos_frame_rate_combo.blockSignals(False)
            message = f"帧率配置写入失败：{exc}"
            self._log_message(f"[Launcher] {message}\n", level=logging.ERROR)
            QMessageBox.warning(self, "帧率", message)
            return

        self.current_hos_frame_rate = frame_rate
        message = f"HOS 验证流帧率已设为 {frame_rate}，已同步 config.json。"
        self._set_status(message)
        self._log_message(f"[Launcher] {message}\n")

    def _start_stream_verification(self):
        if self.batch_active or self.process is not None:
            QMessageBox.information(self, "运行中", "当前已有任务在运行，请先停止。")
            return

        try:
            screen_mode = str(get_screen_mode(str(AUTOGAME_CONFIG_FILE))).strip()
            if screen_mode not in {"0", "1", "2"}:
                raise ValueError(f"未知 screen_mode: {screen_mode}")
            from aw.autogame.stream_client.stream_client import FrameBuffer

            buffer = FrameBuffer(size=1)
        except Exception as exc:
            message = f"验证流准备失败：{exc}"
            log_exception("stream verification prepare failed")
            self._log_message(f"[Launcher] {message}\n", level=logging.ERROR)
            QMessageBox.warning(self, "验证流", message)
            return

        with self.stream_verify_lock:
            self.stream_verify_active = True
            self.stream_verify_first_frame_seen = False
            self.stream_verify_screen_mode = screen_mode
            self.stream_verify_buffer = buffer
            self.stream_verify_client = None
            self.stream_verify_failure_reported = False

        self.stream_verify_button.setText("验证中...")
        self.stream_verify_button.setEnabled(True)
        self.hos_frame_rate_combo.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stream_verify_timer.start()
        self._set_status(f"正在验证视频流，screen_mode={screen_mode}。")
        self._log_message(f"[Launcher] 正在验证视频流，screen_mode={screen_mode}。\n")

        thread = threading.Thread(
            target=self._start_stream_client_for_verification,
            args=(screen_mode, buffer),
            daemon=True,
            name=f"launcher-stream-verify-{screen_mode}",
        )
        self.stream_verify_thread = thread
        thread.start()

    def _resolve_stream_verification_capture_size(
        self,
        screen_width: Optional[int],
        screen_height: Optional[int],
    ) -> tuple[int, int]:
        width = 720
        height = None
        try:
            config = json.loads(AUTOGAME_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(config, dict):
                width = int(config.get("width") or width)
                raw_height = config.get("height")
                if raw_height:
                    height = int(raw_height)
        except Exception:
            log_exception("read stream verification capture size failed")

        if height is None:
            if screen_width and screen_height:
                short_side = min(int(screen_width), int(screen_height))
                long_side = max(int(screen_width), int(screen_height))
                if short_side > 0:
                    height = int(width * (long_side / short_side))
            if height is None:
                height = 1280

        return width, height

    def _start_stream_client_for_verification(self, screen_mode: str, buffer):
        client = None
        try:
            from aw.autogame.tools.GameAutomator import create_stream_client_for_mode

            screen_width, screen_height = get_resolution()
            capture_width, capture_height = self._resolve_stream_verification_capture_size(
                screen_width,
                screen_height,
            )
            screen_width = screen_width or capture_width
            screen_height = screen_height or capture_height
            self._set_preview_render_screen_size(screen_width, screen_height, "stream_verification")
            display_rotation = get_display_rotation() if screen_mode == "0" else None
            client = create_stream_client_for_mode(
                screen_mode,
                buffer,
                screen_width,
                screen_height,
                capture_width,
                capture_height,
                display_rotation,
            )
            if hasattr(client, "set_save_frame"):
                client.set_save_frame(False)

            with self.stream_verify_lock:
                if (
                    not self.stream_verify_active
                    or self.stream_verify_buffer is not buffer
                    or self.stream_verify_screen_mode != screen_mode
                ):
                    return
                self.stream_verify_client = client

            if screen_mode == "0":
                client.start_backend(
                    lowh=0,
                    highh=10000,
                    skip=20,
                    width=capture_width,
                    height=capture_height,
                )
            else:
                client.start_backend()
            LOGGER.info(
                "stream verification client started: screen_mode=%s capture=%sx%s screen=%sx%s",
                screen_mode,
                capture_width,
                capture_height,
                screen_width,
                screen_height,
            )
        except Exception as exc:
            log_exception("stream verification start failed")
            if client is not None:
                try:
                    client.stop()
                except Exception:
                    pass
            self.stream_verification_failed.emit(f"验证流启动失败：{exc}")

    def _handle_stream_verification_failed(self, message: str):
        with self.stream_verify_lock:
            if not self.stream_verify_active or self.stream_verify_failure_reported:
                return
            self.stream_verify_failure_reported = True
        self._stop_stream_verification("")
        self._log_message(f"[Launcher] {message}\n", level=logging.ERROR)
        self._set_status(message)
        QMessageBox.warning(self, "验证流", message)

    def _stop_stream_verification(self, message: str = "验证流已关闭。"):
        self.stream_verify_timer.stop()
        with self.stream_verify_lock:
            client = self.stream_verify_client
            self.stream_verify_active = False
            self.stream_verify_first_frame_seen = False
            self.stream_verify_screen_mode = ""
            self.stream_verify_client = None
            self.stream_verify_buffer = None

        if client is not None:
            try:
                client.stop()
            except Exception:
                log_exception("stop stream verification client failed")

        self.stream_verify_button.setText("验证流")
        self.stream_verify_button.setEnabled(not self.batch_active and self.process is None)
        self.hos_frame_rate_combo.setEnabled(not self.batch_active and self.process is None)
        self.start_button.setEnabled(not self.batch_active and self.process is None)
        if message:
            self._set_status(message)
            self._log_message(f"[Launcher] {message}\n")

    def _poll_stream_verification_frame(self):
        if not self.stream_verify_active or self.stream_verify_buffer is None:
            return

        frame = self.stream_verify_buffer.get_latest(timeout=0.01)
        if frame is None:
            self._check_stream_verification_client_state()
            return

        pixmap = stream_frame_to_qpixmap(frame)
        if pixmap.isNull():
            return

        if not self.stream_verify_first_frame_seen:
            self.stream_verify_first_frame_seen = True
            self.stream_verify_button.setText("关闭流")
            self._set_status(f"验证流已出图，screen_mode={self.stream_verify_screen_mode}。")
            self._log_message(
                f"[Launcher] 验证流已出图，screen_mode={self.stream_verify_screen_mode}。\n"
            )

        payload = {
            "stage": "验证流",
            "screen_mode": self.stream_verify_screen_mode,
            "frame": {
                "source": "launcher_stream_verification",
                "width": pixmap.width(),
                "height": pixmap.height(),
            },
        }
        self.latest_preview_file = None
        self.latest_preview_pixmap = pixmap
        self.latest_preview_payload = payload
        self.preview_image_label.setText("")
        self._update_preview_info_list(payload)
        self._adjust_preview_splitter_sizes()
        self._refresh_preview_pixmap()

    def _check_stream_verification_client_state(self):
        client = self.stream_verify_client
        if client is None or self.stream_verify_first_frame_seen:
            return

        main_thread = getattr(client, "main_thread", None)
        if main_thread is not None and not main_thread.is_alive():
            last_error = getattr(client, "_last_error", None)
            message = str(last_error) if last_error is not None else "验证流未拿到首帧，客户端已退出。"
            self._handle_stream_verification_failed(f"验证流已停止：{message}")

    def _start_run(self):
        LOGGER.info(
            "start_button clicked: batch_active=%s process_exists=%s",
            self.batch_active,
            self.process is not None,
        )
        if self.batch_active or self.process is not None:
            QMessageBox.information(self, "运行中", "当前已有任务在运行，请先停止。")
            return

        issues = ValidationIssues()
        plan = self._collect_plan(issues)
        if plan is not None:
            try:
                self.hdc_debug_level = resolve_hdc_debug_level()
                source_path = restart_hdc_debug_server(self.hdc_debug_level)
                self._log_message(
                    "[Launcher] HDC DEBUG server 已以 level=%s 启动，source=%s\n"
                    % (self.hdc_debug_level, source_path)
                )
            except Exception as exc:
                log_exception("start HDC DEBUG server failed")
                issues.add_error("HDC DEBUG 启动失败", str(exc))
            else:
                self._prepare_capture_mode_for_plan(plan, issues)

        if self._show_validation_issues("无法启动任务", issues):
            LOGGER.info("start_run aborted because validation failed")
            return

        if plan is None:
            LOGGER.error("start_run aborted without plan and without validation issue")
            return

        self._begin_batch(plan)

    def _read_process_output(self):
        if self.process is None:
            return
        raw_text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        text = self._take_complete_process_output(raw_text)
        if not text:
            return
        self._append_output(text)
        self._handle_stream_output(text)
        stripped = "".join(line for _, line in decode_output_text(text)).strip()
        if stripped:
            LOGGER.info("child_output: %s", stripped)

    def _take_complete_process_output(self, text: str, flush: bool = False) -> str:
        combined = self.process_output_buffer + str(text or "")
        if flush:
            self.process_output_buffer = ""
            return combined

        newline_index = max(combined.rfind("\n"), combined.rfind("\r"))
        if newline_index < 0:
            self.process_output_buffer = combined
            return ""

        self.process_output_buffer = combined[newline_index + 1:]
        return combined[:newline_index + 1]

    def _flush_process_output_buffer(self):
        text = self._take_complete_process_output("", flush=True)
        if not text:
            return
        self._append_output(text)
        self._handle_stream_output(text)
        stripped = "".join(line for _, line in decode_output_text(text)).strip()
        if stripped:
            LOGGER.info("child_output: %s", stripped)

    def _handle_stream_output(self, text: str):
        if not text:
            return

        self._handle_sp_output(text)

        if self.current_run_stream_disconnected:
            return

        if any(marker in text for marker in STREAM_CONNECTED_MARKERS):
            self.current_run_stream_started = True

        if not self._stream_disconnect_recovery_enabled():
            return

        matched_pattern = next(
            (pattern for pattern in STREAM_DISCONNECT_PATTERNS if pattern in text),
            None,
        )
        if matched_pattern is None:
            return

        message = self._extract_stream_disconnect_line(
            text,
            matched_pattern,
        )
        self._mark_stream_disconnected(message, "stdout")

    def _handle_sp_output(self, text: str):
        decoded_text = "".join(line for _, line in decode_output_text(text))
        if SP_SAVE_PROTECTION_LOG_MARKER in decoded_text:
            self.run_timeout_timer.stop()
            self._log_message(
                "[Launcher] 检测到 SP 长按保存开始，"
                "已停止本轮安全超时计时，等待后台数据落盘。\n"
            )
        if any(marker in decoded_text for marker in SP_RECORD_EVER_STARTED_MARKERS):
            self._mark_current_run_sp_started("stdout")

    def _extract_stream_disconnect_line(self, text: str, matched_pattern: str) -> str:
        for line in text.splitlines():
            if matched_pattern in line:
                return line.strip()
        return matched_pattern

    def _on_process_error(self, error):
        if self.process is None:
            LOGGER.error("process error signaled after process cleanup: error=%s", error)
            return
        LOGGER.error(
            "process error: error=%s error_string=%s state=%s pid=%s",
            error,
            self.process.errorString(),
            self.process.state(),
            int(self.process.processId()),
        )
        self._log_message(
            f"[Launcher] 子进程错误：error={error}, detail={self.process.errorString()}\n",
            level=logging.ERROR,
        )

    def _reinitialize_stream_service(self) -> bool:
        if self.manual_stop_requested:
            return False
        self._log_message("[Launcher] 仅执行流服务初始化...\n")

        hdc = resolve_hdc_executable()

        init_commands = [
            {
                "cmd": [hdc, "shell", "setenforce", "0"],
                "required": False,
                "desc": "关闭 SELinux 强制模式",
            },
            {
                "cmd": [hdc, "fport", "rm", "tcp:12345", "tcp:12345"],
                "required": False,
                "desc": "清理旧端口转发",
            },
            {
                "cmd": [hdc, "fport", "tcp:12345", "tcp:12345"],
                "required": True,
                "desc": "建立端口转发 tcp:12345 -> tcp:12345",
            },
        ]

        for item in init_commands:
            command = item["cmd"]
            required = item["required"]
            desc = item["desc"]

            try:
                self._log_message(f"[Launcher][init] 执行：{desc}，command={command}\n")

                result = self._run_interruptible_recovery_command(command, timeout=30)
                if result is None:
                    return False

                output = (result.stdout or "") + (result.stderr or "")
                if output.strip():
                    self._log_message(f"[Launcher][init] {output.rstrip()}\n")

                if result.returncode != 0:
                    msg = (
                        f"[Launcher] 初始化命令失败：{desc}，"
                        f"command={command}，returncode={result.returncode}\n"
                    )

                    if required:
                        self._log_message(msg, level=logging.ERROR)
                        return False

                    self._log_message(msg, level=logging.WARNING)
                    self._log_message(
                        "[Launcher] 非关键初始化命令失败，继续执行后续步骤。\n",
                        level=logging.WARNING,
                    )

            except Exception as exc:
                log_exception("reinitialize stream service failed")

                msg = (
                    f"[Launcher] 初始化命令异常：{desc}，"
                    f"command={command}，detail={exc}\n"
                )

                if required:
                    self._log_message(msg, level=logging.ERROR)
                    return False

                self._log_message(msg, level=logging.WARNING)
                self._log_message(
                    "[Launcher] 非关键初始化命令异常，继续执行后续步骤。\n",
                    level=logging.WARNING,
                )

        self._log_message("[Launcher] 流服务初始化完成。\n")
        return True

    def _track_recovery_process(self, proc: subprocess.Popen):
        self.recovery_processes = [
            item for item in self.recovery_processes if item.poll() is None
        ]
        self.recovery_processes.append(proc)

    def _untrack_recovery_process(self, proc: subprocess.Popen):
        self.recovery_processes = [
            item for item in self.recovery_processes if item is not proc and item.poll() is None
        ]

    def _stop_recovery_processes(self):
        processes, self.recovery_processes = self.recovery_processes, []
        for proc in processes:
            if proc.poll() is not None:
                continue
            terminate_popen_process_tree(proc, force=True)

    def _run_interruptible_recovery_command(
        self,
        command: list[str],
        timeout: float,
    ) -> Optional[subprocess.CompletedProcess]:
        popen_kwargs = hidden_subprocess_kwargs()
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            command,
            cwd=str(APP_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            **popen_kwargs,
        )
        self._track_recovery_process(proc)
        deadline = time.monotonic() + float(timeout)
        output = ""
        try:
            while True:
                if self.manual_stop_requested:
                    terminate_popen_process_tree(proc, force=True)
                    try:
                        output, _ = proc.communicate(timeout=1)
                    except Exception:
                        output = output or ""
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    terminate_popen_process_tree(proc, force=True)
                    raise subprocess.TimeoutExpired(command, timeout, output=output)
                try:
                    output, _ = proc.communicate(timeout=min(0.1, remaining))
                    return subprocess.CompletedProcess(
                        command,
                        int(proc.returncode or 0),
                        stdout=output,
                        stderr="",
                    )
                except subprocess.TimeoutExpired as exc:
                    if exc.output:
                        output = str(exc.output)
                    QApplication.processEvents()
        finally:
            self._untrack_recovery_process(proc)

    def _recover_stream_only_for_stream_disconnect(self) -> bool:
        if self.manual_stop_requested:
            return False
        if "hdc offline没有恢复" in self.current_run_stream_disconnect_message.lower():
            self._log_message(
                "[Launcher] hdc offline没有恢复，停止恢复；不再尝试 gRPC 重连或重跑用例。\n",
                level=logging.ERROR,
            )
            self._set_runtime("运行信息：hdc offline没有恢复，流恢复已终止。")
            QApplication.processEvents()
            return False

        self._log_message(
            "[Launcher] HOScrcpy 断流恢复：不重启手机，只重新启动用例；"
            "HOS SDK 会重新 setup、拉起投屏服务并等待新帧。\n"
        )
        self._set_runtime("运行信息：检测到 HOScrcpy 断流，正在只恢复流服务。")
        QApplication.processEvents()
        return not self.manual_stop_requested

    def _stream_recovery_failure_message(self) -> str:
        if "hdc offline没有恢复" in self.current_run_stream_disconnect_message.lower():
            return "hdc offline没有恢复，批量任务已终止。"
        return "HOScrcpy 流恢复失败，批量任务已终止。"

    def _restart_device_for_stream_disconnect(self) -> bool:
        if self.manual_stop_requested:
            return False
        self._log_message("[Launcher] 开始执行断流恢复命令。\n")
        self._set_runtime("运行信息：检测到断流，正在重启手机并等待开机。")
        QApplication.processEvents()

        script_path = APP_DIR / "restart.bat"
        if not script_path.is_file():
            self._log_message(f"[Launcher] 未找到重启脚本：{script_path}\n", level=logging.ERROR)
            return False

        self._log_message(f"[Launcher] 弹出 cmd 窗口执行断流恢复脚本：{script_path}\n")
        try:
            restart_proc = launch_restart_bat_cmd_window(script_path)
            self._track_recovery_process(restart_proc)
        except Exception:
            log_exception("restart device after stream disconnect failed")
            self._log_message(
                f"[Launcher] 执行断流恢复脚本失败：{script_path}，请查看 launcher_debug.log 或弹出的 cmd 窗口。\n",
                level=logging.ERROR,
            )
            return False

        self._log_message(
            f"[Launcher] 重启后固定等待 {REBOOT_RELAUNCH_DELAY_SECONDS}s，再重新启动用例。\n"
        )
        for _ in range(REBOOT_RELAUNCH_DELAY_SECONDS):
            QApplication.processEvents()
            if self.manual_stop_requested:
                self._stop_recovery_processes()
                self._log_message("[Launcher] 手动停止已取消断流恢复等待。\n")
                return False
            time.sleep(1)

        if self.manual_stop_requested:
            self._stop_recovery_processes()
            return False
        if not self._reinitialize_stream_service():
            return False

        if self.manual_stop_requested:
            self._stop_recovery_processes()
            return False

        self._log_message("[Launcher] 手机重启与端口恢复完成。\n")
        self._stop_recovery_processes()
        self.dismiss_reboot_prompt_on_next_case_start = True
        return True

    def _on_process_finished(self, exit_code: int, _exit_status):
        if self.sp_save_settle_in_progress:
            self.pending_process_finished = (int(exit_code), _exit_status)
            LOGGER.info(
                "process finished while SP save settle wait is active; defer cleanup: exit_code=%s",
                exit_code,
            )
            return
        LOGGER.info(
            "process finished: exit_code=%s exit_status=%s current_run_index=%s timed_out=%s",
            exit_code,
            _exit_status,
            self.current_run_index,
            self.current_run_timed_out,
        )
        self._flush_process_output_buffer()
        self.run_timeout_timer.stop()
        self.stream_disconnect_signal_timer.stop()
        self._stop_current_memory_capture()
        self._stop_current_hilog_capture()
        self._stop_current_hdc_debug_capture()
        trace_path = self.process_launch_tracer.stop()
        if trace_path is not None:
            self._log_message(f"[Launcher] 进程创建追踪已停止：{trace_path}\n")
        if not self.current_run_stream_disconnected:
            self._poll_stream_disconnect_signal()
        finish_prefix = "进程结束"
        if self.stop_requested:
            finish_prefix = "进程已手动停止"
        self._log_message(f"\n[Launcher] {finish_prefix}，exit_code={exit_code}\n")
        self._poll_preview_frame()
        self._refresh_current_run_sp_state()
        self._refresh_current_run_failure_signal()
        if exit_code != 0 and not self.current_run_failure_code:
            self.current_run_failure_code = "testcase_nonzero_exit"
            self.current_run_failure_reason = f"xDevice testcase exit_code={exit_code}"
            self.current_run_failure_details = {"exit_code": int(exit_code)}
            self._log_message(
                f"[Launcher] 本次用例执行失败：exit_code={exit_code}。\n",
                level=logging.ERROR,
            )
        if self.manual_stop_requested:
            self._preserve_manual_stop_run_state()
        if self._current_run_failed_by_inactivity_timeout():
            self._log_message(
                "[Launcher] 本次用例因长时间无操控主动结束，正在执行无操作保全。\n"
            )
            self._set_status("当前用例长时间无操控，正在长按 SP 保存并归档。")
            self._preserve_inactivity_timeout_run_state()
        run_no = self.current_run_index + 1
        startup_stream_disconnect = (
            self.current_run_stream_disconnected
            and self.current_run_stream_disconnect_startup
        )
        self._archive_run_outputs(run_no, exit_code)
        if self.manual_stop_requested or not startup_stream_disconnect:
            self._pull_current_run_sp_artifacts(run_no)
        self.preview_timer.stop()
        if self.process is not None:
            self.process.deleteLater()
            self.process = None

        if not self.batch_active or self.current_plan is None:
            self._finish_batch(f"任务已结束，退出码：{exit_code}")
            return

        if self.manual_stop_requested:
            self._cleanup_apps_between_runs("手动停止后清理", force=True)
            self._finish_batch("任务已停止。")
            return

        if self._current_run_stopped_by_marathon_battery():
            controller_state = self.current_run_sp_state.get("controller", {})
            battery = controller_state.get("last_battery_percent")
            threshold = controller_state.get("end_battery_percent")
            self._log_message(
                f"[Launcher] 马拉松因电量 {battery}% <= 结束电量 {threshold}% 已保存 SP，"
                "停止整个批次，不再启动后续运行。\n"
            )
            self._cleanup_apps_between_runs("马拉松低电量结束后清理")
            self._finish_batch(
                f"马拉松已因电量 {battery}% 达到结束电量 {threshold}% 而结束。"
            )
            return

        if self.current_run_stream_disconnected:
            if self.current_run_stream_disconnect_startup:
                if self._current_plan_recovers_stream_only_on_disconnect():
                    if not self._recover_stream_only_for_stream_disconnect():
                        if self.manual_stop_requested:
                            return
                        self._cleanup_apps_between_runs("断流恢复失败后清理")
                        self._finish_batch(self._stream_recovery_failure_message())
                        return
                    self._set_status(
                        f"第 {self.current_run_index + 1}/{self.current_plan['run_count']} 次 HOScrcpy 断流，"
                        "已只恢复流服务，准备重跑当前用例。"
                    )
                    self._set_runtime(
                        f"运行信息：HOScrcpy 断流已恢复流服务，准备重新执行第 "
                        f"{self.current_run_index + 1}/{self.current_plan['run_count']} 次。"
                    )
                    self._check_and_start_if_safe()
                    if self.batch_active and self.process is None and not self.safety_timer.isActive():
                        self.safety_timer.start()
                    return

                if self._current_plan_stops_on_stream_disconnect():
                    self._log_message(
                        "[Launcher] 本次 HOScrcpy 断流按直接停止处理，不计入已执行次数，"
                        "本轮不保存产物，不执行重启恢复。\n"
                    )
                    self._cleanup_apps_between_runs("断流自动停止后清理")
                    self._finish_batch("HOScrcpy 抓图流断开，当前任务已停止。")
                    return

                self._log_message(
                    "[Launcher] 本次断流发生在 SP 记录开始前，不计入已执行次数，"
                    "本轮不保存产物；重启手机后将重新运行当前用例。\n"
                )
                if not self._restart_device_for_stream_disconnect():
                    if self.manual_stop_requested:
                        return
                    self._cleanup_apps_between_runs("断流恢复失败后清理")
                    self._finish_batch("断流恢复失败，批量任务已终止。")
                    return
                self._set_status(
                    f"第 {self.current_run_index + 1}/{self.current_plan['run_count']} 次启动阶段断流，已重启手机，准备重跑。"
                )
                self._set_runtime(
                    f"运行信息：启动阶段断流已恢复，准备重新执行第 {self.current_run_index + 1}/{self.current_plan['run_count']} 次。"
                )
                self._check_and_start_if_safe()
                if self.batch_active and self.process is None and not self.safety_timer.isActive():
                    self.safety_timer.start()
                return

            if is_marathon_plan(self.current_plan):
                self._log_message(
                    "[Launcher] 马拉松本轮发生断流，已长按 SP 并按实际运行时长"
                    "完成后台落盘等待；中断状态已归档，清理进程后继续下一轮。\n"
                )
            else:
                self._log_message(
                    "[Launcher] 本次断流发生在 SP 记录开始后，结果已归档并写入 stream_disconnected 标志；"
                    "当前用例计为完成。\n"
                )

            self.current_run_index += 1
            if has_reached_plan_run_limit(self.current_plan, self.current_run_index):
                self._log_message(
                    "[Launcher] 本次中途断流发生在最后一轮，已保存 SP 并归档本轮状态，跳过手机重启。\n"
                )
                self._cleanup_apps_between_runs("最后一轮断流后清理")
                self._finish_batch("所有运行次数已完成。最后一轮断流已归档，未执行重启。")
                return

            if self._current_plan_recovers_stream_only_on_disconnect():
                if not self._recover_stream_only_for_stream_disconnect():
                    if self.manual_stop_requested:
                        return
                    self._cleanup_apps_between_runs("断流恢复失败后清理")
                    self._finish_batch(self._stream_recovery_failure_message())
                    return
                next_run = self.current_run_index + 1
                self._set_status(
                    f"第 {self.current_run_index}/{self.current_plan['run_count']} 次因 HOScrcpy 断流结束，"
                    f"已只恢复流服务，检查第 {next_run} 次启动条件。"
                )
                self._set_runtime(
                    f"运行信息：已完成 {self.current_run_index}/{self.current_plan['run_count']} 次，"
                    "HOScrcpy 断流已保存并恢复流服务，准备下一次安全检查。"
                )
                self._check_and_start_if_safe()
                if self.batch_active and self.process is None and not self.safety_timer.isActive():
                    self.safety_timer.start()
                return

            if not self._restart_device_for_stream_disconnect():
                if self.manual_stop_requested:
                    return
                self._cleanup_apps_between_runs("断流恢复失败后清理")
                self._finish_batch("断流恢复失败，批量任务已终止。")
                return

            next_run = self.current_run_index + 1
            self._set_status(
                f"第 {self.current_run_index}/{self.current_plan['run_count']} 次因断流结束，检查第 {next_run} 次启动条件。"
            )
            self._set_runtime(
                f"运行信息：已完成 {self.current_run_index}/{self.current_plan['run_count']} 次，断流恢复完成，准备下一次安全检查。"
            )
            self._check_and_start_if_safe()
            if self.batch_active and self.process is None and not self.safety_timer.isActive():
                self.safety_timer.start()
            return

        self._cleanup_apps_between_runs("轮次结束清理")
        self.current_run_index += 1
        if self.current_run_timed_out:
            self._log_message("[Launcher] 本次用例因超过安全时间被停止，计入已执行次数。\n")

        if has_reached_plan_run_limit(self.current_plan, self.current_run_index):
            self._finish_batch("所有运行次数已完成。")
            return

        next_run = self.current_run_index + 1
        if is_marathon_plan(self.current_plan):
            self._set_status(
                f"马拉松第 {self.current_run_index} 轮已保存 SP 并关闭游戏/SP 进程，"
                f"准备启动第 {next_run} 轮。"
            )
            self._set_runtime(
                f"运行信息：已完成 {self.current_run_index} 轮马拉松，准备下一轮。"
            )
        else:
            self._set_status(f"第 {self.current_run_index}/{self.current_plan['run_count']} 次已结束，检查第 {next_run} 次启动条件。")
            self._set_runtime(f"运行信息：已完成 {self.current_run_index}/{self.current_plan['run_count']} 次，准备下一次安全检查。")
        self._check_and_start_if_safe()
        if self.batch_active and self.process is None and not self.safety_timer.isActive():
            self.safety_timer.start()

    def _stop_run(self):
        LOGGER.info(
            "stop_button clicked: batch_active=%s process_exists=%s preserve_game_process=%s",
            self.batch_active,
            self.process is not None,
            should_preserve_game_process_for_plan(self.current_plan),
        )
        if not self.batch_active and self.process is None:
            return

        self.stop_requested = True
        self.manual_stop_requested = True
        self.stop_button.setEnabled(False)
        self.safety_timer.stop()
        self.run_timeout_timer.stop()
        self.stream_disconnect_signal_timer.stop()
        self.preview_timer.stop()
        self._stop_recovery_processes()

        if self.process is None:
            self._log_message("\n[Launcher] 手动停止已取消断流恢复和后续运行。\n")
            self._cleanup_apps_between_runs("手动停止后清理", force=True)
            self._finish_batch("任务已停止。")
            return

        self._log_message(
            "\n[Launcher] 手动停止优先：立即强制结束当前子进程，"
            "取消断流恢复和后续运行；子进程结束后保全日志与记录。\n"
        )
        self._set_status("手动停止中：正在结束进程并保全日志。")
        self.process.kill()


def _run_helper_command(args: argparse.Namespace) -> int:
    LOGGER.info("run_helper_command: args=%s", args)
    exit_code = 0
    try:
        if args.run_testcase:
            exit_code = run_testcase_entry(args.run_testcase)
            return exit_code

        if args.run_direct:
            project_case, target_case = args.run_direct
            run_direct_entry(project_case, target_case)
            return exit_code

        if args.run_game_recording:
            exit_code = run_game_recording_entry()
            return exit_code

        return exit_code
    except SystemExit as exc:
        raw_code = exc.code
        if raw_code is None:
            exit_code = 0
        elif isinstance(raw_code, int):
            exit_code = int(raw_code)
        else:
            exit_code = 1
        return exit_code
    except Exception:
        log_exception("helper command failed")
        traceback.print_exc()
        exit_code = 1
        return exit_code
    finally:
        LOGGER.info("helper exit_code=%s", exit_code)


def main():
    old_cwd = Path.cwd()
    chdir_error = ""
    try:
        os.chdir(APP_DIR)
    except Exception as exc:
        chdir_error = str(exc)

    setup_logging()
    install_global_exception_hooks()
    hidden_patch_installed = install_hidden_subprocess_patch()
    hidden_window_suppressor_started = start_hidden_subprocess_window_suppressor()
    LOGGER.info(
        "path context: frozen=%s sys_executable=%s __file__=%s APP_DIR=%s INTERNAL_DIR=%s ROOT_DIR=%s TEMP_DIR=%s old_cwd=%s new_cwd=%s chdir_error=%s hidden_subprocess_patch=%s hidden_window_suppressor=%s",
        bool(getattr(sys, "frozen", False)),
        sys.executable,
        __file__,
        APP_DIR,
        INTERNAL_DIR,
        ROOT_DIR,
        TEMP_DIR,
        old_cwd,
        Path.cwd(),
        chdir_error,
        hidden_patch_installed,
        hidden_window_suppressor_started,
    )
    if is_multiprocessing_child():
        LOGGER.info("detected multiprocessing fork argv=%s", sys.argv)
        close_pyinstaller_splash("multiprocessing-child")
        multiprocessing.freeze_support()
        LOGGER.info("skip LauncherWindow for multiprocessing child")
        return 0

    LOGGER.info(
        "main start: argv=%s cwd=%s executable=%s frozen=%s meipass=%s",
        sys.argv,
        os.getcwd(),
        sys.executable,
        bool(getattr(sys, "frozen", False)),
        getattr(sys, "_MEIPASS", None),
    )
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-testcase")
    parser.add_argument("--run-direct", nargs=2, metavar=("PROJECT_CASE", "TARGET_CASE"))
    parser.add_argument("--run-game-recording", action="store_true")
    args, unknown_args = parser.parse_known_args()
    is_helper = bool(args.run_testcase or args.run_direct or args.run_game_recording)
    LOGGER.info(
        "parsed args: %s unknown_args=%s is_helper=%s",
        args,
        unknown_args,
        is_helper,
    )

    if is_helper:
        LOGGER.info(
            "enter helper mode before QApplication: qapp_exists=%s",
            QApplication.instance() is not None,
        )
        close_pyinstaller_splash("helper")
        install_helper_signal_handlers()
        exit_code = _run_helper_command(args)
        LOGGER.info("helper mode exiting via SystemExit: exit_code=%s", exit_code)
        raise SystemExit(exit_code)

    LOGGER.info(
        "before QApplication: argv=%s qapp_exists=%s",
        sys.argv,
        QApplication.instance() is not None,
    )
    ensure_pyqt6_platform_plugin_path()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    LOGGER.info("before LauncherWindow: qapp_exists=%s", QApplication.instance() is not None)
    window = LauncherWindow()
    window.show()
    close_pyinstaller_splash("launcher-window-shown")
    LOGGER.info("launcher window shown")
    raise SystemExit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main() or 0)
