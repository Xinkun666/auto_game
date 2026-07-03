import argparse
import ast
import base64
import importlib
import json
import logging
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
from pathlib import Path
from typing import Dict, NamedTuple, Optional

from PyQt6.QtCore import QByteArray, QObject, QProcess, QProcessEnvironment, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QKeySequence, QPainter, QPen, QPixmap, QShortcut, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
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
from aw.autogame.tools.GameLaunchProfile import DEFAULT_SP_PACKAGE, should_use_sp_recording_for_profile
from aw.autogame.tools.Utils import archive_run_artifacts, get_resolution, resolve_run_archive_dir, select_scene_resolution
from aw.autogame.tools.AreaResolver import resolve_area_rect_for_frame

class AppPaths(NamedTuple):
    app_dir: Path
    internal_dir: Path
    root_dir: Path


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


def resolve_preview_frame_dir(app_dir: Optional[Path] = None) -> Path:
    return resolve_runtime_temp_dir(app_dir) / "logs" / "process_temp_logs"


def resolve_history_temp_dir() -> Path:
    return Path("aw") / "autogame" / "temp"


APP_PATHS = resolve_app_paths()
APP_DIR = APP_PATHS.app_dir
INTERNAL_DIR = APP_PATHS.internal_dir
ROOT_DIR = APP_PATHS.root_dir
AUTOGAME_CONFIG_FILE = ROOT_DIR / "aw" / "autogame" / "config" / "config.json"
TESTCASES_DIR = APP_DIR / "testcases"
CUSTOMS_EXAMPLES_DIR = ROOT_DIR / "aw" / "autogame" / "customs_examples"
CUSTOMS_GAME_EXAMPLES_DIR = ROOT_DIR / "aw" / "autogame" / "customs_game_examples"
TEMP_DIR = resolve_runtime_temp_dir(APP_DIR)
PREVIEW_DIR = resolve_preview_frame_dir(APP_DIR)
LOG_DIR = TEMP_DIR / "logs"
LAUNCHER_LOG_FILE = LOG_DIR / "launcher_debug.log"
PACKAGE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+){2,}")
LOGGER = logging.getLogger("launcher")
PREVIEW_FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}
PYINSTALLER_SUPPRESS_SPLASH_ENV = "PYINSTALLER_SUPPRESS_SPLASH_SCREEN"
PYINSTALLER_SPLASH_IPC_ENV = "_PYI_SPLASH_IPC"
PROCESS_TRACE_ENV = "AUTOGAME_PROCESS_TRACE"
STREAM_CONNECTED_MARKERS = (
    "[Stream] Start receiving...",
    "[HDC] First frame received.",
)
REBOOT_RELAUNCH_DELAY_SECONDS = 80
STREAM_DISCONNECT_SP_LONG_PRESS_MS = 3000
STREAM_DISCONNECT_SP_NORM_POS = (0.048, 0.295)
STREAM_DISCONNECT_GRACEFUL_STOP_TIMEOUT_MS = 60000
STREAM_DISCONNECT_FORCE_KILL_TIMEOUT_MS = 5000
STREAM_DISCONNECT_PATTERNS = (
    "[Stream] Channel ready timeout.",
    "[Stream] Receive loop ended unexpectedly.",
    "[Stream] gRPC Error:",
    "[Stream] Runtime Error:",
    "[HOS] Runtime Error:",
    "[HOS] Stream Error:",
)
SP_RECORD_EVER_STARTED_MARKERS = (
    "[Timer] sp 记录已开始",
    "[Timer] sp 记录已停止",
    "[Timer] sp 数据已保存",
)
DISMISS_REBOOT_PROMPT_ENV = "AUTOGAME_DISMISS_REBOOT_PROMPT"
DEVICE_LOG_SETTLE_TIMEOUT_SECONDS = 3.0
DEVICE_LOG_SETTLE_INTERVAL_SECONDS = 0.2
DEVICE_LOG_STOP_WAIT_TIMEOUT_SECONDS = 15.0
HDC_SHELL_TIMEOUT_SECONDS = float(os.environ.get("AUTOGAME_HDC_SHELL_TIMEOUT_SECONDS", "5"))
TEST_PROFILE_SCREEN_MODES = {
    "power": "2",
    "function": "2",
}
PUBG_CASE_KEYWORDS = ("和平精英", "pubg")
PUBG_CASE_DEFAULT_LOOP_COUNT = 2
PUBG_CASE_RUNTIME_DESCRIPTION = "和平精英用例默认10分钟搜房、10分钟开车、10分钟跑图，总测试时长60分钟，要循环2次。"
LOG_FILTER_ALL = "总的"
LOG_CATEGORY_SYSTEM = "系统日志"
LOG_CATEGORY_TIME = "时间日志"
LOG_CATEGORY_LOGIC = "逻辑日志"
LOG_CATEGORY_UI = "UI和控点日志"
LOG_CATEGORY_OTHER = "其他日志"
LOG_FILTERS = (
    LOG_FILTER_ALL,
    LOG_CATEGORY_SYSTEM,
    LOG_CATEGORY_TIME,
    LOG_CATEGORY_LOGIC,
    LOG_CATEGORY_UI,
    LOG_CATEGORY_OTHER,
)
LOG_CATEGORIES = set(LOG_FILTERS) - {LOG_FILTER_ALL}


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


def filter_output_text(text: str, selected_filter: str) -> str:
    if selected_filter == LOG_FILTER_ALL:
        return text

    return "".join(
        line
        for line in str(text or "").splitlines(keepends=True)
        if classify_output_line(line) == selected_filter
    )


class CaptureStreamCheckResult(NamedTuple):
    ok: bool
    message: str


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOGGER.handlers:
        return

    LOGGER.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [pid=%(process)d] %(message)s"
    )

    file_handler = logging.FileHandler(LAUNCHER_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    LOGGER.propagate = False
    LOGGER.info("launcher logging initialized, log_file=%s", LAUNCHER_LOG_FILE)


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

        if target_name not in {"project_case", "target_case"} or value_node is None:
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


def is_pubg_testcase_keyword_match(*values) -> bool:
    text = "\n".join(str(value or "") for value in values)
    lower_text = text.lower()
    return any(keyword in text or keyword in lower_text for keyword in PUBG_CASE_KEYWORDS)


def _read_text_for_keyword_match(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        if Path(path).exists():
            return Path(path).read_text(encoding="utf-8")
    except Exception:
        LOGGER.debug("read testcase text for keyword match failed: %s", path, exc_info=True)
    return ""


def is_pubg_testcase_file(py_file: Optional[Path], parsed: Optional[dict] = None) -> bool:
    parsed = parsed or {}
    values = [
        py_file,
        py_file.name if py_file else "",
        py_file.parent.name if py_file else "",
        parsed.get("project_case", ""),
        parsed.get("target_case", ""),
        _read_text_for_keyword_match(py_file),
    ]
    return is_pubg_testcase_keyword_match(*values)


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


def resolve_screen_mode_for_test_profile(test_profile: str) -> str:
    profile = str(test_profile or "").strip().lower()
    return TEST_PROFILE_SCREEN_MODES.get(profile, TEST_PROFILE_SCREEN_MODES["power"])


def resolve_test_profile_from_radio_selection(power_checked: bool, function_checked: bool) -> str:
    if function_checked:
        return "function"
    return "power"


def write_screen_mode_config(screen_mode: str, config_path: Path = AUTOGAME_CONFIG_FILE) -> None:
    config_path = Path(config_path)
    screen_mode = str(screen_mode).strip()
    if screen_mode not in {"0", "1", "2"}:
        raise ValueError(f"unsupported screen_mode: {screen_mode}")

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


def build_launcher_plan_env_values(plan: Optional[dict]) -> dict[str, str]:
    plan = plan or {}
    test_profile = str(plan.get("test_profile") or "power")
    screen_mode = str(plan.get("screen_mode") or resolve_screen_mode_for_test_profile(test_profile))
    case_loop_count = int(plan.get("case_loop_count") or 1)
    return {
        "AUTOGAME_TEST_PROFILE": test_profile,
        "AUTOGAME_SCREEN_MODE": screen_mode,
        "AUTOGAME_SINGLE_CASE_LOOPS": str(max(1, case_loop_count)),
        "AUTOGAME_SP_RECORDING_ENABLED": "1" if should_use_sp_recording_for_profile(test_profile) else "0",
        "AUTOGAME_LOG_DIR": str(LOG_DIR),
        "AUTOGAME_PREVIEW_DIR": str(PREVIEW_DIR),
        "AUTOGAME_SAVE_FRAMES_DIR": str(LOG_DIR / "process_save_frames"),
        "AUTOGAME_TMP_FRAMES_DIR": str(TEMP_DIR / "tmp_frames"),
    }


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
        log_dir: Path = LOG_DIR,
        os_name: Optional[str] = None,
        root_pid: Optional[int] = None,
    ):
        self.log_dir = Path(log_dir)
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

    foreach ($proc in Get-AutoGameProcesses) {{
        $pidValue = [int]$proc.ProcessId
        if (-not $seen.ContainsKey($pidValue)) {{
            $seen[$pidValue] = $true
            Write-AutoGameProcessLine "POLL_CREATE" $pidValue ([int]$proc.ParentProcessId) ([string]$proc.Name)
        }}
    }}

    Start-Sleep -Milliseconds 200
}}
"""


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

        if os.name == "nt":
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
    if (path / "process_temp_logs").is_dir():
        return True
    if (path / "process_save_frames").is_dir():
        return True
    if (path / "preview_10fps.mp4").is_file():
        return True
    return False


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
        if not batch_dir.is_dir() or not batch_dir.name.startswith("game_cases_"):
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
        logs_dir = archive_dir / "logs"
        launcher_output = _read_history_text(logs_dir / "launcher_output.txt")
        if not launcher_output:
            launcher_output = _read_history_text(logs_dir / "launcher_output_partial.txt")

        preview_video = archive_dir / "preview_10fps.mp4"
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
            "stream_disconnected": metadata.get("stream_disconnected", ""),
            "stream_disconnect_startup": metadata.get("stream_disconnect_startup", ""),
            "archive_metadata": metadata,
            "launcher_output": launcher_output,
            "log_file_count": _count_files(logs_dir),
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
    def value(name: str, fallback: str = "-") -> str:
        current = record.get(name)
        if current is None or current == "":
            return fallback
        return str(current)

    archive_dir = record.get("archive_dir")
    preview_text = "有" if record.get("preview_video_exists") else "无"
    lines = [
        f"archive_time: {value('archive_time')}",
        f"run_index: {value('run_index')}",
        f"mode: {value('mode')}",
        f"project_case: {value('project_case')}",
        f"target_case: {value('target_case')}",
        f"testcase_label: {value('testcase_label')}",
        f"exit_code: {value('exit_code')}",
        f"timed_out: {value('timed_out')}",
        f"stream_disconnected: {value('stream_disconnected')}",
        f"stream_disconnect_startup: {value('stream_disconnect_startup')}",
        f"log_file_count: {value('log_file_count', '0')}",
        f"process_temp_file_count: {value('process_temp_file_count', '0')}",
        f"process_save_frame_count: {value('process_save_frame_count', '0')}",
        f"frame_log_count: {value('frame_log_count', '0')}",
        f"preview_10fps.mp4: {preview_text}",
        f"archive_dir: {archive_dir}",
    ]
    return "\n".join(lines)


def _preview_frame_sequence(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.stem)
    if not match:
        return -1
    return int(match.group(1))


def _preview_frame_sort_key(path: Path):
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = -1.0
    return (mtime, _preview_frame_sequence(path), path.name)


def find_latest_preview_frame(preview_dir: Path) -> Optional[Path]:
    if not preview_dir.exists():
        return None

    candidates = []
    for path in preview_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in PREVIEW_FRAME_SUFFIXES:
            continue
        if not path.stem.startswith("frame_"):
            continue
        candidates.append(path)

    if not candidates:
        return None
    return max(candidates, key=_preview_frame_sort_key)


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


def _format_history_log_entries(entries: list[dict]) -> list[str]:
    lines = []
    if not isinstance(entries, list):
        return lines
    for entry in entries[-8:]:
        if not isinstance(entry, dict):
            continue
        message = str(entry.get("message") or entry.get("raw_message") or "").strip()
        if message:
            lines.append(f"- {message}")
            continue
        observation = str(entry.get("observation") or "").strip()
        action = str(entry.get("action") or "").strip()
        if observation or action:
            lines.append(f"- {observation} {action}".strip())
    return lines


def _format_semantic_values(values: dict, limit: int = 24) -> list[str]:
    lines = []
    if not isinstance(values, dict):
        return lines
    for key, value in list(values.items())[:limit]:
        lines.append(f"- {key}: {value}")
    return lines


def _format_semantic_actions(actions: list[dict]) -> list[str]:
    lines = []
    if not isinstance(actions, list) or not actions:
        return ["- 暂无控制动作"]
    for action in actions[-12:]:
        if not isinstance(action, dict):
            continue
        name = action.get("action") or action.get("name") or "-"
        target = action.get("target") or "-"
        control_point = action.get("control_point") or target
        actual_pos = action.get("actual_pos") or "-"
        start_pos = action.get("start_pos") or "-"
        end_pos = action.get("end_pos") or "-"
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        duration = action.get("duration") or params.get("dura") or params.get("duration") or "-"
        reason = action.get("reason") or "-"
        params_text = ", ".join(f"{key}={value}" for key, value in params.items()) or "-"
        lines.append(
            f"- action={name}; control_point={control_point}; target={target}; "
            f"actual_pos={actual_pos}; start={start_pos}; end={end_pos}; "
            f"duration={duration}; params={params_text}; reason={reason}"
        )
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
    summary = str(payload.get("frame_summary") or "").strip()
    seen_summary = str(seen.get("summary") or "").strip()
    if not seen_summary and info_payload:
        seen_summary = "看到 " + ", ".join(str(key) for key in info_payload.keys())

    lines = [
        f"帧: {frame_info.get('image') or Path(str(frame_record.get('image_path') or '')).name}",
        f"序号: {frame_info.get('index', frame_record.get('index', '-'))}",
    ]

    if summary:
        lines.extend(["", "帧摘要", f"- {summary}"])

    semantic_stage = semantic_log.get("current_stage") if isinstance(semantic_log.get("current_stage"), dict) else {}
    semantic_perception = semantic_log.get("perception") if isinstance(semantic_log.get("perception"), dict) else {}
    semantic_judgment = semantic_log.get("judgment") if isinstance(semantic_log.get("judgment"), dict) else {}
    semantic_branch = semantic_log.get("branch") if isinstance(semantic_log.get("branch"), dict) else {}
    semantic_actions = semantic_log.get("actions") if isinstance(semantic_log.get("actions"), list) else []

    lines.extend([
        "",
        "当前阶段",
        f"- stage: {semantic_stage.get('stage') or stage_name}",
        f"- scene/子状态: {semantic_stage.get('scene_state') or '-'}",
        f"- group: {semantic_stage.get('group') or stage_group}",
        f"- frame_index: {semantic_stage.get('frame_index', frame_info.get('index', frame_record.get('index', '-')))}",
        "",
        "当前感知结果",
        f"- {semantic_perception.get('summary') or seen_summary or '-'}",
    ])
    semantic_values = _format_semantic_values(semantic_perception.get("critical_values"))
    if semantic_values:
        lines.extend(semantic_values)

    lines.extend([
        "",
        "当前判断",
        f"- reason: {semantic_judgment.get('reason') or decision_payload.get('observation') or code_branch.get('observation') or '-'}",
        f"- decision: {semantic_judgment.get('decision') or decision_payload.get('decision') or code_branch.get('action') or next_action}",
        f"- evidence: {semantic_judgment.get('evidence') or decision_payload.get('method') or code_branch.get('method') or '-'}",
        f"- result: {semantic_judgment.get('result_expectation') or decision_payload.get('result') or code_branch.get('result') or '-'}",
        "",
        "当前分支",
        f"- branch: {semantic_branch.get('name') or code_branch.get('target') or decision_payload.get('target') or '-'}",
        f"- observation: {semantic_branch.get('observation') or code_branch.get('observation') or decision_payload.get('observation') or '-'}",
        f"- action: {semantic_branch.get('action') or code_branch.get('action') or decision_payload.get('action') or '-'}",
        f"- method: {semantic_branch.get('method') or code_branch.get('method') or decision_payload.get('method') or '-'}",
        f"- result: {semantic_branch.get('result') or code_branch.get('result') or decision_payload.get('result') or '-'}",
        "",
        "当前动作",
        *_format_semantic_actions(semantic_actions),
    ])

    lines.extend([
        "",
        "阶段信息",
        f"- 阶段: {stage_name}",
        f"- 分组: {stage_group}",
        "",
        "这一帧看到了",
        f"- {seen_summary or '-'}",
    ])

    if decision_payload:
        lines.extend([
            "",
            "本帧决策",
            f"- 观察: {decision_payload.get('observation') or '-'}",
            f"- 决策: {decision_payload.get('decision') or decision_payload.get('action') or '-'}",
            f"- 控制: {decision_payload.get('method') or '-'}",
            f"- 结果: {decision_payload.get('result') or '-'}",
        ])

    if info_payload:
        lines.append("- info:")
        for key, value in list(info_payload.items())[:80]:
            lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "代码分支",
        f"- 目标/分支: {code_branch.get('target') or '-'}",
        f"- 观察: {code_branch.get('observation') or '-'}",
        f"- 决策: {code_branch.get('action') or '-'}",
        f"- 控制: {code_branch.get('method') or '-'}",
        f"- 结果: {code_branch.get('result') or '-'}",
        "",
        "下一步",
        f"- {next_action}",
    ])

    time_lines = _format_history_log_entries(payload.get("time_logs", []))
    if time_lines:
        lines.extend(["", "时间日志", *time_lines])

    logic_lines = _format_history_log_entries(payload.get("logic_logs", []))
    if logic_lines:
        lines.extend(["", "逻辑日志", *logic_lines])

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


def run_testcase_entry(testcase_label: str):
    install_hidden_subprocess_patch()
    start_hidden_subprocess_window_suppressor()
    LOGGER.info("run_testcase_entry: testcase_label=%s", testcase_label)
    from xdevice.__main__ import main_process

    with hidden_subprocess_context(
        target_executables=("icpm_xdc.exe", "hdc.exe", "hdc"),
        hide_all=True,
    ):
        main_process(f"run -l {testcase_label}")


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

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
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


class LauncherWindow(QWidget):
    restart_phone_script_finished = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        LOGGER.info("LauncherWindow init start")
        self.process: Optional[QProcess] = None
        self.restart_phone_script_running = False
        self.selected_testcase_file: Optional[Path] = None
        self._updating_targets = False
        self.latest_preview_file: Optional[Path] = None
        self.latest_preview_pixmap: Optional[QPixmap] = None
        self.latest_preview_payload: Optional[dict] = None
        self.stage_info_cache: dict[str, dict] = {}
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(150)
        self.safety_timer = QTimer(self)
        self.safety_timer.setInterval(5000)
        self.run_timeout_timer = QTimer(self)
        self.run_timeout_timer.setSingleShot(True)
        self.stream_disconnect_signal_timer = QTimer(self)
        self.stream_disconnect_signal_timer.setInterval(500)
        self.restart_phone_script_finished.connect(self._finish_restart_phone_script)

        self.batch_active = False
        self.stop_requested = False
        self.current_run_index = 0
        self.total_runs = 1
        self.current_plan: Optional[dict] = None
        self.current_run_timed_out = False
        self.current_run_output_start = 0
        self.current_run_stream_started = False
        self.current_run_stream_disconnected = False
        self.current_run_stream_disconnect_startup = False
        self.current_run_stream_disconnect_message = ""
        self.current_run_stream_preserved = False
        self.current_run_sp_started = False
        self.current_run_sp_state: dict = {}
        self.dismiss_reboot_prompt_on_next_case_start = False
        self.preserve_device_apps_on_manual_stop = True
        self.current_batch_start_timestamp: Optional[str] = None
        self.current_run_start_timestamp: Optional[str] = None
        self.current_run_archive_dir: Optional[Path] = None
        self.preview_target_info_height = 90
        self.preview_target_info_width = 360
        self._adjusting_preview_splitter = False
        self.preset_buttons: list[QPushButton] = []
        self.theme_mode = "light"
        self.inputs_enabled = True
        self.label_tool = None
        self.label_tool_project_dir: Optional[Path] = None
        self.history_records: list[dict] = []
        self.selected_history_record: Optional[dict] = None
        self.history_frame_records: list[dict] = []
        self.history_frame_index = -1
        self.process_launch_tracer = WindowsProcessLaunchTracer(LOG_DIR)

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
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.setToolTip("刷新配置")
        self.refresh_button.setFixedWidth(64)

        self.project_combo = QComboBox()
        self.target_combo = QComboBox()

        self.status_label = QLabel("请选择启动方式，并选择 testcases 用例或直接指定配置。")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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
        self.test_profile_layout.setSpacing(10)
        self.power_test_radio = QRadioButton("功耗测试")
        self.function_test_radio = QRadioButton("功能测试")
        self.power_test_radio.setChecked(True)
        self.test_profile_button_group = QButtonGroup(self)
        self.test_profile_button_group.setExclusive(True)
        self.test_profile_button_group.addButton(self.power_test_radio)
        self.test_profile_button_group.addButton(self.function_test_radio)
        self.test_profile_layout.addWidget(self.power_test_radio)
        self.test_profile_layout.addWidget(self.function_test_radio)
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
        self.safe_temp_spin.setValue(40.0)
        self.safe_temp_spin.setSuffix(" °C")
        self.safe_temp_field = self._create_spin_with_presets(
            self.safe_temp_spin,
            [35, 38, 40, 42, 45],
            suffix="°C",
        )

        self.safe_battery_spin = QSpinBox()
        self.safe_battery_spin.setRange(0, 100)
        self.safe_battery_spin.setValue(25)
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

        self.start_button = QPushButton("启动")
        self.start_button.setProperty("primaryButton", True)
        self.restart_phone_button = QPushButton("重启手机")
        self.restart_phone_button.setToolTip("执行 launcher 同目录下的 restart.bat 重启手机")
        self.stop_button = QPushButton("停止")
        self.stop_button.setProperty("dangerButton", True)
        self.stop_button.setEnabled(False)
        self.open_history_button = QPushButton("历史输出")
        self.keep_process_on_manual_stop_button = QPushButton("停止保活")
        self.keep_process_on_manual_stop_button.setCheckable(True)
        self.keep_process_on_manual_stop_button.setChecked(False)
        self.keep_process_on_manual_stop_button.setProperty("toggleButton", True)
        self.generate_preview_video_button = QPushButton("生成视频：关")
        self.generate_preview_video_button.setObjectName("generatePreviewVideoButton")
        self.generate_preview_video_button.setCheckable(True)
        self.generate_preview_video_button.setChecked(False)
        self.generate_preview_video_button.setProperty("toggleButton", True)
        self.generate_preview_video_button.setToolTip("关闭时只归档日志和图片，不生成预览视频")
        self.preview_overlay_button = QPushButton("显示标注")
        self.preview_overlay_button.setCheckable(True)
        self.preview_overlay_button.setChecked(False)
        self.preview_overlay_button.setProperty("toggleButton", True)
        self.preview_points_button = QPushButton("显示控点")
        self.preview_points_button.setCheckable(True)
        self.preview_points_button.setChecked(False)
        self.preview_points_button.setProperty("toggleButton", True)

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
        self.output_log_filter = LOG_FILTER_ALL
        self.output_log_entries: list[tuple[str, str]] = []
        self.output_filter_button_group = QButtonGroup(self)
        self.output_filter_button_group.setExclusive(True)
        self.output_filter_buttons: dict[str, QPushButton] = {}
        for filter_name in LOG_FILTERS:
            button = QPushButton(filter_name)
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

        self.preview_info_edit = QPlainTextEdit()
        self.preview_info_edit.setObjectName("previewInfo")
        self.preview_info_edit.setReadOnly(True)
        self.preview_info_edit.setPlaceholderText("当前帧识别信息会显示在这里...")
        self.preview_info_edit.setMinimumHeight(50)
        self.preview_info_edit.setMinimumWidth(280)

        self._build_ui()
        self._bind_signals()
        self._load_project_cases()
        self._sync_mode_ui()
        self._log_message(
            f"[Launcher] 启动器已初始化，日志文件：{LAUNCHER_LOG_FILE}\n",
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
                QLabel#statusLabel,
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
                QPlainTextEdit#outputConsole,
                QPlainTextEdit#previewInfo {
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
            QLabel#statusLabel,
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
            QPlainTextEdit#outputConsole,
            QPlainTextEdit#previewInfo {
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
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        spin.setMinimumWidth(96)
        layout.addWidget(spin)

        for value in values:
            text = f"{value}{suffix}" if suffix else str(value)
            button = QPushButton(text)
            button.setProperty("presetButton", True)
            button.clicked.connect(lambda checked=False, val=value, target=spin: target.setValue(val))
            self.preset_buttons.append(button)
            layout.addWidget(button)

        layout.addStretch(1)
        return container

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
        controls_widget.setFixedHeight(420)
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
        controls_layout.addLayout(launch_row)

        config_group = QGroupBox("配置")
        config_group.setFixedHeight(290)
        config_layout = QGridLayout(config_group)
        config_layout.setContentsMargins(12, 8, 12, 10)
        config_layout.setHorizontalSpacing(12)
        config_layout.setVerticalSpacing(8)

        def add_config_item(row: int, column: int, label_text: str, widget: QWidget):
            label = QLabel(label_text)
            label.setObjectName("formLabel")
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            config_layout.addWidget(label, row, column * 2)
            config_layout.addWidget(widget, row, column * 2 + 1)

        add_config_item(0, 0, "project_case", self.project_combo)
        add_config_item(0, 1, "target_case", self.target_combo)
        add_config_item(1, 0, "运行次数", self.run_count_field)
        add_config_item(1, 1, "测试类型", self.test_profile_field)
        add_config_item(2, 0, "单次循环", self.case_loop_count_field)
        add_config_item(2, 1, "安全温度", self.safe_temp_field)
        add_config_item(3, 0, "安全电量", self.safe_battery_field)
        add_config_item(3, 1, "安全时间", self.safe_time_field)
        add_config_item(4, 0, "无操控超时", self.inactivity_timeout_field)
        add_config_item(4, 1, "视频归档", self.generate_preview_video_button)

        config_layout.setColumnStretch(1, 1)
        config_layout.setColumnStretch(3, 1)
        controls_layout.addWidget(config_group)

        main_layout.addWidget(controls_widget, 0)

        action_bar = QWidget()
        action_bar.setObjectName("actionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(12, 9, 12, 9)
        action_layout.setSpacing(8)
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.stop_button)
        action_layout.addWidget(self.open_history_button)
        action_layout.addWidget(self.keep_process_on_manual_stop_button)
        action_layout.addWidget(self.preview_overlay_button)
        action_layout.addWidget(self.preview_points_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self.restart_phone_button)
        main_layout.addWidget(action_bar, 0)

        status_strip = QWidget()
        status_strip.setObjectName("statusStrip")
        status_layout = QHBoxLayout(status_strip)
        status_layout.setContentsMargins(12, 9, 12, 9)
        status_layout.setSpacing(10)
        status_layout.addWidget(self.status_label, 1)
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
        self.preview_splitter.addWidget(self.preview_info_edit)
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
        log_filter_layout.addWidget(QLabel("显示"))
        for filter_name in LOG_FILTERS:
            log_filter_layout.addWidget(self.output_filter_buttons[filter_name])
        log_filter_layout.addStretch(1)
        log_layout.addLayout(log_filter_layout)
        log_layout.addWidget(self.output_edit)
        content_splitter.addWidget(preview_group)
        content_splitter.addWidget(log_group)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)

        main_layout.addWidget(content_splitter, 1)
        self.page_stack.addWidget(self.launcher_page)
        self.label_tool_page = self._build_label_tool_page()
        self.page_stack.addWidget(self.label_tool_page)
        self.history_page = self._build_history_page()
        self.page_stack.addWidget(self.history_page)
        self._update_header_badges()

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

        self.back_to_launcher_button = QPushButton("返回启动器")
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
        self.history_frame_log_edit.setPlaceholderText("选择历史输出后显示这一帧的阶段、info、时间日志和代码分支...")
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
        self.browse_button.clicked.connect(self._choose_testcase_file)
        self.clear_button.clicked.connect(self._reselect_testcase_file)
        self.open_label_tool_button.clicked.connect(self._open_label_tool_for_selected_case)
        self.back_to_launcher_button.clicked.connect(self._show_launcher_page)
        self.refresh_button.clicked.connect(self._refresh_config_choices)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        self.start_button.clicked.connect(self._start_run)
        self.restart_phone_button.clicked.connect(self._restart_phone_from_button)
        self.stop_button.clicked.connect(self._stop_run)
        self.open_history_button.clicked.connect(self._show_history_page)
        self.history_refresh_button.clicked.connect(self._refresh_history_outputs)
        self.history_open_dir_button.clicked.connect(self._open_selected_history_dir)
        self.history_delete_button.clicked.connect(self._delete_selected_history_output)
        self.history_back_button.clicked.connect(self._show_launcher_page)
        self.history_tree.itemSelectionChanged.connect(self._on_history_selection_changed)
        self.history_prev_frame_button.clicked.connect(self._show_previous_history_frame)
        self.history_next_frame_button.clicked.connect(self._show_next_history_frame)
        self.keep_process_on_manual_stop_button.toggled.connect(self._toggle_keep_process_on_manual_stop)
        self.generate_preview_video_button.toggled.connect(self._toggle_generate_preview_video)
        self.preview_overlay_button.toggled.connect(self._toggle_preview_overlay)
        self.preview_points_button.toggled.connect(self._toggle_preview_points)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        for filter_name, button in self.output_filter_buttons.items():
            button.clicked.connect(lambda checked=False, name=filter_name: self._set_output_log_filter(name))
        self.preview_timer.timeout.connect(self._poll_preview_frame)
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
        QApplication.processEvents()

    def _record_output_text(self, text: str):
        for line in str(text or "").splitlines(keepends=True):
            self.output_log_entries.append((classify_output_line(line), line))

    def _all_output_text(self) -> str:
        return "".join(line for _, line in self.output_log_entries)

    def _filtered_output_text(self) -> str:
        if self.output_log_filter == LOG_FILTER_ALL:
            return self._all_output_text()
        return "".join(
            line
            for category, line in self.output_log_entries
            if category == self.output_log_filter
        )

    def _render_output_filter(self):
        self.output_edit.setPlainText(self._filtered_output_text())
        self.output_edit.moveCursor(QTextCursor.MoveOperation.End)
        QApplication.processEvents()

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

        self._record_output_text(text)
        visible_text = filter_output_text(text, self.output_log_filter)
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

    def _set_status(self, text: str):
        self.status_label.setText(text)
        self._update_header_badges()

    def _set_runtime(self, text: str):
        self.runtime_label.setText(text)
        self._update_header_badges()

    def _toggle_preview_overlay(self, checked: bool):
        self.preview_overlay_button.setText("隐藏标注" if checked else "显示标注")
        LOGGER.info("preview overlay toggled: %s", checked)
        self._refresh_preview_pixmap()

    def _toggle_preview_points(self, checked: bool):
        self.preview_points_button.setText("隐藏控点" if checked else "显示控点")
        LOGGER.info("preview points toggled: %s", checked)
        self._refresh_preview_pixmap()

    def _on_theme_changed(self):
        theme_mode = self.theme_combo.currentData() or "light"
        if theme_mode == self.theme_mode:
            return
        self.theme_mode = str(theme_mode)
        LOGGER.info("launcher theme changed: %s", self.theme_mode)
        self._apply_style()
        self._refresh_preview_pixmap()

    def _toggle_keep_process_on_manual_stop(self, checked: bool):
        self.keep_process_on_manual_stop_button.setText(
            "停止保活: 开" if checked else "停止保活"
        )
        LOGGER.info("keep process on manual stop toggled: %s", checked)

    def _toggle_generate_preview_video(self, checked: bool):
        self.generate_preview_video_button.setText("生成视频：开" if checked else "生成视频：关")
        LOGGER.info("generate preview video toggled: %s", checked)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adjust_preview_splitter_sizes()
        self._refresh_preview_pixmap()

    def _sync_mode_ui(self):
        LOGGER.debug("sync_mode_ui: testcase_mode=%s", self.mode_testcase.isChecked())
        self._sync_testcase_controls_state()

    def _can_open_label_tool_for_selection(self) -> bool:
        if self.selected_testcase_file is None:
            return False
        project_case = self.project_combo.currentText().strip()
        return resolve_label_project_dir(project_case) is not None

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
        self._refresh_preview_pixmap()

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
            QMessageBox.warning(self, "路径错误", "请选择当前项目 testcases 目录下的用例文件。")
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
        self.testcase_path_edit.clear()
        self._sync_testcase_controls_state()
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

    def _open_label_tool_for_selected_case(self):
        LOGGER.info(
            "open_label_tool_for_selected_case: testcase=%s project=%s",
            self.selected_testcase_file,
            self.project_combo.currentText().strip(),
        )
        if self.selected_testcase_file is None:
            QMessageBox.warning(self, "缺少用例", "请先选择一个 testcases 用例。")
            return

        project_case = self.project_combo.currentText().strip()
        project_dir = resolve_label_project_dir(project_case)
        if project_dir is None:
            QMessageBox.warning(
                self,
                "缺少标注资源",
                f"未找到 project_case={project_case} 对应的标注资源目录或 info.py。",
            )
            self._sync_testcase_controls_state()
            return

        try:
            self._ensure_label_tool()
            self.label_tool.load_project_from_dir(str(project_dir))
        except Exception as exc:
            log_exception(f"open label tool failed: project_dir={project_dir}")
            QMessageBox.critical(self, "打开失败", f"无法打开标注工具：\n{exc}")
            return

        self.label_tool_project_dir = project_dir
        self.label_tool_project_label.setText(
            f"当前标注项目：{project_case}    {project_dir}"
        )
        self.page_stack.setCurrentWidget(self.label_tool_page)
        self._set_status(f"已打开标注工具：{project_case}")

    def _show_launcher_page(self):
        LOGGER.info("show launcher page")
        project = self.project_combo.currentText().strip()
        target = self.target_combo.currentText().strip()
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
        has_record = record is not None
        self.history_open_dir_button.setEnabled(has_record)
        self.history_delete_button.setEnabled(has_record)
        if not has_record:
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
            self.history_frame_log_edit.setPlainText("未找到逐帧 JSON。请确认本次运行已生成 process_temp_logs/frame_*.json。")
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
            self._set_selected_history_record(None)
            return

        index = items[0].data(0, Qt.ItemDataRole.UserRole)
        if index is None:
            self._set_selected_history_record(None)
            return
        try:
            record = self.history_records[int(index)]
        except (IndexError, TypeError, ValueError):
            record = None
        self._set_selected_history_record(record)

    def _open_selected_history_dir(self):
        if not self.selected_history_record:
            return
        archive_dir = Path(self.selected_history_record["archive_dir"])
        if not archive_dir.exists():
            QMessageBox.warning(self, "目录不存在", f"历史输出目录不存在：\n{archive_dir}")
            self._refresh_history_outputs()
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(archive_dir.resolve())))

    def _delete_selected_history_output(self):
        if not self.selected_history_record:
            return
        archive_dir = Path(self.selected_history_record["archive_dir"]).resolve()
        history_temp_dir = resolve_history_temp_dir().resolve()
        try:
            archive_dir.relative_to(history_temp_dir)
        except ValueError:
            QMessageBox.warning(self, "拒绝删除", f"只能删除 temp 目录下的历史输出：\n{archive_dir}")
            return

        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除这条历史输出吗？\n\n{archive_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            shutil.rmtree(archive_dir)
        except Exception as exc:
            log_exception(f"delete history output failed: archive_dir={archive_dir}")
            QMessageBox.critical(self, "删除失败", f"删除失败：\n{exc}")
            return

        self._set_status(f"已删除历史输出：{archive_dir.name}")
        self._refresh_history_outputs()

    def _apply_parsed_testcase(self, py_file: Path):
        LOGGER.info("apply_parsed_testcase: %s", py_file)
        try:
            parsed = parse_case_vars(py_file)
        except Exception as exc:
            log_exception(f"apply_parsed_testcase failed: file={py_file}")
            self._set_status(f"解析失败：{exc}")
            return

        project_case = parsed.get("project_case")
        target_case = parsed.get("target_case")

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
            messages.append(PUBG_CASE_RUNTIME_DESCRIPTION)

        self._sync_testcase_controls_state()
        self._set_status("；".join(messages))

    def _build_process_environment(self, project_case: str, target_case: str, run_no: int) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        apply_pyinstaller_splash_suppression(env)
        env.insert("TARGET_PROJECT_CASE", project_case)
        env.insert("TARGET_GAME_CASE", target_case)
        env.insert("AUTOGAME_VIS_MODE", "launcher")
        env.insert("AUTOGAME_RUN_SOURCE", "launcher")
        env.insert("AUTOGAME_RUN_INDEX", str(int(run_no)))
        env.insert("AUTOGAME_DEVICE_LOG_PATH", str(LOG_DIR / f"{target_case}.txt"))
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
            env.insert("AUTOGAME_RUN_ARCHIVE_DIR", str(run_archive_dir))
        except Exception:
            log_exception(f"resolve run archive dir failed: run_no={run_no}")
        if self.current_plan is not None:
            env.insert(
                "AUTOGAME_LAUNCHER_INACTIVITY_TIMEOUT_MINUTES",
                str(float(self.current_plan.get("inactivity_timeout_minutes", 5.0))),
            )
        LOGGER.debug(
            "build_process_environment: project_case=%s target_case=%s run_no=%s batch_start=%s run_start=%s inactivity_timeout=%s",
            project_case,
            target_case,
            run_no,
            self.current_batch_start_timestamp,
            self.current_run_start_timestamp,
            self.current_plan.get("inactivity_timeout_minutes") if self.current_plan else None,
        )
        return env

    def _set_inputs_enabled(self, enabled: bool):
        self.inputs_enabled = enabled
        self.mode_testcase.setEnabled(enabled)
        self.mode_direct.setEnabled(enabled)
        self.refresh_button.setEnabled(enabled)
        self.open_history_button.setEnabled(enabled)
        self.restart_phone_button.setEnabled(enabled and not self.restart_phone_script_running)
        self.project_combo.setEnabled(enabled)
        self.target_combo.setEnabled(enabled)
        self.run_count_spin.setEnabled(enabled)
        self.test_profile_field.setEnabled(enabled)
        self.case_loop_count_spin.setEnabled(enabled)
        self.safe_temp_spin.setEnabled(enabled)
        self.safe_battery_spin.setEnabled(enabled)
        self.safe_time_spin.setEnabled(enabled)
        self.inactivity_timeout_spin.setEnabled(enabled)
        self.generate_preview_video_button.setEnabled(enabled)
        for button in self.preset_buttons:
            button.setEnabled(enabled)
        self._sync_testcase_controls_state()

    def _clear_preview_files(self):
        LOGGER.info("preview frame dir=%s", PREVIEW_DIR)
        LOGGER.debug("clear_preview_files: dir=%s", PREVIEW_DIR)
        self.latest_preview_file = None
        self.latest_preview_pixmap = None
        self.latest_preview_payload = None
        self.preview_image_label.setText("启动后将在这里实时显示可视化帧")
        self.preview_image_label.setPixmap(QPixmap())
        self.preview_info_edit.clear()

        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        for path in PREVIEW_DIR.iterdir():
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

            min_info_width = max(self.preview_info_edit.minimumWidth(), 280)
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

            min_info_height = max(self.preview_info_edit.minimumHeight(), 150)
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
        display_pixmap = self._build_preview_display_pixmap()
        if display_pixmap.isNull():
            LOGGER.warning(
                "render preview fail: display pixmap is null latest_frame=%s",
                self.latest_preview_file,
            )
            return False
        self._adjust_preview_splitter_sizes()
        scaled = display_pixmap.scaled(
            self.preview_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if scaled.isNull():
            LOGGER.warning(
                "render preview fail: scaled pixmap is null latest_frame=%s label_size=%s",
                self.latest_preview_file,
                self.preview_image_label.size(),
            )
            return False
        self.preview_image_label.setPixmap(scaled)
        LOGGER.debug(
            "render preview success: latest_frame=%s label_size=%s scaled_size=%s",
            self.latest_preview_file,
            self.preview_image_label.size(),
            scaled.size(),
        )
        return True

    def _get_preview_project_case(self) -> str:
        if self.current_plan is not None:
            return str(self.current_plan.get("project_case") or "").strip()
        return self.project_combo.currentText().strip()

    def _load_stage_info(self, project_case: str) -> dict:
        if not project_case:
            return {}
        if project_case in self.stage_info_cache:
            return self.stage_info_cache[project_case]

        try:
            module = importlib.import_module(
                f"aw.autogame.customs_examples.{project_case}.info"
            )
            stage_info = getattr(module, "STAGE_INFO", {})
        except Exception:
            log_exception(f"load stage info failed: project_case={project_case}")
            stage_info = {}

        if not isinstance(stage_info, dict):
            stage_info = {}
        self.stage_info_cache[project_case] = stage_info
        return stage_info

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
        painter.setPen(pen)
        painter.drawRect(x1, y1, width, height)
        painter.fillRect(x1, y1, width, height, QColor(color.red(), color.green(), color.blue(), 35))
        painter.drawText(x1 + 4, max(14, y1 + 16), label)

    def _build_preview_display_pixmap(self) -> QPixmap:
        if self.latest_preview_pixmap is None:
            return QPixmap()
        show_overlay = self.preview_overlay_button.isChecked()
        show_points = self.preview_points_button.isChecked()
        if not show_overlay and not show_points:
            return self.latest_preview_pixmap

        payload = self.latest_preview_payload or {}
        stage = payload.get("stage")
        project_case = self._get_preview_project_case()
        stage_info = self._load_stage_info(project_case)
        stage_entry = stage_info.get(stage, {}) if isinstance(stage_info, dict) else {}
        scenes = stage_entry.get("scenes", {}) if isinstance(stage_entry, dict) else {}
        if not scenes:
            return self.latest_preview_pixmap

        pixmap = self.latest_preview_pixmap.copy()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        screen_width, screen_height = get_resolution()

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

        painter.end()
        return pixmap

    def _poll_preview_frame(self):
        LOGGER.debug("preview frame dir=%s exists=%s", PREVIEW_DIR, PREVIEW_DIR.exists())
        if not PREVIEW_DIR.exists():
            return

        latest_image = find_latest_preview_frame(PREVIEW_DIR)
        LOGGER.debug(
            "latest frame path=%s latest frame exists=%s",
            latest_image,
            bool(latest_image and latest_image.exists()),
        )

        if latest_image is None or latest_image == self.latest_preview_file:
            return

        json_path = latest_image.with_suffix(".json")
        LOGGER.info(
            "latest frame path=%s latest frame exists=%s preview frame dir=%s",
            latest_image,
            latest_image.exists(),
            PREVIEW_DIR,
        )

        pixmap = QPixmap(str(latest_image))
        if pixmap.isNull():
            LOGGER.warning("QPixmap load fail: latest_frame=%s", latest_image)
            return
        LOGGER.info(
            "QPixmap load success: latest_frame=%s size=%sx%s",
            latest_image,
            pixmap.width(),
            pixmap.height(),
        )

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
        self._adjust_preview_splitter_sizes()
        render_success = self._refresh_preview_pixmap()
        LOGGER.info(
            "render preview %s: latest_frame=%s",
            "success" if render_success else "fail",
            latest_image,
        )
        self.preview_info_edit.setPlainText(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _validate_selection(self) -> Optional[tuple[str, str]]:
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
            QMessageBox.warning(self, "缺少配置", "请选择 project_case。")
            return None

        if not target_case:
            QMessageBox.warning(self, "缺少配置", "请选择 target_case。")
            return None

        return project_case, target_case

    def _collect_plan(self) -> Optional[dict]:
        config = self._validate_selection()
        if config is None:
            return None

        project_case, target_case = config
        testcase_label = None
        mode = "direct"

        if self.mode_testcase.isChecked():
            if self.selected_testcase_file is None:
                QMessageBox.warning(self, "缺少用例", "testcases 模式下请先选择一个用例文件。")
                return None
            testcase_label = self.selected_testcase_file.relative_to(APP_DIR).with_suffix("").as_posix()
            mode = "testcase"

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
        )
        if not should_use_sp_recording_for_profile(test_profile):
            cleanup_apps.discard(DEFAULT_SP_PACKAGE)
        screen_mode = resolve_screen_mode_for_test_profile(test_profile)
        runtime_description = ""
        if is_pubg_testcase_keyword_match(testcase_label, project_case, target_case):
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
            "inactivity_timeout_minutes": float(self.inactivity_timeout_spin.value()),
            "generate_preview_video": bool(self.generate_preview_video_button.isChecked()),
            "cleanup_apps": sorted(cleanup_apps),
            "runtime_description": runtime_description,
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
        self.current_batch_start_timestamp = time.strftime("%Y%m%d%H%M%S")
        self.current_run_start_timestamp = None
        self.batch_active = True
        self.stop_requested = False
        self.current_run_index = 0
        self.current_run_timed_out = False
        self.output_log_entries.clear()
        self.output_edit.clear()
        self._clear_preview_files()
        self.start_button.setEnabled(False)
        self.restart_phone_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_inputs_enabled(False)
        self._set_status("已开始批量执行，准备进行安全检查。")
        self._set_runtime(f"运行信息：共 {plan['run_count']} 次，等待第 1 次启动。")
        trace_label = f"{plan['mode']}:{plan['project_case']}:{plan['target_case']}"
        trace_path = self.process_launch_tracer.start(trace_label)
        self._log_message(
            f"[Launcher] 批量运行开始，mode={plan['mode']}, runs={plan['run_count']}, "
            f"test_profile={plan['test_profile']}, screen_mode={plan['screen_mode']}, "
            f"case_loops={plan['case_loop_count']}, "
            f"generate_preview_video={plan['generate_preview_video']}, "
            f"safe_temp={plan['safe_temp']}°C, safe_battery={plan['safe_battery']}%, "
            f"safe_time={plan['safe_minutes']}分钟, inactivity_timeout={plan['inactivity_timeout_minutes']}分钟, "
            f"cleanup_apps={plan['cleanup_apps']}\n"
        )
        if plan.get("runtime_description"):
            self._log_message(f"[Launcher] {plan['runtime_description']}\n")
        if plan.get("capture_preflight_message"):
            self._log_message(f"[Launcher] 截图流预检：{plan['capture_preflight_message']}\n")
        if trace_path is not None:
            self._log_message(f"[Launcher] 进程创建追踪日志：{trace_path}\n")
        else:
            self._log_message("[Launcher] 当前环境未启用 Windows 进程创建追踪。\n")
        self._cleanup_apps_between_runs("批次启动前预清理")
        self._check_and_start_if_safe()

    def _finish_batch(self, message: str):
        LOGGER.info("finish_batch: %s", message)
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
        self.current_run_sp_state = {}
        self.dismiss_reboot_prompt_on_next_case_start = False
        self.preserve_device_apps_on_manual_stop = True
        self.current_batch_start_timestamp = None
        self.current_run_start_timestamp = None
        self.current_run_archive_dir = None
        self.start_button.setEnabled(True)
        self.restart_phone_button.setEnabled(not self.restart_phone_script_running)
        self.stop_button.setEnabled(False)
        self._set_inputs_enabled(True)
        self.preview_timer.stop()
        self.safety_timer.stop()
        self.run_timeout_timer.stop()
        self.stream_disconnect_signal_timer.stop()
        self._set_status(message)
        self._set_runtime(message)

    def _cleanup_apps_between_runs(self, reason: str):
        if self.current_plan is None:
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
        if self.current_run_index >= self.current_plan["run_count"]:
            self._finish_batch("所有运行次数已完成。")
            return

        run_no = self.current_run_index + 1
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

    def _launch_iteration(self, run_no: int, temperature: float, battery: int):
        if self.current_plan is None:
            return

        LOGGER.info(
            "launch_iteration start: run_no=%s temperature=%s battery=%s plan=%s",
            run_no,
            temperature,
            battery,
            self.current_plan,
        )
        self.current_run_timed_out = False
        self.current_run_stream_started = False
        self.current_run_stream_disconnected = False
        self.current_run_stream_disconnect_startup = False
        self.current_run_stream_disconnect_message = ""
        self.current_run_stream_preserved = False
        self.current_run_sp_started = False
        self.current_run_sp_state = {}
        self.current_run_start_timestamp = time.strftime("%Y%m%d%H%M%S")
        self.current_run_archive_dir = None
        self._clear_preview_files()
        self.current_run_output_start = len(self._all_output_text())

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
            self._set_status(
                f"第 {run_no}/{self.current_plan['run_count']} 次启动：{testcase_label}"
            )
            self._log_message(f"\n[Launcher] 第 {run_no}/{self.current_plan['run_count']} 次：通过 testcase 启动 {testcase_label}\n")
        else:
            args = build_launcher_process_args("--run-direct", project_case, target_case)
            self._set_status(
                f"第 {run_no}/{self.current_plan['run_count']} 次启动：project_case={project_case}, target_case={target_case}"
            )
            self._log_message(
                f"\n[Launcher] 第 {run_no}/{self.current_plan['run_count']} 次：直接启动 "
                f"project_case={project_case}, target_case={target_case}\n"
            )

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
            QMessageBox.critical(self, "启动失败", "子进程启动失败，请检查 Python 环境。")
            self.process.deleteLater()
            self.process = None
            self._finish_batch("启动失败，批量任务已终止。")
            return

        if self.dismiss_reboot_prompt_on_next_case_start:
            self._log_message("[Launcher] 已通知本次用例在打开 sp 后关闭重启弹窗。\n")
            self.dismiss_reboot_prompt_on_next_case_start = False

        self.preview_timer.start()
        safe_minutes = self.current_plan["safe_minutes"]
        if safe_minutes > 0:
            self.run_timeout_timer.start(int(safe_minutes * 60 * 1000))
        if self._stream_disconnect_recovery_enabled():
            self.stream_disconnect_signal_timer.start()
        else:
            self.stream_disconnect_signal_timer.stop()

    def _resolve_current_device_log_path(self) -> Optional[Path]:
        if self.current_plan is None:
            return None
        target_case = str(self.current_plan.get("target_case") or "").strip()
        if not target_case:
            return None
        return LOG_DIR / f"{target_case}.txt"

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

    def _wait_for_device_log_stopped_signal(self) -> Optional[dict]:
        archive_dir = self.current_run_archive_dir
        if archive_dir is None:
            return None

        signal_path = archive_dir / "device_log_state.json"
        deadline = time.time() + DEVICE_LOG_STOP_WAIT_TIMEOUT_SECONDS
        last_payload = None

        while time.time() < deadline:
            QApplication.processEvents()
            if signal_path.exists():
                try:
                    payload = json.loads(signal_path.read_text(encoding="utf-8"))
                    last_payload = payload
                    if payload.get("event") == "device_log_stopped":
                        return payload
                except Exception:
                    log_exception(f"read device log state failed: signal_path={signal_path}")
                    return last_payload
            time.sleep(DEVICE_LOG_SETTLE_INTERVAL_SECONDS)

        return last_payload

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

    def _build_stream_disconnect_notice(self, device_log_path: Optional[Path], archived_log_name: Optional[str]) -> str:
        uses_sp_recording = self._current_plan_uses_sp_recording()
        if self.current_run_stream_disconnect_startup:
            phase_text = "启动阶段(SP未开始)" if uses_sp_recording else "启动阶段(首帧未到达)"
        else:
            phase_text = "SP记录后" if uses_sp_recording else "功能测试首帧后"
        lines = [
            "gRPC 流断连提醒",
            f"发生阶段: {phase_text}",
            f"断流信息: {self.current_run_stream_disconnect_message}",
            f"SP记录启用: {uses_sp_recording}",
            f"SP是否已开始: {self.current_run_sp_started}",
            f"设备日志源文件: {device_log_path if device_log_path else '未定位'}",
            f"归档日志文件: logs/{archived_log_name}" if archived_log_name else "归档日志文件: 未找到设备日志源文件",
            "",
            "说明: 本文件由 launcher 在归档时生成，用于快速定位断流对应的 testcases 设备日志。",
        ]
        return "\n".join(lines) + "\n"

    def _resolve_current_run_archive_dir(self) -> Optional[Path]:
        if self.current_run_archive_dir is not None:
            self.current_run_archive_dir.mkdir(parents=True, exist_ok=True)
            return self.current_run_archive_dir

        if self.current_plan is None:
            return None

        archive_metadata = {}
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
            logs_dir = archive_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            run_output_text = self._all_output_text()[self.current_run_output_start:]
            (logs_dir / "launcher_output_partial.txt").write_text(
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
        LOGGER.info("sp recording started detected: source=%s state=%s", source, state)

    def _refresh_current_run_sp_state(self):
        archive_dir = self.current_run_archive_dir
        if archive_dir is None:
            return

        state_path = archive_dir / "sp_recording_state.json"
        if not state_path.exists():
            return

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            log_exception(f"read sp recording state failed: state_path={state_path}")
            return

        self.current_run_sp_state = state
        if (
            state.get("sp_started_ever")
            or state.get("sp_recording")
            or state.get("sp_saved")
        ):
            self._mark_current_run_sp_started("state_file", state)

    def _current_plan_uses_sp_recording(self) -> bool:
        if self.current_plan is None:
            return True
        return should_use_sp_recording_for_profile(self.current_plan.get("test_profile"))

    def _current_plan_uses_hdc_capture(self) -> bool:
        if self.current_plan is None:
            return False
        screen_mode = str(self.current_plan.get("screen_mode") or "").strip()
        return screen_mode == "1"

    def _stream_disconnect_recovery_enabled(self) -> bool:
        return not self._current_plan_uses_hdc_capture()

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

    def _save_sp_on_stream_disconnect(self) -> bool:
        try:
            resolution = get_resolution()
        except Exception:
            resolution = None

        if resolution:
            screen_w, screen_h = int(resolution[0]), int(resolution[1])
        else:
            screen_w, screen_h = 2832, 1316

        x = int(round(screen_w * STREAM_DISCONNECT_SP_NORM_POS[0]))
        y = int(round(screen_h * STREAM_DISCONNECT_SP_NORM_POS[1]))
        command = f"uinput -T -d {x} {y} -i {STREAM_DISCONNECT_SP_LONG_PRESS_MS} -u {x} {y}"

        self._log_message(
            f"[Launcher] 断流保全：尝试长按 SP 保存，pos=({x},{y}), duration={STREAM_DISCONNECT_SP_LONG_PRESS_MS}ms。\n"
        )
        result = run_hdc_shell(command)
        ok = result is not None
        if not ok:
            self._log_message("[Launcher] 断流保全：SP 保存指令执行失败，请检查 hdc/uinput 状态。\n", level=logging.WARNING)
        return ok

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
        }

        if archive_dir is not None:
            screenshot_path = self._capture_stream_disconnect_screenshot(archive_dir)
            preserve_result["screenshot_path"] = str(screenshot_path) if screenshot_path else None

        if not self.current_run_stream_disconnect_startup and self._current_plan_uses_sp_recording():
            preserve_result["sp_save_attempted"] = True
            preserve_result["sp_save_ok"] = self._save_sp_on_stream_disconnect()

        if archive_dir is not None:
            try:
                (archive_dir / "stream_disconnect_preserve.json").write_text(
                    json.dumps(preserve_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                log_exception(f"write stream disconnect preserve result failed: archive_dir={archive_dir}")

    def _mark_stream_disconnected(self, message: str, source: str):
        if self.current_run_stream_disconnected:
            return

        self._refresh_current_run_sp_state()
        self.current_run_stream_disconnected = True
        uses_sp_recording = self._current_plan_uses_sp_recording()
        if uses_sp_recording:
            self.current_run_stream_disconnect_startup = not self.current_run_sp_started
        else:
            self.current_run_stream_disconnect_startup = not self.current_run_stream_started
        self.current_run_stream_disconnect_message = str(message or source or "stream disconnected")
        if self.current_run_stream_disconnect_startup:
            phase_text = "启动阶段" if uses_sp_recording else "启动阶段(首帧未到达)"
        else:
            phase_text = "SP记录后" if uses_sp_recording else "功能测试首帧后"
        self._log_message(
            f"\n[Launcher] 检测到 gRPC 流断连({phase_text}, source={source})："
            f"{self.current_run_stream_disconnect_message}\n"
        )
        if self.current_run_stream_disconnect_startup:
            if uses_sp_recording:
                self._log_message(
                    "[Launcher] SP 记录尚未开始，本次按启动阶段断流处理："
                    "不截图、不保存 SP、不归档本轮日志，直接停止当前子进程并准备重启。\n"
                )
            else:
                self._log_message(
                    "[Launcher] 功能测试首帧尚未到达，本次按启动阶段断流处理："
                    "不截图、不归档本轮日志，直接停止当前子进程并准备重启。\n"
                )
            self._set_status("启动阶段 gRPC 流断连，正在停止当前子进程并准备重启手机。")
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

    def _archive_run_outputs(self, run_no: int, exit_code: int):
        if self.current_plan is None:
            return

        run_output_text = self._all_output_text()[self.current_run_output_start:]
        extra_text_files = {"launcher_output.txt": run_output_text}
        extra_log_files = {}
        stream_device_log_path = None
        stream_archived_log_name = None
        stream_notice_written = False
        device_log_stop_state = None

        if self.current_run_stream_disconnected:
            stream_device_log_path = self._resolve_current_device_log_path()
            if stream_device_log_path is not None:
                if not self.current_run_stream_disconnect_startup:
                    device_log_stop_state = self._wait_for_device_log_stopped_signal()
                    if device_log_stop_state and device_log_stop_state.get("event") == "device_log_stopped":
                        self._log_message(
                            "[Launcher] 已确认 testcases 执行 stop_device_log()，"
                            f"log_path={device_log_stop_state.get('log_path')}。\n"
                        )
                    else:
                        self._log_message(
                            "[Launcher] 未等到 testcases 的 device_log_stopped 标记，"
                            "将继续尝试复制当前日志文件；请检查子进程是否被提前终止或 stop_log 是否失败。\n",
                            level=logging.WARNING,
                        )
                self._wait_for_device_log_stable(stream_device_log_path)
                stream_notice_written = self._append_stream_disconnect_notice_to_device_log(
                    stream_device_log_path,
                    exit_code,
                )
                if stream_device_log_path.exists() and stream_device_log_path.is_file():
                    stream_archived_log_name = f"stream_disconnect_device_log_{stream_device_log_path.name}"
                    extra_log_files[stream_archived_log_name] = str(stream_device_log_path)

            extra_text_files["stream_disconnect_notice.txt"] = self._build_stream_disconnect_notice(
                stream_device_log_path,
                stream_archived_log_name,
            )

        try:
            archive_dir = archive_run_artifacts(
                run_index=run_no,
                source="launcher",
                extra_text_files=extra_text_files,
                extra_log_files=extra_log_files or None,
                generate_preview_video=bool(self.current_plan.get("generate_preview_video")),
                extra_metadata={
                    "mode": self.current_plan["mode"],
                    "project_case": self.current_plan["project_case"],
                    "target_case": self.current_plan["target_case"],
                    "testcase_label": self.current_plan["testcase_label"],
                    "exit_code": exit_code,
                    "timed_out": self.current_run_timed_out,
                    "stream_disconnected": self.current_run_stream_disconnected,
                    "stream_disconnect_startup": self.current_run_stream_disconnect_startup,
                    "stream_disconnect_message": self.current_run_stream_disconnect_message,
                    "stream_disconnect_device_log_source": str(stream_device_log_path) if stream_device_log_path else None,
                    "stream_disconnect_device_log_archived": stream_archived_log_name,
                    "stream_disconnect_notice_written": stream_notice_written,
                    "device_log_stop_state": device_log_stop_state,
                    "stream_started": self.current_run_stream_started,
                    "sp_started": self.current_run_sp_started,
                    "sp_state": self.current_run_sp_state,
                    "batch_start_timestamp": self.current_batch_start_timestamp,
                    "run_start_timestamp": self.current_run_start_timestamp,
                    "inactivity_timeout_minutes": self.current_plan["inactivity_timeout_minutes"],
                    "generate_preview_video": bool(self.current_plan.get("generate_preview_video")),
                },
                reuse_existing=True,
            )
            self._log_message(f"[Launcher] 本次运行产物已归档到：{archive_dir}\n")
            if self.current_run_stream_disconnected:
                if stream_archived_log_name:
                    self._log_message(
                        f"[Launcher] 断流设备日志已额外归档：{archive_dir / 'logs' / stream_archived_log_name}\n"
                    )
                else:
                    self._log_message(
                        "[Launcher] 未找到本次断流对应的设备日志文件，已写入 stream_disconnect_notice.txt。\n",
                        level=logging.WARNING,
                    )
        except Exception:
            log_exception(f"archive_run_outputs failed: run_no={run_no}")
            self._log_message("[Launcher] 运行产物归档失败，请查看 launcher_debug.log。\n", level=logging.ERROR)

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
        self.process.kill()

    def _prepare_capture_mode_for_plan(self, plan: dict) -> bool:
        screen_mode = str(plan.get("screen_mode") or "0")
        test_profile = str(plan.get("test_profile") or "power")
        try:
            write_screen_mode_config(screen_mode)
        except Exception as exc:
            log_exception("write screen_mode config failed")
            QMessageBox.warning(self, "截图模式配置失败", f"写入 screen_mode={screen_mode} 失败：{exc}")
            return False

        profile_text = "功耗测试" if test_profile == "power" else "功能测试"
        self._log_message(
            f"[Launcher] 已切换为{profile_text}，screen_mode={screen_mode}。\n"
        )
        check_result = check_capture_stream_for_screen_mode(screen_mode)
        plan["capture_preflight_message"] = check_result.message
        self._log_message(f"[Launcher] 截图流预检：{check_result.message}\n")
        if not check_result.ok:
            QMessageBox.warning(self, "截图流预检失败", check_result.message)
            return False
        return True

    def _restart_phone_from_button(self):
        if self.restart_phone_script_running:
            return
        if self.batch_active or self.process is not None:
            QMessageBox.information(self, "运行中", "当前已有任务在运行，请先停止。")
            return

        script_path = APP_DIR / "restart.bat"
        if not script_path.is_file():
            message = f"未找到重启脚本：{script_path}"
            LOGGER.warning("restart phone script missing: %s", script_path)
            self._log_message(f"[Launcher] {message}\n", level=logging.WARNING)
            QMessageBox.warning(self, "重启手机", message)
            return

        self.restart_phone_script_running = True
        self.start_button.setEnabled(False)
        self.restart_phone_button.setEnabled(False)
        self._log_message(f"[Launcher] 正在执行重启手机脚本：{script_path}\n")
        thread = threading.Thread(
            target=self._run_restart_phone_script,
            args=(script_path,),
            daemon=True,
            name="launcher-restart-phone",
        )
        thread.start()

    def _run_restart_phone_script(self, script_path: Path):
        try:
            launch_restart_bat_with_system_shell(script_path)
            self.restart_phone_script_finished.emit(True, "已通过 cmd 打开 restart.bat。")
        except Exception as exc:
            log_exception(f"restart phone script failed: script_path={script_path}")
            self.restart_phone_script_finished.emit(False, f"执行 restart.bat 失败：{exc}")

    def _finish_restart_phone_script(self, success: bool, message: str):
        self.restart_phone_script_running = False
        if not self.batch_active and self.process is None:
            self.start_button.setEnabled(True)
            self.restart_phone_button.setEnabled(True)
        level = logging.INFO if success else logging.ERROR
        self._log_message(f"[Launcher] {message}\n", level=level)
        if not success:
            QMessageBox.warning(self, "重启手机", message)

    def _start_run(self):
        LOGGER.info(
            "start_button clicked: batch_active=%s process_exists=%s",
            self.batch_active,
            self.process is not None,
        )
        if self.batch_active or self.process is not None:
            QMessageBox.information(self, "运行中", "当前已有任务在运行，请先停止。")
            return

        plan = self._collect_plan()
        if plan is None:
            LOGGER.info("start_run aborted because plan is None")
            return

        if not self._prepare_capture_mode_for_plan(plan):
            LOGGER.info("start_run aborted because capture mode preflight failed")
            return

        self._begin_batch(plan)

    def _read_process_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_output(text)
        self._handle_stream_output(text)
        stripped = text.strip()
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
        if any(marker in text for marker in SP_RECORD_EVER_STARTED_MARKERS):
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

                result = subprocess.run(
                    command,
                    cwd=str(APP_DIR),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    timeout=30,
                    **hidden_subprocess_kwargs(),
                )

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

    def _restart_device_for_stream_disconnect(self) -> bool:
        self._log_message("[Launcher] 开始执行断流恢复命令。\n")
        self._set_runtime("运行信息：检测到断流，正在重启手机并等待开机。")
        QApplication.processEvents()

        script_path = APP_DIR / "restart.bat"
        if not script_path.is_file():
            self._log_message(f"[Launcher] 未找到重启脚本：{script_path}\n", level=logging.ERROR)
            return False

        self._log_message(f"[Launcher] 弹出 cmd 窗口执行断流恢复脚本：{script_path}\n")
        try:
            launch_restart_bat_with_system_shell(script_path)
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
            time.sleep(1)

        if not self._reinitialize_stream_service():
            return False

        self._log_message("[Launcher] 手机重启与端口恢复完成。\n")
        self.dismiss_reboot_prompt_on_next_case_start = True
        return True

    def _on_process_finished(self, exit_code: int, _exit_status):
        LOGGER.info(
            "process finished: exit_code=%s exit_status=%s current_run_index=%s timed_out=%s",
            exit_code,
            _exit_status,
            self.current_run_index,
            self.current_run_timed_out,
        )
        self.run_timeout_timer.stop()
        self.stream_disconnect_signal_timer.stop()
        if not self.current_run_stream_disconnected:
            self._poll_stream_disconnect_signal()
        finish_prefix = "进程结束"
        if self.stop_requested:
            finish_prefix = "进程已手动停止"
        self._log_message(f"\n[Launcher] {finish_prefix}，exit_code={exit_code}\n")
        self._poll_preview_frame()
        run_no = self.current_run_index + 1
        startup_stream_disconnect = (
            self.current_run_stream_disconnected
            and self.current_run_stream_disconnect_startup
        )
        if startup_stream_disconnect:
            self._discard_startup_disconnect_archive_dir()
        else:
            self._archive_run_outputs(run_no, exit_code)
        self.preview_timer.stop()
        if self.process is not None:
            self.process.deleteLater()
            self.process = None

        if not self.batch_active or self.current_plan is None:
            self._finish_batch(f"任务已结束，退出码：{exit_code}")
            return

        if self.stop_requested:
            if self.preserve_device_apps_on_manual_stop:
                self._log_message("[Launcher] 手动停止后保留设备现场，跳过应用清理。\n")
            else:
                self._cleanup_apps_between_runs("停止后清理")
            self._finish_batch("任务已停止。")
            return

        if self.current_run_stream_disconnected:
            if self.current_run_stream_disconnect_startup:
                self._log_message(
                    "[Launcher] 本次断流发生在 SP 记录开始前，不计入已执行次数，"
                    "本轮不保存产物；重启手机后将重新运行当前用例。\n"
                )
                if not self._restart_device_for_stream_disconnect():
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

            self._log_message(
                "[Launcher] 本次断流发生在 SP 记录开始后，结果已归档并写入 stream_disconnected 标志；"
                "当前用例计为完成。\n"
            )

            self.current_run_index += 1
            if self.current_run_index >= self.current_plan["run_count"]:
                self._log_message(
                    "[Launcher] 本次中途断流发生在最后一轮，已保存 SP 并归档本轮状态，跳过手机重启。\n"
                )
                self._cleanup_apps_between_runs("最后一轮断流后清理")
                self._finish_batch("所有运行次数已完成。最后一轮断流已归档，未执行重启。")
                return

            if not self._restart_device_for_stream_disconnect():
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

        if self.current_run_index >= self.current_plan["run_count"]:
            self._finish_batch("所有运行次数已完成。")
            return

        next_run = self.current_run_index + 1
        self._set_status(f"第 {self.current_run_index}/{self.current_plan['run_count']} 次已结束，检查第 {next_run} 次启动条件。")
        self._set_runtime(f"运行信息：已完成 {self.current_run_index}/{self.current_plan['run_count']} 次，准备下一次安全检查。")
        self._check_and_start_if_safe()
        if self.batch_active and self.process is None and not self.safety_timer.isActive():
            self.safety_timer.start()

    def _stop_run(self):
        LOGGER.info(
            "stop_button clicked: batch_active=%s process_exists=%s keep_process=%s",
            self.batch_active,
            self.process is not None,
            self.keep_process_on_manual_stop_button.isChecked(),
        )
        if not self.batch_active and self.process is None:
            return

        self.stop_requested = True
        self.preserve_device_apps_on_manual_stop = True
        self.safety_timer.stop()
        self.run_timeout_timer.stop()

        if self.process is None:
            self._log_message("\n[Launcher] 已取消后续运行。\n")
            self._log_message("[Launcher] 手动停止后保留设备现场，跳过应用清理。\n")
            self._finish_batch("任务已停止。")
            return

        if self.keep_process_on_manual_stop_button.isChecked():
            self._log_message(
                "\n[Launcher] 已取消后续运行，当前子进程将继续运行，直至自行结束。\n"
            )
            self._set_status("已手动停止后续运行，当前子进程继续运行中。")
            self._set_runtime("运行信息：后续轮次已取消，等待当前子进程自然结束。")
            return

        self._log_message("\n[Launcher] 正在停止当前子进程，并取消后续运行...\n")
        self.preview_timer.stop()
        self.process.kill()


def _run_helper_command(args: argparse.Namespace) -> int:
    LOGGER.info("run_helper_command: args=%s", args)
    exit_code = 0
    try:
        if args.run_testcase:
            run_testcase_entry(args.run_testcase)
            return exit_code

        if args.run_direct:
            project_case, target_case = args.run_direct
            run_direct_entry(project_case, target_case)
            return exit_code

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
        "path context: frozen=%s sys_executable=%s __file__=%s APP_DIR=%s INTERNAL_DIR=%s ROOT_DIR=%s TEMP_DIR=%s LOG_DIR=%s PREVIEW_DIR=%s old_cwd=%s new_cwd=%s chdir_error=%s hidden_subprocess_patch=%s hidden_window_suppressor=%s",
        bool(getattr(sys, "frozen", False)),
        sys.executable,
        __file__,
        APP_DIR,
        INTERNAL_DIR,
        ROOT_DIR,
        TEMP_DIR,
        LOG_DIR,
        PREVIEW_DIR,
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
    args, unknown_args = parser.parse_known_args()
    is_helper = bool(args.run_testcase or args.run_direct)
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
