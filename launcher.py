import argparse
import ast
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QRadioButton,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from aw.autogame.tools.Utils import archive_run_artifacts, get_resolution, select_scene_resolution
from aw.autogame.tools.AreaResolver import resolve_area_rect_for_frame

ROOT_DIR = Path(__file__).resolve().parent
TESTCASES_DIR = ROOT_DIR / "testcases"
CUSTOMS_EXAMPLES_DIR = ROOT_DIR / "aw" / "autogame" / "customs_examples"
CUSTOMS_GAME_EXAMPLES_DIR = ROOT_DIR / "aw" / "autogame" / "customs_game_examples"
PREVIEW_DIR = ROOT_DIR / "aw" / "autogame" / "temp" / "logs" / "process_temp_logs"
LOG_DIR = ROOT_DIR / "aw" / "autogame" / "temp" / "logs"
LAUNCHER_LOG_FILE = LOG_DIR / "launcher_debug.log"
PACKAGE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+){2,}")
LOGGER = logging.getLogger("launcher")


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
    LOGGER.info("run_testcase_entry: testcase_label=%s", testcase_label)
    from xdevice.__main__ import main_process

    main_process(f"run -l {testcase_label}")


def run_direct_entry(project_case: str, target_case: str):
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
    hdc_executable = shutil.which("hdc") or shutil.which("hdc.exe") or "hdc"
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
            timeout=20,
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


def get_battery_temperature_c() -> Optional[float]:
    raw = run_hdc_shell("cat /sys/class/power_supply/Battery/temp")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value / 10.0 if value > 100 else value


def get_battery_capacity() -> Optional[int]:
    raw = run_hdc_shell("cat /sys/class/power_supply/Battery/capacity")
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


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
    def __init__(self):
        super().__init__()
        LOGGER.info("LauncherWindow init start")
        self.process: Optional[QProcess] = None
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

        self.batch_active = False
        self.stop_requested = False
        self.current_run_index = 0
        self.total_runs = 1
        self.current_plan: Optional[dict] = None
        self.current_run_timed_out = False
        self.current_run_output_start = 0
        self.preserve_device_apps_on_manual_stop = True
        self.current_batch_start_timestamp: Optional[str] = None
        self.current_run_start_timestamp: Optional[str] = None
        self.preview_target_info_height = 220
        self.preview_target_info_width = 360
        self._adjusting_preview_splitter = False

        self.setWindowTitle("Auto Game 启动器")
        self.resize(1260, 860)

        self.mode_testcase = QRadioButton("通过 testcases 用例启动")
        self.mode_direct = QRadioButton("直接指定 project_case / target_case")
        self.mode_testcase.setChecked(True)

        self.testcase_path_edit = QLineEdit()
        self.testcase_path_edit.setReadOnly(True)
        self.testcase_path_edit.setPlaceholderText("未选择 testcases 用例文件")

        self.browse_button = QPushButton("选择用例")
        self.clear_button = QPushButton("清空")
        self.refresh_button = QPushButton("刷新配置")

        self.project_combo = QComboBox()
        self.target_combo = QComboBox()

        self.status_label = QLabel("请选择启动方式，并选择 testcases 用例或直接指定配置。")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.runtime_label = QLabel("运行信息：未开始")
        self.runtime_label.setWordWrap(True)
        self.runtime_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.run_count_spin = QSpinBox()
        self.run_count_spin.setRange(1, 9999)
        self.run_count_spin.setValue(1)

        self.safe_temp_spin = QDoubleSpinBox()
        self.safe_temp_spin.setRange(0.0, 100.0)
        self.safe_temp_spin.setDecimals(1)
        self.safe_temp_spin.setSingleStep(0.5)
        self.safe_temp_spin.setValue(40.0)
        self.safe_temp_spin.setSuffix(" °C")

        self.safe_battery_spin = QSpinBox()
        self.safe_battery_spin.setRange(0, 100)
        self.safe_battery_spin.setValue(25)
        self.safe_battery_spin.setSuffix(" %")

        self.safe_time_spin = QDoubleSpinBox()
        self.safe_time_spin.setRange(0.0, 10000.0)
        self.safe_time_spin.setDecimals(1)
        self.safe_time_spin.setSingleStep(1.0)
        self.safe_time_spin.setValue(0.0)
        self.safe_time_spin.setSuffix(" 分钟")

        self.inactivity_timeout_spin = QDoubleSpinBox()
        self.inactivity_timeout_spin.setRange(0.0, 10000.0)
        self.inactivity_timeout_spin.setDecimals(1)
        self.inactivity_timeout_spin.setSingleStep(1.0)
        self.inactivity_timeout_spin.setValue(5.0)
        self.inactivity_timeout_spin.setSuffix(" 分钟")

        self.start_button = QPushButton("启动")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.keep_process_on_manual_stop_button = QPushButton("停止保活")
        self.keep_process_on_manual_stop_button.setCheckable(True)
        self.keep_process_on_manual_stop_button.setChecked(False)
        self.preview_overlay_button = QPushButton("显示标注")
        self.preview_overlay_button.setCheckable(True)
        self.preview_overlay_button.setChecked(False)
        self.preview_points_button = QPushButton("显示控点")
        self.preview_points_button.setCheckable(True)
        self.preview_points_button.setChecked(False)

        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("运行输出会显示在这里...")

        self.preview_image_label = QLabel("启动后将在这里实时显示可视化帧")
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_label.setMinimumWidth(640)
        self.preview_image_label.setMinimumHeight(320)
        self.preview_image_label.setStyleSheet("border: 1px solid #666; background: #111; color: #ddd;")

        self.preview_info_edit = QPlainTextEdit()
        self.preview_info_edit.setReadOnly(True)
        self.preview_info_edit.setPlaceholderText("当前帧识别信息会显示在这里...")
        self.preview_info_edit.setMinimumHeight(150)
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

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        mode_group = QGroupBox("启动方式")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.addWidget(self.mode_testcase)
        mode_layout.addWidget(self.mode_direct)
        main_layout.addWidget(mode_group)

        testcase_group = QGroupBox("testcases 用例")
        testcase_layout = QHBoxLayout(testcase_group)
        testcase_layout.addWidget(self.testcase_path_edit, 1)
        testcase_layout.addWidget(self.browse_button)
        testcase_layout.addWidget(self.clear_button)
        main_layout.addWidget(testcase_group)

        config_group = QGroupBox("配置")
        config_layout = QFormLayout(config_group)
        config_layout.addRow("project_case", self.project_combo)
        config_layout.addRow("target_case", self.target_combo)
        config_layout.addRow("运行次数", self.run_count_spin)
        config_layout.addRow("安全温度", self.safe_temp_spin)
        config_layout.addRow("安全电量", self.safe_battery_spin)
        config_layout.addRow("安全时间", self.safe_time_spin)
        config_layout.addRow("无操控超时", self.inactivity_timeout_spin)
        config_layout.addRow("解析结果", self.status_label)
        config_layout.addRow("运行信息", self.runtime_label)
        config_layout.addRow("", self.refresh_button)
        main_layout.addWidget(config_group)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.stop_button)
        action_layout.addWidget(self.keep_process_on_manual_stop_button)
        action_layout.addWidget(self.preview_overlay_button)
        action_layout.addWidget(self.preview_points_button)
        action_layout.addStretch(1)
        main_layout.addLayout(action_layout)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setHandleWidth(8)

        preview_group = QGroupBox("实时可视化")
        preview_layout = QVBoxLayout(preview_group)
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
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.output_edit)
        content_splitter.addWidget(preview_group)
        content_splitter.addWidget(log_group)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)

        main_layout.addWidget(content_splitter, 1)

    def _bind_signals(self):
        self.mode_testcase.toggled.connect(self._sync_mode_ui)
        self.browse_button.clicked.connect(self._choose_testcase_file)
        self.clear_button.clicked.connect(self._clear_testcase_file)
        self.refresh_button.clicked.connect(self._refresh_config_choices)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        self.start_button.clicked.connect(self._start_run)
        self.stop_button.clicked.connect(self._stop_run)
        self.keep_process_on_manual_stop_button.toggled.connect(self._toggle_keep_process_on_manual_stop)
        self.preview_overlay_button.toggled.connect(self._toggle_preview_overlay)
        self.preview_points_button.toggled.connect(self._toggle_preview_points)
        self.preview_timer.timeout.connect(self._poll_preview_frame)
        self.safety_timer.timeout.connect(self._check_and_start_if_safe)
        self.run_timeout_timer.timeout.connect(self._handle_run_timeout)
        LOGGER.debug("signals bound")

    def _append_output(self, text: str):
        if not text:
            return

        scrollbar = self.output_edit.verticalScrollBar()
        should_follow = scrollbar.value() >= max(0, scrollbar.maximum() - 4)

        self.output_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.output_edit.insertPlainText(text)
        if should_follow:
            self.output_edit.moveCursor(QTextCursor.MoveOperation.End)
            scrollbar.setValue(scrollbar.maximum())
        QApplication.processEvents()

    def _log_message(self, text: str, level: int = logging.INFO):
        self._append_output(text)
        message = text.rstrip()
        if message:
            LOGGER.log(level, message)

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _set_runtime(self, text: str):
        self.runtime_label.setText(text)

    def _toggle_preview_overlay(self, checked: bool):
        self.preview_overlay_button.setText("隐藏标注" if checked else "显示标注")
        LOGGER.info("preview overlay toggled: %s", checked)
        self._refresh_preview_pixmap()

    def _toggle_preview_points(self, checked: bool):
        self.preview_points_button.setText("隐藏控点" if checked else "显示控点")
        LOGGER.info("preview points toggled: %s", checked)
        self._refresh_preview_pixmap()

    def _toggle_keep_process_on_manual_stop(self, checked: bool):
        self.keep_process_on_manual_stop_button.setText(
            "停止保活: 开" if checked else "停止保活"
        )
        LOGGER.info("keep process on manual stop toggled: %s", checked)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adjust_preview_splitter_sizes()
        self._refresh_preview_pixmap()

    def _sync_mode_ui(self):
        testcase_mode = self.mode_testcase.isChecked()
        LOGGER.debug("sync_mode_ui: testcase_mode=%s", testcase_mode)
        self.testcase_path_edit.setEnabled(testcase_mode)
        self.browse_button.setEnabled(testcase_mode)
        self.clear_button.setEnabled(testcase_mode)

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
        self._set_status("已刷新 project_case 和 target_case 列表。")

    def _on_project_changed(self, project_case: str):
        if self._updating_targets:
            return
        LOGGER.info("project changed: %s", project_case)
        self._load_target_cases(preferred=None)
        if project_case:
            self._set_status(f"已选择 project_case={project_case}，请确认 target_case。")
        self._refresh_preview_pixmap()

    def _choose_testcase_file(self):
        LOGGER.info("choose_testcase_file dialog open")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 testcases 用例",
            str(TESTCASES_DIR),
            "Python Files (*.py)",
        )
        if not file_path:
            LOGGER.info("choose_testcase_file canceled")
            return

        py_file = Path(file_path).resolve()
        try:
            py_file.relative_to(TESTCASES_DIR)
            rel_path = py_file.relative_to(ROOT_DIR)
        except ValueError:
            LOGGER.warning("choose_testcase_file invalid path: %s", py_file)
            QMessageBox.warning(self, "路径错误", "请选择当前项目 testcases 目录下的用例文件。")
            return

        self.selected_testcase_file = py_file
        LOGGER.info("choose_testcase_file selected: %s", py_file)
        self.testcase_path_edit.setText(rel_path.as_posix())
        self.mode_testcase.setChecked(True)
        self._apply_parsed_testcase(py_file)

    def _clear_testcase_file(self):
        LOGGER.info("clear_testcase_file")
        self.selected_testcase_file = None
        self.testcase_path_edit.clear()
        self._set_status("已清空 testcases 选择。可以直接指定 project_case / target_case 启动。")

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

        self._set_status("；".join(messages))

    def _build_process_environment(self, project_case: str, target_case: str, run_no: int) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        env.insert("TARGET_PROJECT_CASE", project_case)
        env.insert("TARGET_GAME_CASE", target_case)
        env.insert("AUTOGAME_VIS_MODE", "launcher")
        env.insert("AUTOGAME_RUN_SOURCE", "launcher")
        env.insert("AUTOGAME_RUN_INDEX", str(int(run_no)))
        if self.current_batch_start_timestamp:
            env.insert("AUTOGAME_BATCH_START_TIMESTAMP", self.current_batch_start_timestamp)
        if self.current_run_start_timestamp:
            env.insert("AUTOGAME_RUN_START_TIMESTAMP", self.current_run_start_timestamp)
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
        self.mode_testcase.setEnabled(enabled)
        self.mode_direct.setEnabled(enabled)
        self.testcase_path_edit.setEnabled(enabled and self.mode_testcase.isChecked())
        self.browse_button.setEnabled(enabled and self.mode_testcase.isChecked())
        self.clear_button.setEnabled(enabled and self.mode_testcase.isChecked())
        self.refresh_button.setEnabled(enabled)
        self.project_combo.setEnabled(enabled)
        self.target_combo.setEnabled(enabled)
        self.run_count_spin.setEnabled(enabled)
        self.safe_temp_spin.setEnabled(enabled)
        self.safe_battery_spin.setEnabled(enabled)
        self.safe_time_spin.setEnabled(enabled)
        self.inactivity_timeout_spin.setEnabled(enabled)

    def _clear_preview_files(self):
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

    def _refresh_preview_pixmap(self):
        if self.latest_preview_pixmap is None:
            return
        display_pixmap = self._build_preview_display_pixmap()
        self._adjust_preview_splitter_sizes()
        scaled = display_pixmap.scaled(
            self.preview_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_image_label.setPixmap(scaled)

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
        if not PREVIEW_DIR.exists():
            return

        latest_image = None
        latest_mtime = -1.0
        for path in PREVIEW_DIR.glob("frame_*.jpg"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_image = path

        if latest_image is None or latest_image == self.latest_preview_file:
            return

        json_path = latest_image.with_suffix(".json")
        if not json_path.exists():
            return

        pixmap = QPixmap(str(latest_image))
        if pixmap.isNull():
            LOGGER.warning("preview pixmap is null: %s", latest_image)
            return

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            log_exception(f"preview json load failed: {json_path}")
            payload = {"error": "json 读取失败"}

        self.latest_preview_file = latest_image
        self.latest_preview_pixmap = pixmap
        self.latest_preview_payload = payload if isinstance(payload, dict) else {"raw": payload}
        self.preview_image_label.setText("")
        self._adjust_preview_splitter_sizes()
        self._refresh_preview_pixmap()
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
            testcase_label = self.selected_testcase_file.relative_to(ROOT_DIR).with_suffix("").as_posix()
            mode = "testcase"

        cleanup_apps = set()
        if self.selected_testcase_file is not None:
            cleanup_apps.update(extract_package_names(self.selected_testcase_file))

        target_logic_file = (
            CUSTOMS_GAME_EXAMPLES_DIR / project_case / f"{target_case}.py"
        )
        cleanup_apps.update(extract_package_names(target_logic_file))

        plan = {
            "mode": mode,
            "project_case": project_case,
            "target_case": target_case,
            "testcase_label": testcase_label,
            "run_count": int(self.run_count_spin.value()),
            "safe_temp": float(self.safe_temp_spin.value()),
            "safe_battery": int(self.safe_battery_spin.value()),
            "safe_minutes": float(self.safe_time_spin.value()),
            "inactivity_timeout_minutes": float(self.inactivity_timeout_spin.value()),
            "cleanup_apps": sorted(cleanup_apps),
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
        self.output_edit.clear()
        self._clear_preview_files()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_inputs_enabled(False)
        self._set_status("已开始批量执行，准备进行安全检查。")
        self._set_runtime(f"运行信息：共 {plan['run_count']} 次，等待第 1 次启动。")
        self._log_message(
            f"[Launcher] 批量运行开始，mode={plan['mode']}, runs={plan['run_count']}, "
            f"safe_temp={plan['safe_temp']}°C, safe_battery={plan['safe_battery']}%, "
            f"safe_time={plan['safe_minutes']}分钟, inactivity_timeout={plan['inactivity_timeout_minutes']}分钟, "
            f"cleanup_apps={plan['cleanup_apps']}\n"
        )
        self._cleanup_apps_between_runs("批次启动前预清理")
        self._check_and_start_if_safe()

    def _finish_batch(self, message: str):
        LOGGER.info("finish_batch: %s", message)
        self.batch_active = False
        self.stop_requested = False
        self.current_plan = None
        self.current_run_timed_out = False
        self.current_run_output_start = 0
        self.preserve_device_apps_on_manual_stop = True
        self.current_batch_start_timestamp = None
        self.current_run_start_timestamp = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_inputs_enabled(True)
        self.preview_timer.stop()
        self.safety_timer.stop()
        self.run_timeout_timer.stop()
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
            self._set_status("无法读取手机温度或电量，稍后重试。")
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
        self.current_run_start_timestamp = time.strftime("%Y%m%d%H%M%S")
        self._clear_preview_files()
        self.current_run_output_start = len(self.output_edit.toPlainText())

        project_case = self.current_plan["project_case"]
        target_case = self.current_plan["target_case"]

        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setWorkingDirectory(str(ROOT_DIR))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.setProcessEnvironment(self._build_process_environment(project_case, target_case, run_no))
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)

        if self.current_plan["mode"] == "testcase":
            testcase_label = self.current_plan["testcase_label"]
            args = [str(ROOT_DIR / "launcher.py"), "--run-testcase", testcase_label]
            self._set_status(
                f"第 {run_no}/{self.current_plan['run_count']} 次启动：{testcase_label}"
            )
            self._log_message(f"\n[Launcher] 第 {run_no}/{self.current_plan['run_count']} 次：通过 testcase 启动 {testcase_label}\n")
        else:
            args = [str(ROOT_DIR / "launcher.py"), "--run-direct", project_case, target_case]
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
            ROOT_DIR,
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

        self.preview_timer.start()
        safe_minutes = self.current_plan["safe_minutes"]
        if safe_minutes > 0:
            self.run_timeout_timer.start(int(safe_minutes * 60 * 1000))

    def _archive_run_outputs(self, run_no: int, exit_code: int):
        if self.current_plan is None:
            return

        run_output_text = self.output_edit.toPlainText()[self.current_run_output_start:]
        try:
            archive_dir = archive_run_artifacts(
                run_index=run_no,
                source="launcher",
                extra_text_files={"launcher_output.txt": run_output_text},
                extra_metadata={
                    "mode": self.current_plan["mode"],
                    "project_case": self.current_plan["project_case"],
                    "target_case": self.current_plan["target_case"],
                    "testcase_label": self.current_plan["testcase_label"],
                    "exit_code": exit_code,
                    "timed_out": self.current_run_timed_out,
                    "batch_start_timestamp": self.current_batch_start_timestamp,
                    "run_start_timestamp": self.current_run_start_timestamp,
                    "inactivity_timeout_minutes": self.current_plan["inactivity_timeout_minutes"],
                },
                reuse_existing=True,
            )
            self._log_message(f"[Launcher] 本次运行产物已归档到：{archive_dir}\n")
        except Exception:
            log_exception(f"archive_run_outputs failed: run_no={run_no}")
            self._log_message("[Launcher] 运行产物归档失败，请查看 launcher_debug.log。\n", level=logging.ERROR)

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

        self._begin_batch(plan)

    def _read_process_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_output(text)
        stripped = text.strip()
        if stripped:
            LOGGER.info("child_output: %s", stripped)

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

    def _on_process_finished(self, exit_code: int, _exit_status):
        LOGGER.info(
            "process finished: exit_code=%s exit_status=%s current_run_index=%s timed_out=%s",
            exit_code,
            _exit_status,
            self.current_run_index,
            self.current_run_timed_out,
        )
        self.run_timeout_timer.stop()
        finish_prefix = "进程结束"
        if self.stop_requested:
            finish_prefix = "进程已手动停止"
        self._log_message(f"\n[Launcher] {finish_prefix}，exit_code={exit_code}\n")
        self._poll_preview_frame()
        run_no = self.current_run_index + 1
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
    try:
        if args.run_testcase:
            run_testcase_entry(args.run_testcase)
            return 0

        if args.run_direct:
            project_case, target_case = args.run_direct
            run_direct_entry(project_case, target_case)
            return 0

        return 0
    except Exception:
        log_exception("helper command failed")
        traceback.print_exc()
        return 1


def main():
    setup_logging()
    install_global_exception_hooks()
    LOGGER.info("main start: argv=%s cwd=%s", sys.argv, os.getcwd())
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-testcase")
    parser.add_argument("--run-direct", nargs=2, metavar=("PROJECT_CASE", "TARGET_CASE"))
    args, _ = parser.parse_known_args()
    LOGGER.info("parsed args: %s", args)

    if args.run_testcase or args.run_direct:
        LOGGER.info("enter helper mode")
        raise SystemExit(_run_helper_command(args))

    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    LOGGER.info("launcher window shown")
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
