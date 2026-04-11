import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PyQt6.QtGui import QPixmap, QTextCursor
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


ROOT_DIR = Path(__file__).resolve().parent
TESTCASES_DIR = ROOT_DIR / "testcases"
CUSTOMS_EXAMPLES_DIR = ROOT_DIR / "aw" / "autogame" / "customs_examples"
CUSTOMS_GAME_EXAMPLES_DIR = ROOT_DIR / "aw" / "autogame" / "customs_game_examples"
PREVIEW_DIR = ROOT_DIR / "aw" / "autogame" / "temp" / "logs" / "process_temp_logs"


def parse_case_vars(py_file: Path) -> Dict[str, str]:
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


def discover_project_cases() -> list[str]:
    if not CUSTOMS_EXAMPLES_DIR.exists():
        return []

    cases = []
    for path in sorted(CUSTOMS_EXAMPLES_DIR.iterdir()):
        if path.is_dir() and (path / "info.py").exists():
            cases.append(path.name)
    return cases


def discover_target_cases(project_case: str) -> list[str]:
    project_dir = CUSTOMS_GAME_EXAMPLES_DIR / project_case
    if not project_dir.exists():
        return []

    cases = []
    for path in sorted(project_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        cases.append(path.stem)
    return cases


def run_testcase_entry(testcase_label: str):
    from xdevice.__main__ import main_process

    main_process(f"run -l {testcase_label}")


def run_direct_entry(project_case: str, target_case: str):
    os.environ["TARGET_PROJECT_CASE"] = project_case
    os.environ["TARGET_GAME_CASE"] = target_case

    from aw.autogame.tools.GameAutomator import GameAutomator

    automator = GameAutomator(driver=None, logger=None)
    automator.start()


def run_hdc_shell(command: str) -> Optional[str]:
    try:
        result = subprocess.run(
            f'hdc shell "{command}"',
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        print(f"[Launcher] hdc shell 执行失败: {command}\n{exc.stderr}")
        return None
    return result.stdout.strip()


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


class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.process: Optional[QProcess] = None
        self.selected_testcase_file: Optional[Path] = None
        self._updating_targets = False
        self.latest_preview_file: Optional[Path] = None
        self.latest_preview_pixmap: Optional[QPixmap] = None
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

        self.start_button = QPushButton("启动")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)

        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("运行输出会显示在这里...")

        self.preview_image_label = QLabel("启动后将在这里实时显示可视化帧")
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_label.setMinimumSize(640, 360)
        self.preview_image_label.setStyleSheet("border: 1px solid #666; background: #111; color: #ddd;")

        self.preview_info_edit = QPlainTextEdit()
        self.preview_info_edit.setReadOnly(True)
        self.preview_info_edit.setPlaceholderText("当前帧识别信息会显示在这里...")

        self._build_ui()
        self._bind_signals()
        self._load_project_cases()
        self._sync_mode_ui()

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
        config_layout.addRow("解析结果", self.status_label)
        config_layout.addRow("运行信息", self.runtime_label)
        config_layout.addRow("", self.refresh_button)
        main_layout.addWidget(config_group)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.stop_button)
        action_layout.addStretch(1)
        main_layout.addLayout(action_layout)

        content_layout = QHBoxLayout()

        preview_group = QGroupBox("实时可视化")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(self.preview_image_label, 3)
        preview_layout.addWidget(self.preview_info_edit, 2)
        content_layout.addWidget(preview_group, 3)

        log_group = QGroupBox("运行输出")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.output_edit)
        content_layout.addWidget(log_group, 2)

        main_layout.addLayout(content_layout, 1)

    def _bind_signals(self):
        self.mode_testcase.toggled.connect(self._sync_mode_ui)
        self.browse_button.clicked.connect(self._choose_testcase_file)
        self.clear_button.clicked.connect(self._clear_testcase_file)
        self.refresh_button.clicked.connect(self._refresh_config_choices)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        self.start_button.clicked.connect(self._start_run)
        self.stop_button.clicked.connect(self._stop_run)
        self.preview_timer.timeout.connect(self._poll_preview_frame)
        self.safety_timer.timeout.connect(self._check_and_start_if_safe)
        self.run_timeout_timer.timeout.connect(self._handle_run_timeout)

    def _append_output(self, text: str):
        if not text:
            return
        self.output_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.output_edit.insertPlainText(text)
        self.output_edit.moveCursor(QTextCursor.MoveOperation.End)

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _set_runtime(self, text: str):
        self.runtime_label.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    def _sync_mode_ui(self):
        testcase_mode = self.mode_testcase.isChecked()
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
        self._load_project_cases(preferred=project)
        self._load_target_cases(preferred=target)
        self._set_status("已刷新 project_case 和 target_case 列表。")

    def _on_project_changed(self, project_case: str):
        if self._updating_targets:
            return
        self._load_target_cases(preferred=None)
        if project_case:
            self._set_status(f"已选择 project_case={project_case}，请确认 target_case。")

    def _choose_testcase_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 testcases 用例",
            str(TESTCASES_DIR),
            "Python Files (*.py)",
        )
        if not file_path:
            return

        py_file = Path(file_path).resolve()
        try:
            py_file.relative_to(TESTCASES_DIR)
            rel_path = py_file.relative_to(ROOT_DIR)
        except ValueError:
            QMessageBox.warning(self, "路径错误", "请选择当前项目 testcases 目录下的用例文件。")
            return

        self.selected_testcase_file = py_file
        self.testcase_path_edit.setText(rel_path.as_posix())
        self.mode_testcase.setChecked(True)
        self._apply_parsed_testcase(py_file)

    def _clear_testcase_file(self):
        self.selected_testcase_file = None
        self.testcase_path_edit.clear()
        self._set_status("已清空 testcases 选择。可以直接指定 project_case / target_case 启动。")

    def _apply_parsed_testcase(self, py_file: Path):
        try:
            parsed = parse_case_vars(py_file)
        except Exception as exc:
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

    def _build_process_environment(self, project_case: str, target_case: str) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        env.insert("TARGET_PROJECT_CASE", project_case)
        env.insert("TARGET_GAME_CASE", target_case)
        env.insert("AUTOGAME_VIS_MODE", "launcher")
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

    def _clear_preview_files(self):
        self.latest_preview_file = None
        self.latest_preview_pixmap = None
        self.preview_image_label.setText("启动后将在这里实时显示可视化帧")
        self.preview_image_label.setPixmap(QPixmap())
        self.preview_info_edit.clear()

        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        for path in PREVIEW_DIR.iterdir():
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    def _refresh_preview_pixmap(self):
        if self.latest_preview_pixmap is None:
            return
        scaled = self.latest_preview_pixmap.scaled(
            self.preview_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_image_label.setPixmap(scaled)

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
            return

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {"error": "json 读取失败"}

        self.latest_preview_file = latest_image
        self.latest_preview_pixmap = pixmap
        self.preview_image_label.setText("")
        self._refresh_preview_pixmap()
        self.preview_info_edit.setPlainText(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _validate_selection(self) -> Optional[tuple[str, str]]:
        project_case = self.project_combo.currentText().strip()
        target_case = self.target_combo.currentText().strip()

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

        return {
            "mode": mode,
            "project_case": project_case,
            "target_case": target_case,
            "testcase_label": testcase_label,
            "run_count": int(self.run_count_spin.value()),
            "safe_temp": float(self.safe_temp_spin.value()),
            "safe_battery": int(self.safe_battery_spin.value()),
            "safe_minutes": float(self.safe_time_spin.value()),
        }

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
        self.current_plan = plan
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
        self._append_output(
            f"[Launcher] 批量运行开始，mode={plan['mode']}, runs={plan['run_count']}, "
            f"safe_temp={plan['safe_temp']}°C, safe_battery={plan['safe_battery']}%, "
            f"safe_time={plan['safe_minutes']}分钟\n"
        )
        self._check_and_start_if_safe()

    def _finish_batch(self, message: str):
        self.batch_active = False
        self.stop_requested = False
        self.current_plan = None
        self.current_run_timed_out = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_inputs_enabled(True)
        self.preview_timer.stop()
        self.safety_timer.stop()
        self.run_timeout_timer.stop()
        self._set_status(message)
        self._set_runtime(message)

    def _check_and_start_if_safe(self):
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
        self._launch_iteration(run_no, temperature, battery)

    def _launch_iteration(self, run_no: int, temperature: float, battery: int):
        if self.current_plan is None:
            return

        self.current_run_timed_out = False
        self._clear_preview_files()

        project_case = self.current_plan["project_case"]
        target_case = self.current_plan["target_case"]

        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setWorkingDirectory(str(ROOT_DIR))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.setProcessEnvironment(self._build_process_environment(project_case, target_case))
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._on_process_finished)

        if self.current_plan["mode"] == "testcase":
            testcase_label = self.current_plan["testcase_label"]
            args = [str(ROOT_DIR / "launcher.py"), "--run-testcase", testcase_label]
            self._set_status(
                f"第 {run_no}/{self.current_plan['run_count']} 次启动：{testcase_label}"
            )
            self._append_output(f"\n[Launcher] 第 {run_no}/{self.current_plan['run_count']} 次：通过 testcase 启动 {testcase_label}\n")
        else:
            args = [str(ROOT_DIR / "launcher.py"), "--run-direct", project_case, target_case]
            self._set_status(
                f"第 {run_no}/{self.current_plan['run_count']} 次启动：project_case={project_case}, target_case={target_case}"
            )
            self._append_output(
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
        self.process.start()
        started = self.process.waitForStarted(3000)
        if not started:
            QMessageBox.critical(self, "启动失败", "子进程启动失败，请检查 Python 环境。")
            self.process.deleteLater()
            self.process = None
            self._finish_batch("启动失败，批量任务已终止。")
            return

        self.preview_timer.start()
        safe_minutes = self.current_plan["safe_minutes"]
        if safe_minutes > 0:
            self.run_timeout_timer.start(int(safe_minutes * 60 * 1000))

    def _handle_run_timeout(self):
        if self.process is None or self.current_plan is None:
            return
        self.current_run_timed_out = True
        self._append_output(
            f"\n[Launcher] 第 {self.current_run_index + 1}/{self.current_plan['run_count']} 次运行已超过 "
            f"{self.current_plan['safe_minutes']} 分钟，正在停止本次用例。\n"
        )
        self._set_status("当前用例超过安全时间，正在停止本次运行。")
        self.process.kill()

    def _start_run(self):
        if self.batch_active or self.process is not None:
            QMessageBox.information(self, "运行中", "当前已有任务在运行，请先停止。")
            return

        plan = self._collect_plan()
        if plan is None:
            return

        self._begin_batch(plan)

    def _read_process_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_output(text)

    def _on_process_finished(self, exit_code: int, _exit_status):
        self.run_timeout_timer.stop()
        self._append_output(f"\n[Launcher] 进程结束，exit_code={exit_code}\n")
        self._poll_preview_frame()
        self.preview_timer.stop()
        if self.process is not None:
            self.process.deleteLater()
            self.process = None

        if not self.batch_active or self.current_plan is None:
            self._finish_batch(f"任务已结束，退出码：{exit_code}")
            return

        if self.stop_requested:
            self._finish_batch("任务已停止。")
            return

        self.current_run_index += 1
        if self.current_run_timed_out:
            self._append_output("[Launcher] 本次用例因超过安全时间被停止，计入已执行次数。\n")

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
        if not self.batch_active and self.process is None:
            return

        self.stop_requested = True
        self.safety_timer.stop()
        self.run_timeout_timer.stop()

        if self.process is None:
            self._append_output("\n[Launcher] 已取消后续运行。\n")
            self._finish_batch("任务已停止。")
            return

        self._append_output("\n[Launcher] 正在停止当前子进程，并取消后续运行...\n")
        self.preview_timer.stop()
        self.process.kill()


def _run_helper_command(args: argparse.Namespace) -> int:
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
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-testcase")
    parser.add_argument("--run-direct", nargs=2, metavar=("PROJECT_CASE", "TARGET_CASE"))
    args, _ = parser.parse_known_args()

    if args.run_testcase or args.run_direct:
        raise SystemExit(_run_helper_command(args))

    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
