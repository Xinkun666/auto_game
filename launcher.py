import argparse
import ast
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
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
    QVBoxLayout,
    QWidget,
)


ROOT_DIR = Path(__file__).resolve().parent
TESTCASES_DIR = ROOT_DIR / "testcases"
CUSTOMS_EXAMPLES_DIR = ROOT_DIR / "aw" / "autogame" / "customs_examples"
CUSTOMS_GAME_EXAMPLES_DIR = ROOT_DIR / "aw" / "autogame" / "customs_game_examples"


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


class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.process: Optional[QProcess] = None
        self.selected_testcase_file: Optional[Path] = None
        self._updating_targets = False

        self.setWindowTitle("Auto Game 启动器")
        self.resize(980, 760)

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

        self.start_button = QPushButton("启动")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)

        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("运行输出会显示在这里...")

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
        config_layout.addRow("解析结果", self.status_label)
        config_layout.addRow("", self.refresh_button)
        main_layout.addWidget(config_group)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.stop_button)
        action_layout.addStretch(1)
        main_layout.addLayout(action_layout)

        log_group = QGroupBox("运行输出")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.output_edit)
        main_layout.addWidget(log_group, 1)

    def _bind_signals(self):
        self.mode_testcase.toggled.connect(self._sync_mode_ui)
        self.browse_button.clicked.connect(self._choose_testcase_file)
        self.clear_button.clicked.connect(self._clear_testcase_file)
        self.refresh_button.clicked.connect(self._refresh_config_choices)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        self.start_button.clicked.connect(self._start_run)
        self.stop_button.clicked.connect(self._stop_run)

    def _append_output(self, text: str):
        if not text:
            return
        self.output_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.output_edit.insertPlainText(text)
        self.output_edit.moveCursor(QTextCursor.MoveOperation.End)

    def _set_status(self, text: str):
        self.status_label.setText(text)

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
        return env

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

    def _start_run(self):
        if self.process is not None:
            QMessageBox.information(self, "运行中", "当前已有任务在运行，请先停止。")
            return

        config = self._validate_selection()
        if config is None:
            return

        project_case, target_case = config
        self.output_edit.clear()

        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setWorkingDirectory(str(ROOT_DIR))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.setProcessEnvironment(self._build_process_environment(project_case, target_case))
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._on_process_finished)

        if self.mode_testcase.isChecked():
            if self.selected_testcase_file is None:
                QMessageBox.warning(self, "缺少用例", "testcases 模式下请先选择一个用例文件。")
                self.process.deleteLater()
                self.process = None
                return

            testcase_label = self.selected_testcase_file.relative_to(ROOT_DIR).with_suffix("").as_posix()
            args = [str(ROOT_DIR / "launcher.py"), "--run-testcase", testcase_label]
            self._set_status(
                f"正在通过 testcases 启动：{testcase_label}，"
                f"project_case={project_case}，target_case={target_case}"
            )
        else:
            args = [str(ROOT_DIR / "launcher.py"), "--run-direct", project_case, target_case]
            self._set_status(
                f"正在直接启动自动化：project_case={project_case}，target_case={target_case}"
            )

        self.process.setArguments(args)
        self.process.start()
        started = self.process.waitForStarted(3000)
        if not started:
            QMessageBox.critical(self, "启动失败", "子进程启动失败，请检查 Python 环境。")
            self.process.deleteLater()
            self.process = None
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _read_process_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_output(text)

    def _on_process_finished(self, exit_code: int, _exit_status):
        self._append_output(f"\n[Launcher] 进程结束，exit_code={exit_code}\n")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_status(f"任务已结束，退出码：{exit_code}")
        if self.process is not None:
            self.process.deleteLater()
            self.process = None

    def _stop_run(self):
        if self.process is None:
            return
        self._append_output("\n[Launcher] 正在停止子进程...\n")
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
