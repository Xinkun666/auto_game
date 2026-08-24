"""start_record 启动前显示的按键绑定与控点编辑窗口。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QKeyEvent, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from .binding_config import BindingConfigError, BindingConfiguration, BindingScene
from .layout import RESERVED_KEYS


SPECIAL_KEYS = {
    Qt.Key.Key_Space: "space",
    Qt.Key.Key_Up: "up",
    Qt.Key.Key_Down: "down",
    Qt.Key.Key_Left: "left",
    Qt.Key.Key_Right: "right",
    Qt.Key.Key_Return: "enter",
    Qt.Key.Key_Enter: "enter",
    Qt.Key.Key_Shift: "shift",
    Qt.Key.Key_Control: "ctrl",
    Qt.Key.Key_Alt: "alt",
    Qt.Key.Key_Tab: "tab",
    Qt.Key.Key_Backspace: "backspace",
    Qt.Key.Key_Escape: "escape",
}


def key_name_from_event(event: QKeyEvent) -> str | None:
    special = SPECIAL_KEYS.get(event.key())
    if special:
        return special
    text = event.text().strip().lower()
    return text if len(text) == 1 else None


class BindingList(QListWidget):
    keyCaptured = pyqtSignal(str)

    def keyPressEvent(self, event: QKeyEvent):
        if self.currentItem() is not None:
            key = key_name_from_event(event)
            if key:
                self.keyCaptured.emit(key)
                event.accept()
                return
        super().keyPressEvent(event)


class PointItem(QGraphicsEllipseItem):
    def __init__(
        self,
        name: str,
        position: QPointF,
        scene_width: float,
        scene_height: float,
        changed: Callable[[str, QPointF], None],
        selected: Callable[[str], None],
    ):
        # 绑定窗口按“屏幕像素”显示标记；32px 在缩放后的手机截图上过于遮挡。
        super().__init__(-7, -7, 14, 14)
        self.point_name = name
        self.scene_width = scene_width
        self.scene_height = scene_height
        self.changed = changed
        self.selected = selected
        self._initialized = False
        self.setBrush(QBrush(QColor(40, 146, 255, 210)))
        self.setPen(QPen(QColor("white"), 1.5))
        self.setZValue(10)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        label = QGraphicsSimpleTextItem(name, self)
        label.setBrush(QBrush(QColor("white")))
        label.setPos(10, -7)
        self.setPos(position)
        self._initialized = True

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            point = value
            return QPointF(
                min(max(float(point.x()), 0.0), self.scene_width),
                min(max(float(point.y()), 0.0), self.scene_height),
            )
        result = super().itemChange(change, value)
        if (
            self._initialized
            and change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
        ):
            self.changed(self.point_name, self.pos())
        elif (
            self._initialized
            and change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and bool(value)
        ):
            self.selected(self.point_name)
        return result


class SceneView(QGraphicsView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() and not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class BindingDialog(QDialog):
    """每次启动都显示，但自动加载 info.py 中上次保存的绑定。"""

    def __init__(self, info_module, parent=None):
        super().__init__(parent)
        self.config = BindingConfiguration(info_module)
        self.current_index = 0
        self.point_items: dict[str, PointItem] = {}
        self._syncing_selection = False

        self.setWindowTitle("Game Recording - 按键绑定与控点调整")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        self.stage_label = QLabel(f"当前阶段：{self.config.stage_name}")
        self.scene_label = QLabel()
        self.scene_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scene_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.previous_button = QPushButton("← 上一个场景")
        self.next_button = QPushButton("下一个场景 →")
        self.previous_button.clicked.connect(lambda: self._switch_scene(-1))
        self.next_button.clicked.connect(lambda: self._switch_scene(1))

        navigation = QHBoxLayout()
        navigation.addWidget(self.stage_label)
        navigation.addStretch(1)
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.scene_label, 1)
        navigation.addWidget(self.next_button)

        self.help_label = QLabel(
            "先在左侧选中一个控点，再按下要绑定的键盘按键；"
            "画面中的蓝色控点可直接拖动。若标注了 center 和 boundary，"
            "会自动生成 8 个绿色摇杆方向供选择性绑定。q/e 现在也可以绑定为普通游戏键。"
        )
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet("color: #555;")

        self.binding_list = BindingList()
        self.binding_list.setMinimumWidth(300)
        self.binding_list.currentItemChanged.connect(self._list_selection_changed)
        self.binding_list.keyCaptured.connect(self._bind_current_key)
        left_title = QLabel("控点 → 键盘按键")
        left_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        left_hint = QLabel("选中一项后，直接按键即可更换绑定")
        left_hint.setWordWrap(True)
        left_hint.setStyleSheet("color: #777;")
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(left_title)
        left_layout.addWidget(left_hint)
        left_layout.addWidget(self.binding_list, 1)

        self.graphics_scene = QGraphicsScene(self)
        self.view = SceneView(self.graphics_scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.setBackgroundBrush(QBrush(QColor("#20242a")))

        content = QHBoxLayout()
        content.addWidget(left_panel)
        content.addWidget(self.view, 1)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #167c36;")

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存并进入录制")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(navigation)
        root.addWidget(self.help_label)
        root.addLayout(content, 1)
        root.addWidget(self.status_label)
        root.addWidget(self.buttons)
        self._load_scene()

    @property
    def current_scene(self) -> BindingScene | None:
        if not self.config.scenes:
            return None
        return self.config.scenes[self.current_index]

    def _switch_scene(self, offset: int):
        if len(self.config.scenes) <= 1:
            return
        self.current_index = (self.current_index + offset) % len(self.config.scenes)
        self._load_scene()

    def _load_scene(self):
        self.graphics_scene.clear()
        self.binding_list.clear()
        self.point_items.clear()
        scene = self.current_scene
        count = len(self.config.scenes)
        self.previous_button.setEnabled(count > 1)
        self.next_button.setEnabled(count > 1)
        if scene is None:
            self.scene_label.setText("尚未标注场景")
            self.status_label.setText("请先用 Label 标注工具导出至 Game_Recording。")
            self.status_label.setStyleSheet("color: #b00020;")
            self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
            return

        self.scene_label.setText(
            f"{self.current_index + 1}/{count}  {scene.display_name}"
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(True)
        scene_width = float(scene.width)
        scene_height = float(scene.height)
        self.graphics_scene.setSceneRect(0, 0, scene_width, scene_height)

        image_path = self.config.project_dir / Path(scene.image)
        pixmap = QPixmap(str(image_path)) if image_path.is_file() else QPixmap()
        if not pixmap.isNull():
            background = self.graphics_scene.addPixmap(pixmap)
            background.setScale(min(scene_width / pixmap.width(), scene_height / pixmap.height()))
            background.setZValue(-10)
            self.status_label.setText("已加载上次绑定；如果无需修改，直接点击“保存并进入录制”。")
            self.status_label.setStyleSheet("color: #167c36;")
        else:
            self.graphics_scene.addRect(
                0,
                0,
                scene_width,
                scene_height,
                QPen(QColor("#555")),
                QBrush(QColor("#282d34")),
            ).setZValue(-10)
            warning = self.graphics_scene.addSimpleText(
                f"场景图片未找到\n{scene.image or 'info.py 未设置 image'}"
            )
            warning.setBrush(QBrush(QColor("#dddddd")))
            warning.setPos(scene_width * 0.05, scene_height * 0.05)
            warning.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                True,
            )
            self.status_label.setText("场景图片缺失，仍可绑定和拖动控点。")
            self.status_label.setStyleSheet("color: #a05a00;")

        controls = {
            control.name: control for control in self.config.bindable_controls(scene)
        }
        for point_name, point_data in scene.points.items():
            rect = point_data.get("rect") if isinstance(point_data, dict) else None
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                continue
            try:
                norm_x = (float(rect[0]) + float(rect[2])) / 2.0
                norm_y = (float(rect[1]) + float(rect[3])) / 2.0
            except (TypeError, ValueError):
                continue
            if str(point_name) in controls:
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, str(point_name))
                self.binding_list.addItem(item)
                self._refresh_list_item(item)
            point_item = PointItem(
                str(point_name),
                QPointF(norm_x * scene_width, norm_y * scene_height),
                scene_width,
                scene_height,
                changed=self._point_moved,
                selected=self._point_selected,
            )
            self.graphics_scene.addItem(point_item)
            self.point_items[str(point_name)] = point_item

        for control in controls.values():
            if not control.is_virtual_joystick_direction:
                continue
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, control.name)
            self.binding_list.addItem(item)
            self._refresh_list_item(item)

            position = QPointF(
                control.normalized_position[0] * scene_width,
                control.normalized_position[1] * scene_height,
            )
            marker = self.graphics_scene.addEllipse(
                -5,
                -5,
                10,
                10,
                QPen(QColor("#d8ffe3"), 1.5),
                QBrush(QColor(30, 170, 90, 220)),
            )
            marker.setPos(position)
            marker.setZValue(11)
            marker.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                True,
            )
            label = self.graphics_scene.addSimpleText(control.display_name)
            label.setBrush(QBrush(QColor("#d8ffe3")))
            label.setPos(position + QPointF(8, -7))
            label.setZValue(11)
            label.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                True,
            )

        if self.binding_list.count():
            self.binding_list.setCurrentRow(0)
        self.view.fitInView(self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _refresh_list_item(self, item: QListWidgetItem):
        scene = self.current_scene
        if scene is None:
            return
        point_name = str(item.data(Qt.ItemDataRole.UserRole))
        bound_key = self.config.binding_for(scene, point_name)
        key = bound_key or "还没绑定"
        control = self.config.control_for(scene, point_name)
        display_name = control.display_name if control is not None else point_name
        item.setText(f"{display_name}    →    {key}")
        item.setForeground(
            QBrush(QColor("#202020" if bound_key else "#b00020"))
        )

    def _list_selection_changed(self, current, previous):
        del previous
        if self._syncing_selection or current is None:
            return
        point_name = str(current.data(Qt.ItemDataRole.UserRole))
        point_item = self.point_items.get(point_name)
        if point_item is not None:
            self._syncing_selection = True
            self.graphics_scene.clearSelection()
            point_item.setSelected(True)
            self._syncing_selection = False

    def _point_selected(self, point_name: str):
        if self._syncing_selection:
            return
        for row in range(self.binding_list.count()):
            item = self.binding_list.item(row)
            if str(item.data(Qt.ItemDataRole.UserRole)) == point_name:
                self._syncing_selection = True
                self.binding_list.setCurrentItem(item)
                self.binding_list.setFocus()
                self._syncing_selection = False
                break

    def _point_moved(self, point_name: str, position: QPointF):
        scene = self.current_scene
        if scene is None:
            return
        self.config.set_point_center(
            scene,
            point_name,
            position.x() / scene.width,
            position.y() / scene.height,
        )
        self.status_label.setText(
            f"已移动控点“{point_name}”，点击保存后会同步写入 info.py。"
        )
        self.status_label.setStyleSheet("color: #167c36;")

    def _bind_current_key(self, key: str):
        item = self.binding_list.currentItem()
        scene = self.current_scene
        if item is None or scene is None:
            return
        point_name = str(item.data(Qt.ItemDataRole.UserRole))
        try:
            self.config.bind_key(scene, point_name, key)
        except BindingConfigError as exc:
            self.status_label.setText(str(exc))
            self.status_label.setStyleSheet("color: #b00020;")
            return
        for row in range(self.binding_list.count()):
            self._refresh_list_item(self.binding_list.item(row))
        if key in RESERVED_KEYS:
            return
        self.status_label.setText(f"控点“{point_name}”已绑定到键盘 {key}。")
        self.status_label.setStyleSheet("color: #167c36;")

    def _save(self):
        try:
            self.config.save()
        except (BindingConfigError, OSError) as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        self.accept()
