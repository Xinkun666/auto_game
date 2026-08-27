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
    ZOOM_STEP = 1.2
    MIN_ZOOM_FACTOR = 0.25
    MAX_ZOOM_FACTOR = 8.0

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self._zoom_factor = 1.0
        self._panning = False
        self._pan_start = None
        self._pan_horizontal_value = 0
        self._pan_vertical_value = 0
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    def zoom_in(self):
        self._zoom_by(self.ZOOM_STEP)

    def zoom_out(self):
        self._zoom_by(1.0 / self.ZOOM_STEP)

    def reset_zoom(self):
        self._zoom_factor = 1.0
        if self.scene() and not self.scene().sceneRect().isEmpty():
            self.fitInView(
                self.scene().sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        else:
            self.resetTransform()

    def _zoom_by(self, factor: float, viewport_position=None):
        if not self.scene() or self.scene().sceneRect().isEmpty() or factor <= 0:
            return
        target_factor = min(
            max(self._zoom_factor * factor, self.MIN_ZOOM_FACTOR),
            self.MAX_ZOOM_FACTOR,
        )
        effective_factor = target_factor / self._zoom_factor
        if abs(effective_factor - 1.0) < 1e-9:
            return
        anchor = (
            viewport_position
            if viewport_position is not None
            else self.viewport().rect().center()
        )
        scene_position_before = self.mapToScene(anchor)
        self.scale(effective_factor, effective_factor)
        scene_position_after = self.mapToScene(anchor)
        delta = scene_position_after - scene_position_before
        self.translate(delta.x(), delta.y())
        self._zoom_factor = target_factor

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y() or event.pixelDelta().y()
            if delta:
                factor = self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP
                self._zoom_by(factor, event.position().toPoint())
            event.accept()
            return
        super().wheelEvent(event)

    def _has_movable_item_at(self, viewport_position) -> bool:
        item = self.itemAt(viewport_position)
        while item is not None:
            if item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable:
                return True
            item = item.parentItem()
        return False

    def mousePressEvent(self, event):
        viewport_position = event.position().toPoint()
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._has_movable_item_at(viewport_position)
        ):
            self._panning = True
            self._pan_start = viewport_position
            self._pan_horizontal_value = self.horizontalScrollBar().value()
            self._pan_vertical_value = self.verticalScrollBar().value()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and event.buttons() & Qt.MouseButton.LeftButton:
            viewport_position = event.position().toPoint()
            delta = viewport_position - self._pan_start
            self.horizontalScrollBar().setValue(
                self._pan_horizontal_value - delta.x()
            )
            self.verticalScrollBar().setValue(self._pan_vertical_value - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self._pan_start = None
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        previous_center = self.mapToScene(self.viewport().rect().center())
        super().resizeEvent(event)
        if self.scene() and not self.scene().sceneRect().isEmpty():
            zoom_factor = self._zoom_factor
            self.fitInView(
                self.scene().sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            if zoom_factor != 1.0:
                self.scale(zoom_factor, zoom_factor)
                self.centerOn(previous_center)


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
            "只需绑定实际录制会用到的控点，其他控点可以保持未绑定。"
            "画面中的蓝色控点可直接拖动，按住图片非控点区域左键可拖动画面；"
            "若标注了 center 和 boundary，"
            "会自动列出 8 个绿色摇杆方向供选择性绑定。"
        )
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet("color: #555;")

        self.binding_list = BindingList()
        self.binding_list.setMinimumWidth(300)
        self.binding_list.currentItemChanged.connect(self._list_selection_changed)
        self.binding_list.keyCaptured.connect(self._bind_current_key)
        left_title = QLabel("控点 → 键盘按键")
        left_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        left_hint = QLabel("全部可绑定控点都会列出；选中后按键绑定，也可清除绑定")
        left_hint.setWordWrap(True)
        left_hint.setStyleSheet("color: #777;")
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(left_title)
        left_layout.addWidget(left_hint)
        left_layout.addWidget(self.binding_list, 1)
        self.clear_binding_button = QPushButton("清除当前绑定")
        self.clear_binding_button.clicked.connect(self._clear_current_binding)
        self.clear_binding_button.setEnabled(False)
        left_layout.addWidget(self.clear_binding_button)

        self.graphics_scene = QGraphicsScene(self)
        self.view = SceneView(self.graphics_scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.setBackgroundBrush(QBrush(QColor("#20242a")))

        self.zoom_in_button = QPushButton("放大")
        self.zoom_out_button = QPushButton("缩小")
        self.zoom_reset_button = QPushButton("恢复")
        zoom_buttons = (
            self.zoom_in_button,
            self.zoom_out_button,
            self.zoom_reset_button,
        )
        for button in zoom_buttons:
            button.setMinimumWidth(72)
            button.setMinimumHeight(32)
        self.zoom_in_button.setToolTip("放大当前场景图（Ctrl + 滚轮向上）")
        self.zoom_out_button.setToolTip("缩小当前场景图（Ctrl + 滚轮向下）")
        self.zoom_reset_button.setToolTip("恢复为适合当前窗口的大小")
        self.zoom_in_button.clicked.connect(self.view.zoom_in)
        self.zoom_out_button.clicked.connect(self.view.zoom_out)
        self.zoom_reset_button.clicked.connect(self.view.reset_zoom)

        zoom_hint = QLabel("Ctrl + 滚轮：以鼠标所在位置为中心缩放")
        zoom_hint.setStyleSheet("color: #777;")
        zoom_controls = QHBoxLayout()
        zoom_controls.addWidget(zoom_hint)
        zoom_controls.addStretch(1)
        zoom_controls.addWidget(self.zoom_in_button)
        zoom_controls.addWidget(self.zoom_out_button)
        zoom_controls.addWidget(self.zoom_reset_button)

        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.addWidget(self.view, 1)
        right_panel.addLayout(zoom_controls)

        content = QHBoxLayout()
        content.addWidget(left_panel)
        content.addLayout(right_panel, 1)

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
        self.view.reset_zoom()

    def _refresh_list_item(self, item: QListWidgetItem):
        scene = self.current_scene
        if scene is None:
            return
        point_name = str(item.data(Qt.ItemDataRole.UserRole))
        bound_key = self.config.binding_for(scene, point_name)
        key = bound_key or "未绑定（可选）"
        control = self.config.control_for(scene, point_name)
        display_name = control.display_name if control is not None else point_name
        item.setText(f"{display_name}    →    {key}")
        item.setForeground(
            QBrush(QColor("#202020" if bound_key else "#777777"))
        )

    def _list_selection_changed(self, current, previous):
        del previous
        if current is None:
            self.clear_binding_button.setEnabled(False)
            return
        if self._syncing_selection:
            return
        self._update_clear_binding_button()
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
                self._update_clear_binding_button()
                break

    def _update_clear_binding_button(self):
        item = self.binding_list.currentItem()
        scene = self.current_scene
        self.clear_binding_button.setEnabled(
            item is not None
            and scene is not None
            and bool(
                self.config.binding_for(
                    scene,
                    str(item.data(Qt.ItemDataRole.UserRole)),
                )
            )
        )

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
        self.clear_binding_button.setEnabled(True)

    def _clear_current_binding(self):
        item = self.binding_list.currentItem()
        scene = self.current_scene
        if item is None or scene is None:
            return
        point_name = str(item.data(Qt.ItemDataRole.UserRole))
        try:
            self.config.unbind_key(scene, point_name)
        except BindingConfigError as exc:
            self.status_label.setText(str(exc))
            self.status_label.setStyleSheet("color: #b00020;")
            return
        self._refresh_list_item(item)
        self.clear_binding_button.setEnabled(False)
        self.status_label.setText(f"已清除控点“{point_name}”的按键绑定。")
        self.status_label.setStyleSheet("color: #167c36;")

    def _save(self):
        try:
            self.config.save()
        except (BindingConfigError, OSError) as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        self.accept()
