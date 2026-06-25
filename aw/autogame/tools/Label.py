import sys
import json
import random
import os
import subprocess
import tempfile
import shutil
import ast
import re
import keyword
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton,
                             QMenu, QFileDialog, QInputDialog, QLabel, QSplitter,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsLineItem, QToolBar, QMessageBox, QFrame,
                             QPinchGesture, QHeaderView, QProgressDialog, QComboBox, QDialog,
                             QLineEdit, QCheckBox, QScrollArea, QDialogButtonBox)
from PyQt6.QtCore import Qt, QRectF, QPointF, QEvent
from PyQt6.QtGui import QAction, QPixmap, QColor, QPen, QBrush, QImage, QPainter, QGuiApplication, QFontMetricsF
from aw.autogame.tools.AreaResolver import resolve_area_rect_for_frame
from aw.autogame.tools.ProcessUtils import hidden_subprocess_kwargs
# ==========================================
# 1. 数据模型 (Data Structure)
# ==========================================
DEFAULT_GROUP_NAME = "默认"
GROUPABLE_ITEM_TYPES = ("area", "special_area")


@dataclass
class RectData:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class GroupItemRef:
    scene_name: str
    item_type: str
    item_name: str


@dataclass
class GroupData:
    name: str
    items: List[GroupItemRef] = field(default_factory=list)
    includes_all: bool = False


def default_stage_groups():
    return [GroupData(name=DEFAULT_GROUP_NAME, includes_all=True)]


@dataclass
class ItemData:
    id: str
    name: str
    item_type: str  # 'area' or 'control'
    rect: RectData  # 核心坐标
    search_scope: Optional[RectData] = None  # 搜索范围 (仅Area有效)
    visible: bool = True
    match_mode: str = "gray"
@dataclass
class SceneData:
    id: str
    name: str
    image_path: str = ""  # 暂时存储路径，实际可能存Base64或相对路径
    pixmap: Optional[QPixmap] = None  # 仅用于UI显示的内存图片
    image_width: int = 0
    image_height: int = 0
    items: List[ItemData] = field(default_factory=list)
@dataclass
class StageData:
    id: str
    name: str
    scenes: List[SceneData] = field(default_factory=list)
    groups: List[GroupData] = field(default_factory=default_stage_groups)
    active_group_name: str = DEFAULT_GROUP_NAME
@dataclass
class ProjectData:
    name: str
    stages: List[StageData] = field(default_factory=list)
@dataclass
class CaptureResolutionApplyResult:
    scene: SceneData
    action: str
    resized_items: bool = False
    source_scene: Optional[SceneData] = None
# ==========================================
# 2. 图形工作区 (Canvas / Image Workspace)
# ==========================================
class DrawingOverlay(QGraphicsRectItem):
    """用于显示标注框的自定义图形项"""
    def __init__(self, x, y, w, h, item_type, label, parent=None):
        super().__init__(x, y, w, h, parent)
        self.item_type = item_type
        self.label = label
        if item_type == 'area':
            self.setPen(QPen(Qt.GlobalColor.blue, 3))
            self.setBrush(QBrush(QColor(0, 0, 255, 60)))
        elif item_type == 'special_area':
            self.setPen(QPen(QColor(255, 140, 0), 3))
            self.setBrush(QBrush(QColor(255, 140, 0, 40)))
        elif item_type == 'search_scope':
            self.setPen(QPen(Qt.GlobalColor.green, 1, Qt.PenStyle.DashLine))
            self.setBrush(QBrush(Qt.GlobalColor.transparent))
        else:  # control
            self.setPen(QPen(Qt.GlobalColor.red, 2))
            self.setBrush(QBrush(QColor(255, 0, 0, 30)))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable)
    def _label_baseline(self):
        rect = self.rect()
        if self.item_type == 'search_scope':
            return QPointF(rect.left() + 5, rect.bottom() - 5)
        return QPointF(rect.left() + 5, rect.top() + 15)
    def _label_rect(self):
        fm = QFontMetricsF(QApplication.font())
        baseline = self._label_baseline()
        width = fm.horizontalAdvance(self.label)
        top = baseline.y() - fm.ascent()
        return QRectF(baseline.x(), top, width, fm.height())
    def boundingRect(self):
        base_rect = super().boundingRect()
        label_rect = self._label_rect().adjusted(-4, -2, 4, 2)
        return base_rect.united(label_rect)
    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        # 绘制标签文字，并给文字加底板以避免与移动引导线混叠
        text_pos = self._label_baseline()
        text_bg_rect = self._label_rect().adjusted(-3, -1, 3, 1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRect(text_bg_rect)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(text_pos, self.label)
class ImageCanvas(QGraphicsView):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.grabGesture(Qt.GestureType.PinchGesture)
        self.current_pixmap = None
        self.mode = "IDLE"  # IDLE, DRAW_AREA, DRAW_CONTROL, DRAW_SCOPE
        self.start_point = None
        self.current_draw_item = None
        self.temp_rect_item = None
        self.active_scene_data = None
        self.target_item_data = None  # 用于修改Scope时的目标
        self.current_scale = 1.0
        self.min_scale = 0.2
        self.max_scale = 5.0
        self.user_zoomed = False
        self.crosshair_h = None
        self.crosshair_v = None
        self.drag_item_data = None
        self.drag_start_point = None
        self.drag_start_rect = None
        self.drag_start_scope = None
    def is_point_on_image(self, pt: QPointF):
        if not self.current_pixmap:
            return False
        pixmap = self.current_pixmap.pixmap()
        if pixmap.isNull():
            return False
        return 0 <= pt.x() <= pixmap.width() and 0 <= pt.y() <= pixmap.height()
    def set_image(self, pixmap):
        self.scene.clear()
        self.current_pixmap = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.init_crosshair_items()
        self.hide_crosshair()
        self.user_zoomed = False
        self.fit_image_to_view()
        self.main_window.update_coord_display(None, None)
    def init_crosshair_items(self):
        pen = QPen(QColor(0, 255, 0, 200), 1, Qt.PenStyle.DashLine)
        self.crosshair_h = QGraphicsLineItem()
        self.crosshair_h.setPen(pen)
        self.crosshair_h.setZValue(1000)
        self.crosshair_h.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene.addItem(self.crosshair_h)
        self.crosshair_v = QGraphicsLineItem()
        self.crosshair_v.setPen(pen)
        self.crosshair_v.setZValue(1000)
        self.crosshair_v.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene.addItem(self.crosshair_v)
    def hide_crosshair(self):
        if self.crosshair_h:
            self.crosshair_h.setVisible(False)
        if self.crosshair_v:
            self.crosshair_v.setVisible(False)
    def update_crosshair(self, pt: QPointF):
        if not self.current_pixmap or not self.crosshair_h or not self.crosshair_v:
            self.hide_crosshair()
            self.main_window.update_coord_display(None, None)
            return
        pixmap = self.current_pixmap.pixmap()
        if pixmap.isNull():
            self.hide_crosshair()
            self.main_window.update_coord_display(None, None)
            return
        w = pixmap.width()
        h = pixmap.height()
        if pt.x() < 0 or pt.y() < 0 or pt.x() > w or pt.y() > h:
            self.hide_crosshair()
            self.main_window.update_coord_display(None, None)
            return
        self.crosshair_h.setLine(0, pt.y(), w, pt.y())
        self.crosshair_v.setLine(pt.x(), 0, pt.x(), h)
        self.crosshair_h.setVisible(True)
        self.crosshair_v.setVisible(True)
        self.main_window.update_coord_display(pt.x(), pt.y())
    def fit_image_to_view(self):
        if not self.current_pixmap:
            return
        pixmap = self.current_pixmap.pixmap()
        if pixmap.isNull():
            return
        viewport_size = self.viewport().size()
        if viewport_size.width() <= 0 or viewport_size.height() <= 0:
            return
        scale_w = viewport_size.width() / pixmap.width()
        scale_h = viewport_size.height() / pixmap.height()
        scale = min(scale_w, scale_h)
        scale = max(self.min_scale, min(self.max_scale, scale))
        self.resetTransform()
        self.scale(scale, scale)
        self.current_scale = scale
    def zoom_by_factor(self, factor):
        new_scale = self.current_scale * factor
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        self.resetTransform()
        self.scale(new_scale, new_scale)
        self.current_scale = new_scale
        self.user_zoomed = True
    def zoom_in(self):
        self.zoom_by_factor(1.15)
    def zoom_out(self):
        self.zoom_by_factor(1 / 1.15)
    def redraw_overlays(self, scene_data: SceneData):
        """根据数据重新绘制所有框"""
        # 清除旧的框 (保留图片)
        for item in self.scene.items():
            if isinstance(item, DrawingOverlay) or isinstance(item, QGraphicsRectItem):
                self.scene.removeItem(item)
        if not scene_data:
            return
        stage = self.main_window._find_stage_for_scene(scene_data)
        for item in scene_data.items:
            if not item.visible:
                continue
            if not self.main_window._is_item_visible_in_stage_group(stage, scene_data, item):
                continue
            # 绘制本体
            r = item.rect
            overlay = DrawingOverlay(r.x, r.y, r.w, r.h, item.item_type, item.name)
            overlay.setData(0, item)
            self.scene.addItem(overlay)
            # 绘制搜索范围 (如果存在且当前选中了该区域或开启了调试显示)
            if item.item_type == 'area' and item.search_scope:
                s = item.search_scope
                scope = DrawingOverlay(s.x, s.y, s.w, s.h, 'search_scope', f"Scope: {item.name}")
                scope.setData(0, item)
                self.scene.addItem(scope)
    def mousePressEvent(self, event):
        if self.mode == "IDLE" and self.current_pixmap and event.button() == Qt.MouseButton.RightButton:
            pt = self.mapToScene(event.pos())
            for item in self.scene.items(pt):
                if isinstance(item, DrawingOverlay):
                    data = item.data(0)
                    if isinstance(data, ItemData):
                        self.main_window.select_item_in_tree(data)
                        self.main_window.show_item_context_menu(data, event.globalPosition().toPoint())
                        event.accept()
                        return
                    break
            if self.is_point_on_image(pt):
                self.main_window.show_canvas_context_menu(event.globalPosition().toPoint())
                event.accept()
                return
        if self.mode == "IDLE" and self.current_pixmap and event.button() == Qt.MouseButton.LeftButton:
            pt = self.mapToScene(event.pos())
            for item in self.scene.items(pt):
                if isinstance(item, DrawingOverlay):
                    data = item.data(0)
                    if data:
                        self.main_window.select_item_in_tree(data)
                        if isinstance(data, ItemData) and item.item_type in ["area", "control", "special_area", "search_scope"]:
                            self.drag_item_data = data
                            self.drag_start_point = pt
                            self.drag_start_rect = RectData(data.rect.x, data.rect.y, data.rect.w, data.rect.h)
                            if data.item_type == "area" and data.search_scope:
                                scope = data.search_scope
                                self.drag_start_scope = RectData(scope.x, scope.y, scope.w, scope.h)
                            else:
                                self.drag_start_scope = None
                            self.setCursor(Qt.CursorShape.SizeAllCursor)
                            event.accept()
                            return
                        break
        if self.mode != "IDLE" and self.current_pixmap:
            pt = self.mapToScene(event.pos())
            self.start_point = pt
            self.temp_rect_item = QGraphicsRectItem()
            self.temp_rect_item.setPen(QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.DashLine))
            self.scene.addItem(self.temp_rect_item)
        else:
            super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        current_pt = self.mapToScene(event.pos())
        self.update_crosshair(current_pt)
        if self.mode == "IDLE" and self.drag_item_data and self.drag_start_point and self.drag_start_rect:
            dx = current_pt.x() - self.drag_start_point.x()
            dy = current_pt.y() - self.drag_start_point.y()
            new_x = self.drag_start_rect.x + dx
            new_y = self.drag_start_rect.y + dy
            pixmap = self.current_pixmap.pixmap() if self.current_pixmap else None
            if pixmap and not pixmap.isNull():
                max_x = max(0.0, pixmap.width() - self.drag_start_rect.w)
                max_y = max(0.0, pixmap.height() - self.drag_start_rect.h)
                new_x = min(max(new_x, 0.0), max_x)
                new_y = min(max(new_y, 0.0), max_y)
            actual_dx = new_x - self.drag_start_rect.x
            actual_dy = new_y - self.drag_start_rect.y
            self.drag_item_data.rect = RectData(new_x, new_y, self.drag_start_rect.w, self.drag_start_rect.h)
            if self.drag_item_data.item_type == "area" and self.drag_start_scope:
                self.drag_item_data.search_scope = RectData(
                    self.drag_start_scope.x + actual_dx,
                    self.drag_start_scope.y + actual_dy,
                    self.drag_start_scope.w,
                    self.drag_start_scope.h,
                )
            if self.active_scene_data:
                self.redraw_overlays(self.active_scene_data)
            event.accept()
            return
        if self.mode != "IDLE" and self.start_point:
            rect = QRectF(self.start_point, current_pt).normalized()
            self.temp_rect_item.setRect(rect)
        else:
            super().mouseMoveEvent(event)
    def leaveEvent(self, event):
        self.hide_crosshair()
        self.main_window.update_coord_display(None, None)
        super().leaveEvent(event)
    def mouseReleaseEvent(self, event):
        if self.mode == "IDLE" and self.drag_item_data and event.button() == Qt.MouseButton.LeftButton:
            moved_item = self.drag_item_data
            self.drag_item_data = None
            self.drag_start_point = None
            self.drag_start_rect = None
            self.drag_start_scope = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if moved_item.item_type == "area":
                self.main_window.status_label.setText(f"已移动区域 {moved_item.name}，搜索范围已同步移动。")
            elif moved_item.item_type == "special_area":
                self.main_window.status_label.setText(f"已移动特殊区域 {moved_item.name}。")
            else:
                self.main_window.status_label.setText(f"已移动控点 {moved_item.name}。")
            event.accept()
            return
        if self.mode != "IDLE" and self.start_point:
            end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, end_point).normalized()
            self.scene.removeItem(self.temp_rect_item)
            self.start_point = None
            # 回调主窗口完成绘制
            self.main_window.finish_drawing(rect)
        else:
            super().mouseReleaseEvent(event)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.user_zoomed:
            self.fit_image_to_view()
    def wheelEvent(self, event):
        wheel_pos = event.position().toPoint()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            zoom_factor = 1.15 if delta > 0 else 1 / 1.15
            self.zoom_by_factor(zoom_factor)
            self.update_crosshair(self.mapToScene(wheel_pos))
            return
        super().wheelEvent(event)
        self.update_crosshair(self.mapToScene(wheel_pos))
    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            gesture_event = event
            pinch = gesture_event.gesture(Qt.GestureType.PinchGesture)
            if isinstance(pinch, QPinchGesture):
                zoom_factor = pinch.scaleFactor()
                self.zoom_by_factor(zoom_factor)
                return True
        return super().event(event)
# ==========================================
# 3. 主界面逻辑 (Main Window)
# ==========================================
class AutoStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GameAuto Studio - 游戏自动化标注工具")
        self.resize(1200, 800)
        self.project = None
        self.imported_resource_dir = None
        self.current_stage = None
        self.current_work_stage = None
        self.current_scene = None
        self.scene_clipboard = None
        self.last_expand_stage_id = None
        self.last_expand_scene_id = None
        self.init_ui()
    def init_ui(self):
        # 1. 顶部菜单
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')
        new_act = QAction('新建项目', self)
        new_act.triggered.connect(self.new_project)
        file_menu.addAction(new_act)
        rename_act = QAction('修改项目名', self)
        rename_act.triggered.connect(self.rename_project)
        file_menu.addAction(rename_act)
        export_act = QAction('导出项目', self)
        export_act.triggered.connect(self.export_project)
        file_menu.addAction(export_act)
        import_act = QAction('导入项目', self)
        import_act.triggered.connect(self.import_project)
        file_menu.addAction(import_act)
        self.add_area_shortcut = QAction("添加区域快捷键", self)
        self.add_area_shortcut.setShortcut("Ctrl+A")
        self.add_area_shortcut.triggered.connect(lambda: self.trigger_add_shortcut("area"))
        self.addAction(self.add_area_shortcut)
        self.add_control_shortcut = QAction("添加控点快捷键", self)
        self.add_control_shortcut.setShortcut("Ctrl+C")
        self.add_control_shortcut.triggered.connect(lambda: self.trigger_add_shortcut("control"))
        self.addAction(self.add_control_shortcut)
        self.add_special_shortcut = QAction("添加特殊区域快捷键", self)
        self.add_special_shortcut.setShortcut("Ctrl+S")
        self.add_special_shortcut.triggered.connect(lambda: self.trigger_add_shortcut("special_area"))
        self.addAction(self.add_special_shortcut)
        # 2. 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        project_toolbar = QHBoxLayout()
        self.btn_new_project = QPushButton("新建项目")
        self.btn_new_project.clicked.connect(self.new_project)
        self.btn_rename_project = QPushButton("修改项目名")
        self.btn_rename_project.clicked.connect(self.rename_project)
        self.btn_import_project = QPushButton("导入项目")
        self.btn_import_project.clicked.connect(self.import_project)
        self.btn_export_project = QPushButton("导出项目")
        self.btn_export_project.clicked.connect(self.export_project)
        project_toolbar.addWidget(self.btn_new_project)
        project_toolbar.addWidget(self.btn_rename_project)
        project_toolbar.addWidget(self.btn_import_project)
        project_toolbar.addWidget(self.btn_export_project)
        project_toolbar.addStretch()
        layout.addLayout(project_toolbar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        # 3. 左侧：图片工作区
        self.canvas = ImageCanvas(self)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        canvas_toolbar = QHBoxLayout()
        btn_fit = QPushButton("适配")
        btn_fit.clicked.connect(self.canvas.fit_image_to_view)
        btn_zoom_in = QPushButton("放大")
        btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        btn_zoom_out = QPushButton("缩小")
        btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        canvas_toolbar.addWidget(btn_fit)
        canvas_toolbar.addWidget(btn_zoom_in)
        canvas_toolbar.addWidget(btn_zoom_out)
        canvas_toolbar.addStretch()
        canvas_toolbar.addWidget(QLabel("分组名:"))
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(140)
        self.group_combo.currentTextChanged.connect(self.on_group_combo_changed)
        canvas_toolbar.addWidget(self.group_combo)
        self.btn_add_group = QPushButton("添加")
        self.btn_add_group.clicked.connect(self.add_group)
        canvas_toolbar.addWidget(self.btn_add_group)
        self.btn_edit_group = QPushButton("修改")
        self.btn_edit_group.clicked.connect(self.edit_current_group)
        canvas_toolbar.addWidget(self.btn_edit_group)
        self.btn_delete_group = QPushButton("删除")
        self.btn_delete_group.clicked.connect(self.delete_current_group)
        canvas_toolbar.addWidget(self.btn_delete_group)
        self.status_label = QLabel("欢迎使用。请新建工程。")
        self.status_label.setStyleSheet("background-color: #ddd; padding: 5px;")
        self.coord_label = QLabel("坐标: (-,-)")
        self.coord_label.setStyleSheet("background-color: #eee; padding: 5px;")
        left_layout.addLayout(canvas_toolbar)
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.coord_label)
        # 4. 右侧：项目树
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        # 控制按钮栏
        self.btn_layout = QHBoxLayout()
        self.btn_add_stage = QPushButton("新建阶段")
        self.btn_add_stage.clicked.connect(self.add_stage)
        self.btn_add_stage.setEnabled(False)
        self.btn_layout.addWidget(self.btn_add_stage)
        right_layout.addLayout(self.btn_layout)
        tree_toolbar = QHBoxLayout()
        tree_toolbar.addWidget(QLabel("项目层级"))
        tree_toolbar.addStretch()
        btn_expand_tree = QPushButton("展开")
        btn_expand_tree.clicked.connect(self.expand_all_tree)
        btn_collapse_tree = QPushButton("折叠")
        btn_collapse_tree.clicked.connect(self.collapse_all_tree)
        tree_toolbar.addWidget(btn_expand_tree)
        tree_toolbar.addWidget(btn_collapse_tree)
        right_layout.addLayout(tree_toolbar)
        # 树控件
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["项目层级", "属性"])
        self.tree.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemClicked.connect(self.on_tree_click)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.open_context_menu)
        right_layout.addWidget(self.tree)
        # 操作面板 (动态显示)
        self.action_panel = QFrame()
        self.action_layout = QVBoxLayout(self.action_panel)
        right_layout.addWidget(self.action_panel)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 85)
        splitter.setStretchFactor(1, 15)
        layout.addWidget(splitter, 1)
        self._updating_group_combo = False
        self.update_group_controls()
    def trigger_add_shortcut(self, mode):
        if not self.current_scene:
            QMessageBox.warning(self, "提示", "请先在项目树中选择一个场景。")
            return
        self.prepare_draw(mode)
    # --- 逻辑处理 ---
    def new_project(self):
        name, ok = QInputDialog.getText(self, "新建工程", "工程名称:")
        if ok and name:
            self.project = ProjectData(name=name)
            self.imported_resource_dir = None
            self.current_stage = None
            self.set_current_work_stage(None)
            self.current_scene = None
            self.update_tree_view()
            self.btn_add_stage.setEnabled(True)
            self.status_label.setText(f"工程 {name} 已创建。请添加阶段。")
    def set_current_work_stage(self, stage: Optional[StageData]):
        self.current_work_stage = stage
        if hasattr(self, "group_combo"):
            self.update_group_controls()

    def _ensure_stage_default_group(self, stage: Optional[StageData]):
        if not stage:
            return
        if not stage.groups:
            stage.groups = default_stage_groups()
        default_group = None
        custom_groups = []
        for group in stage.groups:
            if group.name == DEFAULT_GROUP_NAME:
                if default_group is None:
                    default_group = group
                continue
            custom_groups.append(group)
        if default_group is None:
            default_group = GroupData(name=DEFAULT_GROUP_NAME, includes_all=True)
        default_group.name = DEFAULT_GROUP_NAME
        default_group.includes_all = True
        default_group.items = []
        stage.groups = [default_group] + custom_groups
        if not self._get_stage_group(stage, stage.active_group_name):
            stage.active_group_name = DEFAULT_GROUP_NAME

    def _get_stage_group(self, stage: Optional[StageData], group_name: Optional[str]) -> Optional[GroupData]:
        if not stage:
            return None
        if not group_name:
            group_name = DEFAULT_GROUP_NAME
        for group in stage.groups:
            if group.name == group_name:
                return group
        return None

    def _get_active_stage_group(self, stage: Optional[StageData]) -> Optional[GroupData]:
        self._ensure_stage_default_group(stage)
        if not stage:
            return None
        return self._get_stage_group(stage, stage.active_group_name) or self._get_stage_group(stage, DEFAULT_GROUP_NAME)

    def _set_active_stage_group(self, stage: Optional[StageData], group_name: str):
        self._ensure_stage_default_group(stage)
        if not stage:
            return False
        group = self._get_stage_group(stage, group_name)
        if not group:
            return False
        stage.active_group_name = group.name
        return True

    def _group_item_ref(self, scene: SceneData, item: ItemData) -> GroupItemRef:
        return GroupItemRef(scene.name, item.item_type, item.name)

    def _iter_groupable_item_refs(self, stage: Optional[StageData]) -> List[GroupItemRef]:
        if not stage:
            return []
        refs = []
        seen = set()
        for scene in stage.scenes:
            for item in scene.items:
                if item.item_type not in GROUPABLE_ITEM_TYPES:
                    continue
                ref = self._group_item_ref(scene, item)
                if ref in seen:
                    continue
                seen.add(ref)
                refs.append(ref)
        return refs

    def _is_item_visible_in_stage_group(
        self,
        stage: Optional[StageData],
        scene: Optional[SceneData],
        item: ItemData,
    ) -> bool:
        if not item or item.item_type not in GROUPABLE_ITEM_TYPES:
            return True
        if not stage or not scene:
            return True
        group = self._get_active_stage_group(stage)
        if not group or group.includes_all:
            return True
        return self._group_item_ref(scene, item) in set(group.items)

    def _serialize_stage_groups(self, stage: StageData) -> Dict[str, Dict[str, object]]:
        self._ensure_stage_default_group(stage)
        group_data = {}
        for group in stage.groups:
            if group.name == DEFAULT_GROUP_NAME or group.includes_all:
                group_data[group.name] = {"all": True}
                continue
            group_data[group.name] = {
                "items": [
                    {
                        "scene": ref.scene_name,
                        "type": ref.item_type,
                        "name": ref.item_name,
                    }
                    for ref in group.items
                ]
            }
        return group_data

    def _deserialize_stage_groups(self, groups_data, stage: StageData) -> List[GroupData]:
        if not isinstance(groups_data, dict) or not groups_data:
            return default_stage_groups()

        valid_refs = set(self._iter_groupable_item_refs(stage))
        has_valid_ref_catalog = bool(valid_refs)
        groups = [GroupData(name=DEFAULT_GROUP_NAME, includes_all=True)]
        seen_names = {DEFAULT_GROUP_NAME}

        for raw_name, raw_group in groups_data.items():
            name = str(raw_name).strip()
            if not name or name in seen_names:
                continue
            if name == DEFAULT_GROUP_NAME:
                continue
            if isinstance(raw_group, dict) and raw_group.get("all"):
                continue
            raw_items = raw_group.get("items", []) if isinstance(raw_group, dict) else []
            items = []
            for raw_ref in raw_items:
                if not isinstance(raw_ref, dict):
                    continue
                ref = GroupItemRef(
                    scene_name=str(raw_ref.get("scene", "")).strip(),
                    item_type=str(raw_ref.get("type", "")).strip(),
                    item_name=str(raw_ref.get("name", "")).strip(),
                )
                if ref.item_type not in GROUPABLE_ITEM_TYPES or not ref.scene_name or not ref.item_name:
                    continue
                if has_valid_ref_catalog and ref not in valid_refs:
                    continue
                if ref not in items:
                    items.append(ref)
            groups.append(GroupData(name=name, items=items))
            seen_names.add(name)
        return groups

    def _rename_group_item_refs(self, stage: Optional[StageData], old_ref: GroupItemRef, new_ref: GroupItemRef):
        if not stage:
            return
        for group in stage.groups:
            if group.includes_all:
                continue
            group.items = [new_ref if ref == old_ref else ref for ref in group.items]

    def _remove_group_item_refs(self, stage: Optional[StageData], predicate):
        if not stage:
            return
        for group in stage.groups:
            if group.includes_all:
                continue
            group.items = [ref for ref in group.items if not predicate(ref)]

    def _rename_group_scene_refs(self, stage: Optional[StageData], old_scene_name: str, new_scene_name: str):
        if not stage or old_scene_name == new_scene_name:
            return
        for group in stage.groups:
            if group.includes_all:
                continue
            updated_items = []
            for ref in group.items:
                if ref.scene_name == old_scene_name:
                    ref = GroupItemRef(new_scene_name, ref.item_type, ref.item_name)
                if ref not in updated_items:
                    updated_items.append(ref)
            group.items = updated_items

    def update_group_controls(self):
        if not hasattr(self, "group_combo"):
            return
        stage = self.current_stage or self.current_work_stage
        enabled = stage is not None
        self._updating_group_combo = True
        self.group_combo.clear()
        if stage:
            self._ensure_stage_default_group(stage)
            for group in stage.groups:
                self.group_combo.addItem(group.name)
            active_group = self._get_active_stage_group(stage)
            active_name = active_group.name if active_group else DEFAULT_GROUP_NAME
            index = self.group_combo.findText(active_name)
            if index >= 0:
                self.group_combo.setCurrentIndex(index)
        else:
            self.group_combo.addItem(DEFAULT_GROUP_NAME)
        self._updating_group_combo = False
        active_name = self.group_combo.currentText() or DEFAULT_GROUP_NAME
        is_default = active_name == DEFAULT_GROUP_NAME
        self.group_combo.setEnabled(enabled)
        self.btn_add_group.setEnabled(enabled)
        self.btn_edit_group.setEnabled(enabled and not is_default)
        self.btn_delete_group.setEnabled(enabled and not is_default)

    def on_group_combo_changed(self, group_name):
        if getattr(self, "_updating_group_combo", False):
            return
        stage = self.current_stage or self.current_work_stage
        if not stage or not self._set_active_stage_group(stage, group_name):
            self.update_group_controls()
            return
        self.update_group_controls()
        self.update_tree_view()
        if self.current_scene:
            self.select_data_in_tree(self.current_scene)
        elif stage:
            self.select_data_in_tree(stage)
        if self.current_scene:
            self.canvas.redraw_overlays(self.current_scene)
        self.status_label.setText(f"当前分组已切换为 {group_name}。")

    def _format_group_item_ref_label(self, ref: GroupItemRef) -> str:
        suffix = "Area" if ref.item_type == "area" else "Special"
        return f"{ref.scene_name}_{ref.item_name} ({suffix})"

    def _open_group_dialog(self, stage: StageData, existing_group: Optional[GroupData] = None):
        dialog = QDialog(self)
        dialog.setWindowTitle("修改分组" if existing_group else "添加分组")
        layout = QVBoxLayout(dialog)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("组名:"))
        name_edit = QLineEdit(existing_group.name if existing_group else "")
        name_row.addWidget(name_edit)
        layout.addLayout(name_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        checks_layout = QVBoxLayout(scroll_widget)
        selected_refs = set(existing_group.items if existing_group else [])
        checkboxes = []
        for ref in self._iter_groupable_item_refs(stage):
            checkbox = QCheckBox(self._format_group_item_ref_label(ref))
            checkbox.setChecked(ref in selected_refs)
            checks_layout.addWidget(checkbox)
            checkboxes.append((ref, checkbox))
        checks_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认")
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        group_name = name_edit.text().strip()
        items = [ref for ref, checkbox in checkboxes if checkbox.isChecked()]
        return group_name, items

    def add_group(self):
        stage = self.current_stage or self.current_work_stage
        if not stage:
            QMessageBox.warning(self, "提示", "请先选择一个阶段。")
            return
        result = self._open_group_dialog(stage)
        if result is None:
            return
        group_name, items = result
        if not group_name:
            QMessageBox.warning(self, "提示", "组名不能为空。")
            return
        self._ensure_stage_default_group(stage)
        if self._get_stage_group(stage, group_name):
            QMessageBox.warning(self, "名称重复", "分组名重复，请重新输入。")
            return
        if group_name == DEFAULT_GROUP_NAME:
            QMessageBox.warning(self, "提示", "默认分组已内置，不能重复创建。")
            return
        stage.groups.append(GroupData(name=group_name, items=items))
        stage.active_group_name = group_name
        self.update_group_controls()
        self.update_tree_view()
        if self.current_scene:
            self.canvas.redraw_overlays(self.current_scene)
        self.status_label.setText(f"已添加分组 {group_name}。")

    def edit_current_group(self):
        stage = self.current_stage or self.current_work_stage
        group = self._get_active_stage_group(stage)
        if not stage or not group:
            QMessageBox.warning(self, "提示", "请先选择一个分组。")
            return
        if group.includes_all or group.name == DEFAULT_GROUP_NAME:
            QMessageBox.information(self, "提示", "默认分组包含全部区域和特殊区域，不能修改。")
            return
        result = self._open_group_dialog(stage, group)
        if result is None:
            return
        group_name, items = result
        if not group_name:
            QMessageBox.warning(self, "提示", "组名不能为空。")
            return
        if group_name == DEFAULT_GROUP_NAME:
            QMessageBox.warning(self, "提示", "不能将自定义分组改名为默认。")
            return
        existing = self._get_stage_group(stage, group_name)
        if existing and existing is not group:
            QMessageBox.warning(self, "名称重复", "分组名重复，请重新输入。")
            return
        old_name = group.name
        group.name = group_name
        group.items = items
        stage.active_group_name = group_name
        self.update_group_controls()
        self.update_tree_view()
        if self.current_scene:
            self.canvas.redraw_overlays(self.current_scene)
        self.status_label.setText(f"已更新分组 {old_name}。")

    def delete_current_group(self):
        stage = self.current_stage or self.current_work_stage
        group = self._get_active_stage_group(stage)
        if not stage or not group:
            QMessageBox.warning(self, "提示", "请先选择一个分组。")
            return
        if group.includes_all or group.name == DEFAULT_GROUP_NAME:
            QMessageBox.information(self, "提示", "默认分组不能删除。")
            return
        reply = QMessageBox.question(self, "删除分组", f"确认删除分组 {group.name}？")
        if reply != QMessageBox.StandardButton.Yes:
            return
        stage.groups.remove(group)
        stage.active_group_name = DEFAULT_GROUP_NAME
        self.update_group_controls()
        self.update_tree_view()
        if self.current_scene:
            self.canvas.redraw_overlays(self.current_scene)
        self.status_label.setText(f"已删除分组 {group.name}。")

    def _scene_size_label(self, scene: SceneData) -> str:
        width, height = self._get_scene_image_size(scene)
        if width > 0 and height > 0:
            return f"{width} * {height}"
        return "未抓图/未导入图片"

    def _group_scenes_by_name(self, stage: StageData) -> Dict[str, List[SceneData]]:
        grouped = {}
        for scene in stage.scenes:
            grouped.setdefault(scene.name, []).append(scene)
        return grouped

    def _find_stage_for_scene(self, scene_data: SceneData) -> Optional[StageData]:
        if not self.project or not scene_data:
            return None
        for stage in self.project.stages:
            if scene_data in stage.scenes:
                return stage
        return None

    def _find_scene_peers(self, scene_data: SceneData) -> List[SceneData]:
        stage = self._find_stage_for_scene(scene_data)
        if not stage:
            return []
        return [scene for scene in stage.scenes if scene.name == scene_data.name and scene is not scene_data]

    def _find_scene_item(self, scene: SceneData, item_type: str, name: str) -> Optional[ItemData]:
        for item in scene.items:
            if item.item_type == item_type and item.name == name:
                return item
        return None
    def rename_project(self):
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "修改项目名", "新项目名:", text=self.project.name)
        if ok and name:
            self.project.name = name.strip()
            self.update_tree_view()
            self.status_label.setText(f"工程名称已更新为 {self.project.name}。")
    def update_tree_view(self):
        """
            刷新右侧项目树状列表控件 (QTreeWidget)。
            将内存中的数据同步渲染到 UI。
        """
        expanded_ids = self.collect_expanded_ids()
        self.tree.clear()
        if not self.project:
            return
        root = QTreeWidgetItem(self.tree)
        root_text = f"ROOT: {self.project.name}"
        root.setText(0, root_text)
        root.setToolTip(0, root_text)
        root.setData(0, Qt.ItemDataRole.UserRole, self.project)
        root.setExpanded(True)
        stage_items = {}
        scene_items = {}
        for stage in self.project.stages:
            s_node = QTreeWidgetItem(root)
            stage_text = f"阶段: {stage.name}"
            s_node.setText(0, stage_text)
            s_node.setToolTip(0, stage_text)
            s_node.setData(0, Qt.ItemDataRole.UserRole, stage)
            stage_items[stage.id] = s_node
            for scene_name, scenes in self._group_scenes_by_name(stage).items():
                group_node = QTreeWidgetItem(s_node)
                scene_text = f"场景: {scene_name}"
                group_node.setText(0, scene_text)
                group_node.setToolTip(0, scene_text)
                group_node.setData(0, Qt.ItemDataRole.UserRole, {
                    "kind": "scene_group",
                    "stage": stage,
                    "scene_name": scene_name,
                })
                for scene in scenes:
                    sc_node = QTreeWidgetItem(group_node)
                    scene_text = self._scene_size_label(scene)
                    sc_node.setText(0, scene_text)
                    sc_node.setToolTip(0, scene_text)
                    sc_node.setData(0, Qt.ItemDataRole.UserRole, scene)
                    scene_items[scene.id] = sc_node
                    for item in scene.items:
                        if not self._is_item_visible_in_stage_group(stage, scene, item):
                            continue
                        i_node = QTreeWidgetItem(sc_node)
                        if item.item_type == 'area':
                            type_icon = "[区域]"
                        elif item.item_type == 'special_area':
                            type_icon = "[特殊区域]"
                        else:
                            type_icon = "[控点]"
                        item_text = f"{type_icon} {item.name}"
                        i_node.setText(0, item_text)
                        i_node.setToolTip(0, item_text)
                        vis_text = "显示" if item.visible else "隐藏"
                        i_node.setText(1, vis_text)
                        i_node.setToolTip(1, vis_text)
                        i_node.setData(0, Qt.ItemDataRole.UserRole, item)
        self.tree.resizeColumnToContents(0)
        self.tree.resizeColumnToContents(1)
        self.restore_expanded_ids(expanded_ids, stage_items, scene_items)
        if self.last_expand_stage_id:
            item = stage_items.get(self.last_expand_stage_id)
            if item:
                item.setExpanded(True)
        if self.last_expand_scene_id:
            item = scene_items.get(self.last_expand_scene_id)
            if item:
                item.setExpanded(True)
        self.last_expand_stage_id = None
        self.last_expand_scene_id = None
        self.update_group_controls()
    def add_stage(self):
        if not self.project: return
        name = self.prompt_unique_name("新建阶段", "阶段名称:",
                                       existing_names=[s.name for s in self.project.stages])
        if name:
            new_stage = StageData(id=str(random.randint(1000, 9999)), name=name)
            self.project.stages.append(new_stage)
            self.current_stage = new_stage
            self.set_current_work_stage(new_stage)
            self.current_scene = None
            self.last_expand_stage_id = new_stage.id
            self.update_tree_view()
            self.select_data_in_tree(new_stage)
    def expand_all_tree(self):
        if not self.project:
            return
        self.tree.expandAll()
    def collapse_all_tree(self):
        if not self.project:
            return
        self.tree.collapseAll()
    # --- 树交互与动态按钮 ---
    def on_tree_click(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        self.clear_action_panel()
        if isinstance(data, StageData):
            self.current_stage = data
            self.set_current_work_stage(data)
            self.current_scene = None
            self.clear_scene_display()
            btn = QPushButton("添加场景")
            btn.clicked.connect(lambda: self.add_scene(data))
            self.action_layout.addWidget(btn)
            btn_paste_scene = QPushButton("📋 粘贴场景")
            btn_paste_scene.clicked.connect(lambda: self.paste_scene_to_stage(data))
            btn_paste_scene.setEnabled(self.scene_clipboard is not None)
            self.action_layout.addWidget(btn_paste_scene)
            if self.project and len(self.project.stages) > 1:
                btn_move_up = QPushButton("⬆ 上移阶段")
                btn_move_up.clicked.connect(lambda: self.move_stage(data, -1))
                btn_move_up.setEnabled(self.project.stages.index(data) > 0)
                self.action_layout.addWidget(btn_move_up)
                btn_move_down = QPushButton("⬇ 下移阶段")
                btn_move_down.clicked.connect(lambda: self.move_stage(data, 1))
                btn_move_down.setEnabled(self.project.stages.index(data) < len(self.project.stages) - 1)
                self.action_layout.addWidget(btn_move_down)
            btn_rename = QPushButton("✏️ 修改阶段名称")
            btn_rename.clicked.connect(lambda: self.rename_stage(data))
            self.action_layout.addWidget(btn_rename)
            btn_delete = QPushButton("🗑 删除阶段")
            btn_delete.clicked.connect(lambda: self.delete_stage(data))
            self.action_layout.addWidget(btn_delete)
        elif isinstance(data, dict) and data.get("kind") == "scene_group":
            self.current_stage = data.get("stage")
            self.set_current_work_stage(self.current_stage)
            self.current_scene = None
            self.clear_scene_display()
            scene_name = data.get("scene_name", "")
            lbl = QLabel(f"当前场景: {scene_name}")
            self.action_layout.addWidget(lbl)
            btn_add_res = QPushButton("复制控件到不同分辨率")
            btn_add_res.clicked.connect(lambda: self.copy_scene_to_different_resolution(self.current_stage, scene_name))
            self.action_layout.addWidget(btn_add_res)
            btn_rename = QPushButton("✏️ 修改场景名称")
            btn_rename.clicked.connect(lambda: self.rename_scene_group(self.current_stage, scene_name))
            self.action_layout.addWidget(btn_rename)
            btn_delete = QPushButton("🗑 删除整个场景")
            btn_delete.clicked.connect(lambda: self.delete_scene_group(self.current_stage, scene_name))
            self.action_layout.addWidget(btn_delete)
        elif isinstance(data, SceneData):
            self.current_stage = self._find_stage_for_scene(data)
            self.set_current_work_stage(self.current_stage)
            self.current_scene = data
            self.show_scene_image(data)
            # 场景操作按钮
            btn_cap = QPushButton("📷 抓图")
            btn_cap.clicked.connect(lambda: self.capture_image(data))
            btn_import = QPushButton("🖼 导入图片")
            btn_import.clicked.connect(lambda: self.import_image(data))
            btn_area = QPushButton("🟦 添加区域 (Area)")
            btn_area.clicked.connect(lambda: self.prepare_draw('area'))
            btn_ctrl = QPushButton("🟥 添加控点 (Control)")
            btn_ctrl.clicked.connect(lambda: self.prepare_draw('control'))
            btn_sp_area = QPushButton("🟧 添加特殊区域 (Special)")
            btn_sp_area.clicked.connect(lambda: self.prepare_draw('special_area'))
            self.action_layout.addWidget(btn_cap)
            self.action_layout.addWidget(btn_import)
            self.action_layout.addWidget(btn_area)
            self.action_layout.addWidget(btn_ctrl)
            self.action_layout.addWidget(btn_sp_area)
            btn_copy_resolution = QPushButton("复制控件到不同分辨率")
            btn_copy_resolution.clicked.connect(lambda: self.copy_scene_to_different_resolution(self.current_stage, data.name, data))
            self.action_layout.addWidget(btn_copy_resolution)
            btn_copy_scene = QPushButton("📋 复制场景")
            btn_copy_scene.clicked.connect(lambda: self.copy_scene(data))
            self.action_layout.addWidget(btn_copy_scene)
            btn_rename = QPushButton("✏️ 修改场景名称")
            btn_rename.clicked.connect(lambda: self.rename_scene(data))
            self.action_layout.addWidget(btn_rename)
            btn_delete = QPushButton("🗑 删除场景")
            btn_delete.clicked.connect(lambda: self.delete_scene(data))
            self.action_layout.addWidget(btn_delete)
        elif isinstance(data, ItemData):
            scene_node = item.parent()
            self.current_scene = scene_node.data(0, Qt.ItemDataRole.UserRole) if scene_node else None
            self.current_stage = self._find_stage_for_scene(self.current_scene)
            self.set_current_work_stage(self.current_stage)
            # 同一场景下只重绘标注，避免点击时触发自动适配
            if self.canvas.active_scene_data is self.current_scene and self.canvas.current_pixmap:
                self.canvas.redraw_overlays(self.current_scene)
            else:
                self.show_scene_image(self.current_scene)
            # 区域/控点操作
            lbl = QLabel(f"当前选中: {data.name}")
            self.action_layout.addWidget(lbl)
            if data.item_type == 'area':
                lbl_mode = QLabel(f"匹配模式: {data.match_mode or 'gray'}")
                self.action_layout.addWidget(lbl_mode)
                btn_match_mode = QPushButton("🎨 设置匹配模式")
                btn_match_mode.clicked.connect(lambda: self.set_area_match_mode(data))
                self.action_layout.addWidget(btn_match_mode)
                btn_scope = QPushButton("🔍 设置搜索范围 (Search Scope)")
                btn_scope.clicked.connect(lambda: self.prepare_draw('search_scope', data))
                self.action_layout.addWidget(btn_scope)
                btn_edit_area = QPushButton("✏️ 修改区域位置")
                btn_edit_area.clicked.connect(lambda: self.prepare_draw('edit_area', data))
                self.action_layout.addWidget(btn_edit_area)
                btn_rename = QPushButton("✏️ 修改区域名称")
                btn_rename.clicked.connect(lambda: self.rename_item(data))
                self.action_layout.addWidget(btn_rename)
            elif data.item_type == 'special_area':
                btn_set_range = QPushButton("📐 设置区域范围 (Normalized)")
                btn_set_range.clicked.connect(lambda: self.set_special_area_range(data))
                self.action_layout.addWidget(btn_set_range)
                btn_edit_area = QPushButton("✏️ 修改区域位置")
                btn_edit_area.clicked.connect(lambda: self.prepare_draw('edit_special_area', data))
                self.action_layout.addWidget(btn_edit_area)
                btn_copy_pos = QPushButton("📋 复制区域坐标")
                btn_copy_pos.clicked.connect(lambda: self.copy_special_area_coords(data))
                self.action_layout.addWidget(btn_copy_pos)
                btn_rename = QPushButton("✏️ 修改区域名称")
                btn_rename.clicked.connect(lambda: self.rename_item(data))
                self.action_layout.addWidget(btn_rename)
            else:
                btn_edit_ctrl = QPushButton("✏️ 修改控点位置")
                btn_edit_ctrl.clicked.connect(lambda: self.prepare_draw('edit_control', data))
                self.action_layout.addWidget(btn_edit_ctrl)
                btn_rename = QPushButton("✏️ 修改控点名称")
                btn_rename.clicked.connect(lambda: self.rename_item(data))
                self.action_layout.addWidget(btn_rename)
            btn_vis = QPushButton("切换显示/隐藏")
            btn_vis.clicked.connect(lambda: self.toggle_visibility(data))
            self.action_layout.addWidget(btn_vis)
            btn_delete = QPushButton("🗑 删除")
            btn_delete.clicked.connect(lambda: self.delete_selected_item(data))
            self.action_layout.addWidget(btn_delete)
    def clear_action_panel(self):
        for i in reversed(range(self.action_layout.count())):
            self.action_layout.itemAt(i).widget().setParent(None)
    def clear_scene_display(self):
        self.canvas.scene.clear()
        self.canvas.current_pixmap = None
        self.canvas.active_scene_data = None
        self.canvas.crosshair_h = None
        self.canvas.crosshair_v = None
        self.canvas.hide_crosshair()
        self.update_coord_display(None, None)
    def update_coord_display(self, x, y):
        if x is None or y is None:
            self.coord_label.setText("坐标: (-,-)")
            return
        self.coord_label.setText(f"坐标: ({int(x)}, {int(y)})")
    def select_item_in_tree(self, target_data):
        self.select_data_in_tree(target_data)
    def select_data_in_tree(self, target_data):
        if not target_data:
            return
        def walk(node):
            for i in range(node.childCount()):
                child = node.child(i)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                if data is target_data:
                    return child
                found = walk(child)
                if found:
                    return found
            return None
        root = self.tree.invisibleRootItem()
        target_item = walk(root)
        if target_item:
            parent = target_item.parent()
            while parent:
                parent.setExpanded(True)
                parent = parent.parent()
            self.tree.setCurrentItem(target_item)
            self.on_tree_click(target_item, 0)
    def open_context_menu(self, position):
        item = self.tree.itemAt(position)
        if not item: return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        menu = QMenu()
        if isinstance(data, StageData):
            paste_action = QAction("📋 粘贴场景", self)
            paste_action.setEnabled(self.scene_clipboard is not None)
            paste_action.triggered.connect(lambda: self.paste_scene_to_stage(data))
            menu.addAction(paste_action)
            menu.addSeparator()
        elif isinstance(data, dict) and data.get("kind") == "scene_group":
            copy_resolution_action = QAction("复制控件到不同分辨率", self)
            copy_resolution_action.triggered.connect(
                lambda: self.copy_scene_to_different_resolution(data.get("stage"), data.get("scene_name", ""))
            )
            menu.addAction(copy_resolution_action)
            rename_group_action = QAction("✏️ 修改场景名称", self)
            rename_group_action.triggered.connect(
                lambda: self.rename_scene_group(data.get("stage"), data.get("scene_name", ""))
            )
            menu.addAction(rename_group_action)
            menu.addSeparator()
        elif isinstance(data, SceneData):
            copy_action = QAction("📋 复制场景", self)
            copy_action.triggered.connect(lambda: self.copy_scene(data))
            menu.addAction(copy_action)
            copy_resolution_action = QAction("复制控件到不同分辨率", self)
            copy_resolution_action.triggered.connect(
                lambda: self.copy_scene_to_different_resolution(self._find_stage_for_scene(data), data.name, data)
            )
            menu.addAction(copy_resolution_action)
            menu.addSeparator()
        del_action = QAction("删除", self)
        del_action.triggered.connect(lambda: self.delete_item(item, data))
        menu.addAction(del_action)
        menu.exec(self.tree.viewport().mapToGlobal(position))
    def show_item_context_menu(self, item_data: ItemData, global_pos):
        if not isinstance(item_data, ItemData):
            return
        menu = QMenu(self)
        if item_data.item_type == 'area':
            action_match_mode = QAction("🎨 设置匹配模式", self)
            action_match_mode.triggered.connect(lambda: self.set_area_match_mode(item_data))
            menu.addAction(action_match_mode)
            action_scope = QAction("🔍 设置搜索范围", self)
            action_scope.triggered.connect(lambda: self.prepare_draw('search_scope', item_data))
            menu.addAction(action_scope)
            action_edit = QAction("✏️ 修改区域位置", self)
            action_edit.triggered.connect(lambda: self.prepare_draw('edit_area', item_data))
            menu.addAction(action_edit)
            action_rename = QAction("✏️ 修改区域名称", self)
            action_rename.triggered.connect(lambda: self.rename_item(item_data))
            menu.addAction(action_rename)
        elif item_data.item_type == 'special_area':
            action_set_range = QAction("📐 设置区域范围 (Normalized)", self)
            action_set_range.triggered.connect(lambda: self.set_special_area_range(item_data))
            menu.addAction(action_set_range)
            action_edit = QAction("✏️ 修改区域位置", self)
            action_edit.triggered.connect(lambda: self.prepare_draw('edit_special_area', item_data))
            menu.addAction(action_edit)
            action_copy = QAction("📋 复制区域坐标", self)
            action_copy.triggered.connect(lambda: self.copy_special_area_coords(item_data))
            menu.addAction(action_copy)
            action_rename = QAction("✏️ 修改区域名称", self)
            action_rename.triggered.connect(lambda: self.rename_item(item_data))
            menu.addAction(action_rename)
        elif item_data.item_type == 'control':
            action_edit = QAction("✏️ 修改控点位置", self)
            action_edit.triggered.connect(lambda: self.prepare_draw('edit_control', item_data))
            menu.addAction(action_edit)
            action_rename = QAction("✏️ 修改控点名称", self)
            action_rename.triggered.connect(lambda: self.rename_item(item_data))
            menu.addAction(action_rename)
        action_vis = QAction("切换显示/隐藏", self)
        action_vis.triggered.connect(lambda: self.toggle_visibility(item_data))
        menu.addAction(action_vis)
        action_delete = QAction("🗑 删除", self)
        action_delete.triggered.connect(lambda: self.delete_selected_item(item_data))
        menu.addAction(action_delete)
        menu.exec(global_pos)
    def show_canvas_context_menu(self, global_pos):
        menu = QMenu(self)
        action_add_stage = QAction("新建阶段", self)
        action_add_stage.triggered.connect(self.add_stage)
        menu.addAction(action_add_stage)
        stage_name = self.current_work_stage.name if self.current_work_stage else "未选择阶段"
        action_add_scene = QAction(f"新建场景 [{stage_name}]", self)
        action_add_scene.setEnabled(self.current_work_stage is not None)
        action_add_scene.triggered.connect(self.add_scene_for_current_stage)
        menu.addAction(action_add_scene)
        action_paste_scene = QAction(f"粘贴场景 [{stage_name}]", self)
        action_paste_scene.setEnabled(self.current_work_stage is not None and self.scene_clipboard is not None)
        action_paste_scene.triggered.connect(lambda: self.paste_scene_to_stage(self.current_work_stage))
        menu.addAction(action_paste_scene)
        menu.exec(global_pos)
    def delete_item(self, tree_item, data):
        parent = tree_item.parent()
        if not parent: return  # Can't delete root
        parent_data = parent.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, StageData):
            self.project.stages.remove(data)
            if self.current_stage == data:
                self.current_stage = None
            if self.current_work_stage == data:
                self.set_current_work_stage(self.project.stages[0] if self.project.stages else None)
        elif isinstance(data, dict) and data.get("kind") == "scene_group":
            self.delete_scene_group(data.get("stage"), data.get("scene_name", ""))
            return
        elif isinstance(data, SceneData):
            stage = self._find_stage_for_scene(data)
            if stage:
                scene_name = data.name
                stage.scenes.remove(data)
                if not any(scene.name == scene_name for scene in stage.scenes):
                    self._remove_group_item_refs(stage, lambda ref: ref.scene_name == scene_name)
            if self.current_scene == data:
                self.current_scene = None
        elif isinstance(data, ItemData):
            parent_data.items.remove(data)
            self.sync_deleted_item_to_scene_peers(parent_data, data)
            self.show_scene_image(parent_data)  # refresh view
        self.update_tree_view()
    # --- 场景与绘图逻辑 ---
    def add_scene(self, stage):
        self.current_stage = stage
        self.set_current_work_stage(stage)
        name = self.prompt_unique_name("新建场景", "场景名称:",
                                       existing_names=[s.name for s in stage.scenes])
        if name:
            new_scene = SceneData(id=str(random.randint(1000, 9999)), name=name)
            stage.scenes.append(new_scene)
            self.current_scene = new_scene
            self.last_expand_stage_id = stage.id
            self.last_expand_scene_id = new_scene.id
            self.update_tree_view()
            self.select_data_in_tree(new_scene)
    def add_scene_for_current_stage(self):
        if not self.current_work_stage:
            QMessageBox.warning(self, "提示", "请先选择一个当前工作阶段。")
            return
        self.add_scene(self.current_work_stage)
    def _clone_rect(self, rect: Optional[RectData]) -> Optional[RectData]:
        if rect is None:
            return None
        return RectData(rect.x, rect.y, rect.w, rect.h)

    def _get_scene_image_size(self, scene: SceneData):
        if not scene:
            return 0, 0
        if scene.image_width > 0 and scene.image_height > 0:
            return scene.image_width, scene.image_height
        if scene.pixmap and not scene.pixmap.isNull():
            return scene.pixmap.width(), scene.pixmap.height()
        if scene.image_path and os.path.exists(scene.image_path):
            pixmap = QPixmap(scene.image_path)
            if not pixmap.isNull():
                return pixmap.width(), pixmap.height()
        return 0, 0

    def _scale_rect_between_images(self, rect: Optional[RectData], old_w, old_h, new_w, new_h) -> Optional[RectData]:
        if rect is None:
            return None
        if old_w <= 0 or old_h <= 0 or new_w <= 0 or new_h <= 0:
            return self._clone_rect(rect)
        rect_norm = [
            rect.x / old_w,
            rect.y / old_h,
            (rect.x + rect.w) / old_w,
            (rect.y + rect.h) / old_h,
        ]
        x1, y1, x2, y2 = resolve_area_rect_for_frame(
            new_w,
            new_h,
            {"rect": rect_norm},
            new_w,
            new_h,
            old_w,
            old_h,
        )
        return RectData(
            x1,
            y1,
            x2 - x1,
            y2 - y1,
        )

    def _rescale_scene_items_for_new_image(self, scene: SceneData, old_size, new_size):
        old_w, old_h = old_size
        new_w, new_h = new_size
        if old_w <= 0 or old_h <= 0 or new_w <= 0 or new_h <= 0:
            return False
        if old_w == new_w and old_h == new_h:
            return False
        if not scene.items:
            return False
        for item in scene.items:
            item.rect = self._scale_rect_between_images(item.rect, old_w, old_h, new_w, new_h)
            if item.search_scope:
                item.search_scope = self._scale_rect_between_images(item.search_scope, old_w, old_h, new_w, new_h)
        return True

    def _replace_scene_image(self, scene_data: SceneData, image_path: str, pixmap: QPixmap):
        old_size = self._get_scene_image_size(scene_data)
        new_size = (pixmap.width(), pixmap.height())
        resized_items = self._rescale_scene_items_for_new_image(scene_data, old_size, new_size)
        scene_data.image_path = image_path
        scene_data.pixmap = pixmap
        scene_data.image_width, scene_data.image_height = new_size
        return resized_items

    def _refresh_tree_for_scene(self, scene_data: SceneData):
        stage = self._find_stage_for_scene(scene_data)
        if stage:
            self.last_expand_stage_id = stage.id
        self.last_expand_scene_id = scene_data.id
        self.update_tree_view()
        self.select_data_in_tree(scene_data)

    def _find_scene_with_image_size(self, scenes: List[SceneData], image_size: Tuple[int, int]) -> Optional[SceneData]:
        for scene in scenes:
            if self._get_scene_image_size(scene) == image_size:
                return scene
        return None

    def _select_resolution_template_scene(
        self,
        scenes: List[SceneData],
        preferred_scene: SceneData,
        target_size: Tuple[int, int],
    ) -> Optional[SceneData]:
        preferred_size = self._get_scene_image_size(preferred_scene)
        if preferred_size[0] > 0 and preferred_size[1] > 0 and preferred_size != target_size:
            return preferred_scene

        candidates = []
        for scene in scenes:
            scene_size = self._get_scene_image_size(scene)
            if scene_size[0] <= 0 or scene_size[1] <= 0 or scene_size == target_size:
                continue
            candidates.append(scene)
        if not candidates:
            return None
        return max(candidates, key=lambda scene: len(scene.items))

    def _apply_capture_pixmap_to_scene_resolution(
        self,
        scene_data: SceneData,
        image_path: str,
        pixmap: QPixmap,
    ) -> CaptureResolutionApplyResult:
        stage = self._find_stage_for_scene(scene_data)
        new_size = (pixmap.width(), pixmap.height())
        if not stage or new_size[0] <= 0 or new_size[1] <= 0:
            resized_items = self._replace_scene_image(scene_data, image_path, pixmap)
            return CaptureResolutionApplyResult(scene=scene_data, action="replaced_current", resized_items=resized_items)

        same_name_scenes = [scene for scene in stage.scenes if scene.name == scene_data.name]
        existing_scene = self._find_scene_with_image_size(same_name_scenes, new_size)
        if existing_scene:
            resized_items = self._replace_scene_image(existing_scene, image_path, pixmap)
            return CaptureResolutionApplyResult(
                scene=existing_scene,
                action="replaced_existing" if existing_scene is not scene_data else "replaced_current",
                resized_items=resized_items,
            )

        source_scene = self._select_resolution_template_scene(same_name_scenes, scene_data, new_size)
        if source_scene is None:
            resized_items = self._replace_scene_image(scene_data, image_path, pixmap)
            return CaptureResolutionApplyResult(scene=scene_data, action="replaced_current", resized_items=resized_items)

        source_size = self._get_scene_image_size(source_scene)
        new_scene = SceneData(
            id=str(random.randint(1000, 9999)),
            name=scene_data.name,
            image_path=image_path,
            pixmap=pixmap,
            image_width=new_size[0],
            image_height=new_size[1],
            items=[
                self._clone_item_for_scene_size(item, source_size, new_size)
                for item in source_scene.items
            ],
        )
        stage.scenes.append(new_scene)
        return CaptureResolutionApplyResult(
            scene=new_scene,
            action="created",
            resized_items=bool(new_scene.items),
            source_scene=source_scene,
        )

    def _clone_item(self, item: ItemData) -> ItemData:
        return ItemData(
            id=str(random.randint(10000, 99999)),
            name=item.name,
            item_type=item.item_type,
            rect=self._clone_rect(item.rect),
            search_scope=self._clone_rect(item.search_scope),
            visible=item.visible,
            match_mode=item.match_mode or "gray",
        )

    def _clone_item_for_scene_size(self, item: ItemData, source_size: Tuple[int, int], target_size: Tuple[int, int]) -> ItemData:
        old_w, old_h = source_size
        new_w, new_h = target_size
        return ItemData(
            id=str(random.randint(10000, 99999)),
            name=item.name,
            item_type=item.item_type,
            rect=self._scale_rect_between_images(item.rect, old_w, old_h, new_w, new_h),
            search_scope=self._scale_rect_between_images(item.search_scope, old_w, old_h, new_w, new_h),
            visible=item.visible,
            match_mode=item.match_mode or "gray",
        )

    def sync_added_item_to_scene_peers(self, source_scene: SceneData, item_data: ItemData):
        source_size = self._get_scene_image_size(source_scene)
        for peer in self._find_scene_peers(source_scene):
            if self._find_scene_item(peer, item_data.item_type, item_data.name):
                continue
            peer_size = self._get_scene_image_size(peer)
            peer.items.append(self._clone_item_for_scene_size(item_data, source_size, peer_size))

    def sync_deleted_item_to_scene_peers(self, source_scene: SceneData, item_data: ItemData):
        for peer in self._find_scene_peers(source_scene):
            target = self._find_scene_item(peer, item_data.item_type, item_data.name)
            if target:
                peer.items.remove(target)
        stage = self._find_stage_for_scene(source_scene)
        self._remove_group_item_refs(
            stage,
            lambda ref: (
                ref.scene_name == source_scene.name
                and ref.item_type == item_data.item_type
                and ref.item_name == item_data.name
            ),
        )

    def sync_renamed_item_to_scene_peers(self, source_scene: SceneData, item_data: ItemData, old_name: str):
        for peer in self._find_scene_peers(source_scene):
            target = self._find_scene_item(peer, item_data.item_type, old_name)
            if target:
                target.name = item_data.name
                target.match_mode = item_data.match_mode or target.match_mode
        stage = self._find_stage_for_scene(source_scene)
        self._rename_group_item_refs(
            stage,
            GroupItemRef(source_scene.name, item_data.item_type, old_name),
            GroupItemRef(source_scene.name, item_data.item_type, item_data.name),
        )
    def _clone_scene(self, scene: SceneData, new_name: Optional[str] = None) -> SceneData:
        pixmap = None
        if scene.pixmap and not scene.pixmap.isNull():
            pixmap = scene.pixmap.copy()
        elif scene.image_path and os.path.exists(scene.image_path):
            loaded_pixmap = QPixmap(scene.image_path)
            if not loaded_pixmap.isNull():
                pixmap = loaded_pixmap.copy()
        image_width, image_height = self._get_scene_image_size(scene)
        return SceneData(
            id=str(random.randint(1000, 9999)),
            name=new_name or scene.name,
            image_path=scene.image_path,
            pixmap=pixmap,
            image_width=image_width,
            image_height=image_height,
            items=[self._clone_item(item) for item in scene.items],
        )
    def copy_scene(self, scene_data: SceneData):
        self.scene_clipboard = self._clone_scene(scene_data)
        self.status_label.setText(f"已复制场景 {scene_data.name}。请选择目标阶段后执行粘贴。")
    def paste_scene_to_stage(self, stage: Optional[StageData]):
        if stage is None:
            QMessageBox.warning(self, "提示", "请先选择一个目标阶段。")
            return
        if self.scene_clipboard is None:
            QMessageBox.warning(self, "提示", "当前没有可粘贴的场景。")
            return
        pasted_scene = self._clone_scene(self.scene_clipboard)
        stage.scenes.append(pasted_scene)
        self.current_stage = stage
        self.set_current_work_stage(stage)
        self.current_scene = pasted_scene
        self.last_expand_stage_id = stage.id
        self.last_expand_scene_id = pasted_scene.id
        self.update_tree_view()
        self.select_data_in_tree(pasted_scene)
        self.status_label.setText(f"已将场景 {self.scene_clipboard.name} 粘贴到阶段 {stage.name}。")

    def _load_new_resolution_pixmap(self, source_scene: SceneData):
        msg = QMessageBox(self)
        msg.setWindowTitle("复制到不同分辨率")
        msg.setText("请选择新分辨率图片来源。")
        capture_btn = msg.addButton("抓图", QMessageBox.ButtonRole.ActionRole)
        import_btn = msg.addButton("导入图片", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == cancel_btn:
            return None, None
        if clicked == capture_btn:
            remote_path = "/data/local/tmp/screenCasting.jpeg"
            local_dir = tempfile.gettempdir()
            local_path = os.path.join(local_dir, f"screenCasting_{source_scene.id}_{random.randint(1000, 9999)}.jpeg")
            try:
                subprocess.run(["hdc", "shell", "snapshot_display", "-f", remote_path],
                               check=True, capture_output=True, text=True,
                               **hidden_subprocess_kwargs())
                subprocess.run(["hdc", "file", "recv", remote_path, local_path],
                               check=True, capture_output=True, text=True,
                               **hidden_subprocess_kwargs())
            except subprocess.CalledProcessError as exc:
                QMessageBox.critical(self, "抓图失败", f"HDc 命令执行失败：\n{exc.stderr or exc.stdout}")
                return None, None
            pixmap = QPixmap(local_path)
            if pixmap.isNull():
                QMessageBox.critical(self, "抓图失败", "读取截图文件失败，请检查路径或权限。")
                return None, None
            return local_path, pixmap
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择不同分辨率图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return None, None
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.critical(self, "导入失败", "读取图片失败，请检查文件格式或路径。")
            return None, None
        return file_path, pixmap

    def copy_scene_to_different_resolution(
        self,
        stage: Optional[StageData],
        scene_name: str,
        source_scene: Optional[SceneData] = None,
    ):
        if not stage:
            QMessageBox.warning(self, "提示", "请先选择一个阶段。")
            return
        candidates = [scene for scene in stage.scenes if scene.name == scene_name]
        if not candidates:
            QMessageBox.warning(self, "提示", "未找到可复制的源场景。")
            return
        source_scene = source_scene or candidates[0]
        source_size = self._get_scene_image_size(source_scene)
        if source_size[0] <= 0 or source_size[1] <= 0:
            QMessageBox.warning(self, "提示", "源场景还没有有效图片，无法进行分辨率转换。")
            return
        image_path, pixmap = self._load_new_resolution_pixmap(source_scene)
        if not image_path or not pixmap or pixmap.isNull():
            return
        new_size = (pixmap.width(), pixmap.height())
        if new_size == source_size:
            QMessageBox.warning(self, "分辨率相同", "复制目标必须是不同分辨率的图片。")
            return
        for scene in candidates:
            if self._get_scene_image_size(scene) == new_size:
                QMessageBox.warning(self, "分辨率已存在", f"场景 {scene_name} 已经存在 {new_size[0]} * {new_size[1]}。")
                return
        new_scene = SceneData(
            id=str(random.randint(1000, 9999)),
            name=scene_name,
            image_path=image_path,
            pixmap=pixmap,
            image_width=new_size[0],
            image_height=new_size[1],
            items=[
                self._clone_item_for_scene_size(item, source_size, new_size)
                for item in source_scene.items
            ],
        )
        stage.scenes.append(new_scene)
        self.current_stage = stage
        self.set_current_work_stage(stage)
        self.current_scene = new_scene
        self.last_expand_stage_id = stage.id
        self.last_expand_scene_id = new_scene.id
        self.update_tree_view()
        self.select_data_in_tree(new_scene)
        self.status_label.setText(
            f"已将场景 {scene_name} 的组件转换到 {new_size[0]} * {new_size[1]}，可在新图片上微调。"
        )
    def capture_image(self, scene_data):
        # 通过 hdc 截图并拉取到本地，再显示到工作区
        remote_path = "/data/local/tmp/screenCasting.jpeg"
        local_dir = tempfile.gettempdir()
        local_path = os.path.join(local_dir, f"screenCasting_{scene_data.id}.jpeg")
        try:
            subprocess.run(["hdc", "shell", "snapshot_display", "-f", remote_path],
                           check=True, capture_output=True, text=True,
                           **hidden_subprocess_kwargs())
            subprocess.run(["hdc", "file", "recv", remote_path, local_path],
                           check=True, capture_output=True, text=True,
                           **hidden_subprocess_kwargs())
        except subprocess.CalledProcessError as exc:
            QMessageBox.critical(self, "抓图失败", f"HDc 命令执行失败：\n{exc.stderr or exc.stdout}")
            return
        img = QPixmap(local_path)
        if img.isNull():
            QMessageBox.critical(self, "抓图失败", "读取截图文件失败，请检查路径或权限。")
            return
        apply_result = self._apply_capture_pixmap_to_scene_resolution(scene_data, local_path, img)
        target_scene = apply_result.scene
        self.current_stage = self._find_stage_for_scene(target_scene)
        self.set_current_work_stage(self.current_stage)
        self.current_scene = target_scene
        self.canvas.set_image(img)
        self.canvas.redraw_overlays(target_scene)
        self._refresh_tree_for_scene(target_scene)
        if apply_result.action == "created":
            self.status_label.setText(
                f"抓图成功，检测到新分辨率 {target_scene.image_width} * {target_scene.image_height}，"
                "已自动复制并转换控件。"
            )
            return
        if apply_result.action == "replaced_existing" and target_scene is not scene_data:
            self.status_label.setText(
                f"抓图成功，已覆盖已有分辨率 {target_scene.image_width} * {target_scene.image_height} 的图片。"
            )
            return
        if apply_result.resized_items:
            self.status_label.setText("抓图成功，已按新图片尺寸同步缩放已有标注。")
            return
        self.status_label.setText("抓图成功。请开始添加区域或控点。")
    def import_image(self, scene_data):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        img = QPixmap(file_path)
        if img.isNull():
            QMessageBox.critical(self, "导入失败", "读取图片失败，请检查文件格式或路径。")
            return
        resized_items = self._replace_scene_image(scene_data, file_path, img)
        self.show_scene_image(scene_data)
        self._refresh_tree_for_scene(scene_data)
        if resized_items:
            self.status_label.setText("图片已导入，已按新图片尺寸同步缩放已有标注。")
            return
        self.status_label.setText("图片已导入。请开始添加区域或控点。")
    def show_scene_image(self, scene_data):
        self.canvas.active_scene_data = scene_data
        if scene_data.pixmap:
            if scene_data.image_width <= 0 or scene_data.image_height <= 0:
                scene_data.image_width = scene_data.pixmap.width()
                scene_data.image_height = scene_data.pixmap.height()
            self.canvas.set_image(scene_data.pixmap)
        elif not scene_data.image_path:
            self.clear_scene_display()
            self.canvas.active_scene_data = scene_data
            return
        self.canvas.redraw_overlays(scene_data)
    def prepare_draw(self, mode, target_data=None):
        if not self.canvas.current_pixmap:
            QMessageBox.warning(self, "错误", "请先抓图再进行标注。")
            return
        self.canvas.mode = mode
        self.canvas.target_item_data = target_data
        msg = {
            'area': "请在图片上框选感兴趣区域...",
            'control': "请在图片上框选控点位置...",
            'special_area': "请在图片上框选特殊区域...",
            'search_scope': "请框选新的搜索范围...",
            'edit_area': "请重新框选区域位置...",
            'edit_special_area': "请重新框选区域位置...",
            'edit_control': "请重新框选控点位置..."
        }
        self.status_label.setText(msg.get(mode, ""))
        self.setCursor(Qt.CursorShape.CrossCursor)
    def finish_drawing(self, rect: QRectF):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        mode = self.canvas.mode
        self.canvas.mode = "IDLE"
        rect_data = RectData(rect.x(), rect.y(), rect.width(), rect.height())
        if mode == 'search_scope' and self.canvas.target_item_data:
            # 更新现有Area的Scope
            self.canvas.target_item_data.search_scope = rect_data
            self.status_label.setText(f"已更新 {self.canvas.target_item_data.name} 的搜索范围。")
        elif mode == 'edit_area' and self.canvas.target_item_data:
            self.canvas.target_item_data.rect = rect_data
            self.canvas.target_item_data.search_scope = rect_data
            self.status_label.setText(f"已更新 {self.canvas.target_item_data.name} 的区域位置。")
        elif mode == 'edit_special_area' and self.canvas.target_item_data:
            self.canvas.target_item_data.rect = rect_data
            self.status_label.setText(f"已更新 {self.canvas.target_item_data.name} 的区域位置。")
        elif mode == 'edit_control' and self.canvas.target_item_data:
            self.canvas.target_item_data.rect = rect_data
            self.status_label.setText(f"已更新 {self.canvas.target_item_data.name} 的控点位置。")
        elif mode in ['area', 'control', 'special_area']:
            name_prefix = "区域" if mode == 'area' else "控点"
            if mode == 'special_area':
                name_prefix = "特殊区域"
            existing_names = self.get_stage_item_names(self.current_stage, mode)
            name = self.prompt_unique_name("命名", f"{name_prefix}名称:", existing_names=existing_names)
            if not name:
                self.status_label.setText("已取消添加。")
                self.canvas.redraw_overlays(self.current_scene)
                return
            new_item = ItemData(
                id=str(random.randint(10000, 99999)),
                name=name,
                item_type=mode,
                rect=rect_data
            )
            # 默认搜索范围 = 自身位置 (如果用户不改)
            if mode == 'area':
                new_item.search_scope = rect_data
            self.current_scene.items.append(new_item)
            self.sync_added_item_to_scene_peers(self.current_scene, new_item)
            active_group = self._get_active_stage_group(self.current_stage)
            if mode in GROUPABLE_ITEM_TYPES and active_group and not active_group.includes_all:
                ref = self._group_item_ref(self.current_scene, new_item)
                if ref not in active_group.items:
                    active_group.items.append(ref)
            if self.current_stage:
                self.last_expand_stage_id = self.current_stage.id
            self.last_expand_scene_id = self.current_scene.id
            self.status_label.setText(f"已添加 {name}")
            self.update_tree_view()
        # 重绘
        self.canvas.redraw_overlays(self.current_scene)
    def rename_item(self, item_data: ItemData):
        if not self.current_scene:
            return
        if item_data.item_type == "special_area":
            existing_names = self.get_stage_item_names(self.current_stage, "special_area")
        elif item_data.item_type == "control":
            existing_names = self.get_stage_item_names(self.current_stage, "control")
        else:
            existing_names = self.get_stage_item_names(self.current_stage, "area")
        existing_names = [n for n in existing_names if n != item_data.name]
        name = self.prompt_unique_name("修改名称", "新名称:", text=item_data.name,
                                       existing_names=existing_names)
        if name:
            old_name = item_data.name
            item_data.name = name
            self.sync_renamed_item_to_scene_peers(self.current_scene, item_data, old_name)
            self.status_label.setText(f"已更新名称为 {name}")
            self.update_tree_view()
            self.canvas.redraw_overlays(self.current_scene)
    def delete_selected_item(self, item_data: ItemData):
        if not self.current_scene:
            return
        self.current_scene.items.remove(item_data)
        self.sync_deleted_item_to_scene_peers(self.current_scene, item_data)
        self.status_label.setText(f"已删除 {item_data.name}")
        self.update_tree_view()
        self.canvas.redraw_overlays(self.current_scene)
    def rename_stage(self, stage_data: StageData):
        if not self.project:
            return
        name = self.prompt_unique_name("修改阶段名称", "新名称:", text=stage_data.name,
                                       existing_names=[s.name for s in self.project.stages if s != stage_data])
        if name:
            stage_data.name = name
            self.status_label.setText(f"已更新阶段名称为 {name}")
            self.update_tree_view()
    def move_stage(self, stage_data: StageData, direction: int):
        if not self.project:
            return
        stages = self.project.stages
        try:
            idx = stages.index(stage_data)
        except ValueError:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(stages):
            return
        stages[idx], stages[new_idx] = stages[new_idx], stages[idx]
        self.last_expand_stage_id = stage_data.id
        self.update_tree_view()
    def delete_stage(self, stage_data: StageData):
        if not self.project:
            return
        self.project.stages.remove(stage_data)
        if self.current_stage == stage_data:
            self.current_stage = None
        if self.current_work_stage == stage_data:
            self.set_current_work_stage(self.project.stages[0] if self.project.stages else None)
        self.update_tree_view()
    def rename_scene(self, scene_data: SceneData):
        stage = self._find_stage_for_scene(scene_data) or self.current_stage
        if not stage:
            return
        name = self.prompt_unique_name("修改场景名称", "新名称:", text=scene_data.name,
                                       existing_names=sorted({s.name for s in stage.scenes if s.name != scene_data.name}))
        if name:
            old_name = scene_data.name
            for scene in stage.scenes:
                if scene.name == old_name:
                    scene.name = name
            self._rename_group_scene_refs(stage, old_name, name)
            self.status_label.setText(f"已更新场景名称为 {name}")
            self.update_tree_view()

    def rename_scene_group(self, stage: Optional[StageData], scene_name: str):
        if not stage:
            return
        name = self.prompt_unique_name("修改场景名称", "新名称:", text=scene_name,
                                       existing_names=sorted({s.name for s in stage.scenes if s.name != scene_name}))
        if name:
            for scene in stage.scenes:
                if scene.name == scene_name:
                    scene.name = name
            self._rename_group_scene_refs(stage, scene_name, name)
            self.status_label.setText(f"已更新场景名称为 {name}")
            self.update_tree_view()

    def delete_scene_group(self, stage: Optional[StageData], scene_name: str):
        if not stage:
            return
        stage.scenes = [scene for scene in stage.scenes if scene.name != scene_name]
        self._remove_group_item_refs(stage, lambda ref: ref.scene_name == scene_name)
        if self.current_scene and self.current_scene.name == scene_name:
            self.current_scene = None
            self.clear_scene_display()
        self.status_label.setText(f"已删除场景 {scene_name} 的所有分辨率。")
        self.update_tree_view()

    def delete_scene(self, scene_data: SceneData):
        stage = self._find_stage_for_scene(scene_data) or self.current_stage
        if not stage:
            return
        scene_name = scene_data.name
        stage.scenes.remove(scene_data)
        if not any(scene.name == scene_name for scene in stage.scenes):
            self._remove_group_item_refs(stage, lambda ref: ref.scene_name == scene_name)
        if self.current_scene == scene_data:
            self.current_scene = None
            self.canvas.scene.clear()
            self.canvas.current_pixmap = None
            self.canvas.crosshair_h = None
            self.canvas.crosshair_v = None
            self.canvas.hide_crosshair()
            self.update_coord_display(None, None)
        self.update_tree_view()
    def get_stage_item_names(self, stage_data: StageData, item_type: str):
        if not stage_data:
            return []
        names = []
        for scene in stage_data.scenes:
            for item in scene.items:
                if item.item_type == item_type:
                    names.append(item.name)
        return names
    def set_area_match_mode(self, item_data: ItemData):
        if item_data.item_type != "area":
            return
        options = ["gray", "rgb", "hsv"]
        current_mode = item_data.match_mode if item_data.match_mode in options else "gray"
        current_index = options.index(current_mode)
        mode, ok = QInputDialog.getItem(
            self,
            "设置匹配模式",
            "请选择区域匹配模式:",
            options,
            current_index,
            False,
        )
        if not ok or not mode:
            return
        item_data.match_mode = mode
        if self.current_scene:
            for peer in self._find_scene_peers(self.current_scene):
                target = self._find_scene_item(peer, item_data.item_type, item_data.name)
                if target:
                    target.match_mode = mode
        self.status_label.setText(f"已将区域 {item_data.name} 的匹配模式设置为 {mode}")
        self.update_tree_view()
        if self.current_scene:
            self.canvas.redraw_overlays(self.current_scene)
    def set_special_area_range(self, item_data: ItemData):
        if not self.canvas.current_pixmap:
            QMessageBox.warning(self, "错误", "请先抓图再进行设置。")
            return
        text, ok = QInputDialog.getText(
            self,
            "设置区域范围",
            "请输入归一化坐标 x1,y1,x2,y2 (0-1):"
        )
        if not ok or not text:
            return
        try:
            parts = [float(p.strip()) for p in text.split(",")]
            if len(parts) != 4:
                raise ValueError
            x1, y1, x2, y2 = parts
        except ValueError:
            QMessageBox.critical(self, "输入错误", "格式错误，请输入 4 个用逗号分隔的数字。")
            return
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            QMessageBox.critical(self, "输入错误", "坐标需满足 0<=x1<x2<=1 且 0<=y1<y2<=1。")
            return
        pixmap = self.canvas.current_pixmap.pixmap()
        if pixmap.isNull():
            QMessageBox.critical(self, "错误", "当前图片无效。")
            return
        w = pixmap.width()
        h = pixmap.height()
        item_data.rect = RectData(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h)
        self.status_label.setText(f"已更新 {item_data.name} 的区域范围。")
        self.update_tree_view()
        self.canvas.redraw_overlays(self.current_scene)
    def copy_special_area_coords(self, item_data: ItemData):
        pixmap = None
        if self.canvas.current_pixmap:
            pixmap = self.canvas.current_pixmap.pixmap()
        elif self.current_scene and self.current_scene.pixmap:
            pixmap = self.current_scene.pixmap
        elif self.current_scene and self.current_scene.image_path and os.path.exists(self.current_scene.image_path):
            pixmap = QPixmap(self.current_scene.image_path)
        if not pixmap or pixmap.isNull():
            QMessageBox.warning(self, "复制失败", "当前图片无效，无法计算归一化坐标。")
            return
        w = pixmap.width()
        h = pixmap.height()
        if w <= 0 or h <= 0:
            QMessageBox.warning(self, "复制失败", "图片尺寸无效，无法计算归一化坐标。")
            return
        rect = item_data.rect
        x1 = rect.x / w
        y1 = rect.y / h
        x2 = (rect.x + rect.w) / w
        y2 = (rect.y + rect.h) / h
        text = f"{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f}"
        QGuiApplication.clipboard().setText(text)
        self.status_label.setText(f"已复制 {item_data.name} 归一化坐标: {text}")
    def prompt_unique_name(self, title, label, existing_names, text=""):
        while True:
            name, ok = QInputDialog.getText(self, title, label, text=text)
            if not ok:
                return None
            name = name.strip()
            if not name:
                return None
            if name in existing_names:
                QMessageBox.critical(self, "名称重复", "名称重复，请重新输入。")
                text = name
                continue
            return name
    def collect_expanded_ids(self):
        expanded_stage_ids = set()
        expanded_scene_ids = set()
        def walk(node):
            for i in range(node.childCount()):
                child = node.child(i)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(data, StageData) and child.isExpanded():
                    expanded_stage_ids.add(data.id)
                if isinstance(data, SceneData) and child.isExpanded():
                    expanded_scene_ids.add(data.id)
                walk(child)
        root = self.tree.invisibleRootItem()
        walk(root)
        return expanded_stage_ids, expanded_scene_ids
    def restore_expanded_ids(self, expanded_ids, stage_items, scene_items):
        expanded_stage_ids, expanded_scene_ids = expanded_ids
        for stage_id in expanded_stage_ids:
            item = stage_items.get(stage_id)
            if item:
                item.setExpanded(True)
        for scene_id in expanded_scene_ids:
            item = scene_items.get(scene_id)
            if item:
                item.setExpanded(True)
    def toggle_visibility(self, item_data):
        item_data.visible = not item_data.visible
        self.update_tree_view()
        self.canvas.redraw_overlays(self.current_scene)
    def get_project_root_dir(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        grand_parent_dir = os.path.dirname(parent_dir)

        # 新结构优先：PROJECT_ROOT/aw/autogame/tools/main.py -> 优先使用 PROJECT_ROOT/aw/autogame
        if os.path.basename(script_dir).lower() == "tools" and os.path.basename(parent_dir).lower() == "autogame":
            if os.path.isdir(os.path.join(parent_dir, "customs_examples")) or os.path.isdir(
                    os.path.join(parent_dir, "customs_game_examples")):
                return parent_dir
            if os.path.isdir(os.path.join(grand_parent_dir, "customs_examples")) or os.path.isdir(
                    os.path.join(grand_parent_dir, "customs_game_examples")):
                return grand_parent_dir
            return parent_dir

        # 先向上查找已存在的导出根目录（兼容旧结构与新结构）。
        for candidate in [script_dir, parent_dir, grand_parent_dir]:
            if os.path.isdir(os.path.join(candidate, "customs_examples")) or os.path.isdir(
                    os.path.join(candidate, "customs_game_examples")):
                return candidate

        # 旧结构兜底：PROJECT_ROOT/tools/main.py -> PROJECT_ROOT
        if os.path.basename(script_dir).lower() == "tools":
            return parent_dir
        return script_dir
    def ensure_special_scene_handler(self, project_dir, special_area_names, preserved_content=None):
        resource_dir = os.path.join(project_dir, "resource")
        os.makedirs(resource_dir, exist_ok=True)
        handler_path = os.path.join(resource_dir, "SpecialSceneHandler.py")
        base_content = ""
        if preserved_content is not None:
            base_content = preserved_content
        elif os.path.exists(handler_path):
            with open(handler_path, "r", encoding="utf-8") as f:
                base_content = f.read()
        normalized_names = sorted({name for name in special_area_names if isinstance(name, str) and name.strip()})
        func_name_map = {}
        used_func_names = set()
        for raw_name in normalized_names:
            func_name = re.sub(r"\W+", "_", raw_name.strip()).strip("_")
            if not func_name:
                func_name = "special_area"
            if func_name[0].isdigit():
                func_name = f"special_{func_name}"
            if keyword.iskeyword(func_name):
                func_name = f"{func_name}_handler"
            base_name = func_name
            idx = 2
            while func_name in used_func_names:
                func_name = f"{base_name}_{idx}"
                idx += 1
            used_func_names.add(func_name)
            func_name_map[raw_name] = func_name
        required_funcs = [func_name_map[name] for name in normalized_names]
        existing_funcs = set()
        if base_content.strip():
            try:
                parsed = ast.parse(base_content)
                existing_funcs = {node.name for node in parsed.body if isinstance(node, ast.FunctionDef)}
            except SyntaxError:
                existing_funcs = set(re.findall(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", base_content, flags=re.MULTILINE))
        missing_funcs = [func_name for func_name in required_funcs if func_name not in existing_funcs]
        if not base_content.strip() and not required_funcs:
            if not os.path.exists(handler_path):
                with open(handler_path, "w", encoding="utf-8") as f:
                    f.write("# -*- coding: utf-8 -*-\n")
            return
        if not missing_funcs and preserved_content is None:
            return
        final_content = base_content
        if missing_funcs:
            if final_content and not final_content.endswith("\n"):
                final_content += "\n"
            if final_content.strip():
                final_content = final_content.rstrip() + "\n\n"
            for i, func_name in enumerate(missing_funcs):
                final_content += f"def {func_name}(img):\n"
                final_content += f"    print('process {func_name} function')\n"
                final_content += "    return\n"
                if i < len(missing_funcs) - 1:
                    final_content += "\n"
        elif not final_content.strip():
            return
        if not final_content.endswith("\n"):
            final_content += "\n"
        with open(handler_path, "w", encoding="utf-8") as f:
            f.write(final_content)

    def _scan_tree(self, root_dir):
        files = set()
        dirs = set()
        if not root_dir or not os.path.isdir(root_dir):
            return files, dirs
        for current_dir, dirnames, filenames in os.walk(root_dir):
            rel_dir = os.path.relpath(current_dir, root_dir)
            if rel_dir != ".":
                dirs.add(rel_dir)
            for dirname in dirnames:
                rel_path = os.path.normpath(os.path.join(rel_dir, dirname)) if rel_dir != "." else dirname
                dirs.add(rel_path)
            for filename in filenames:
                rel_path = os.path.normpath(os.path.join(rel_dir, filename)) if rel_dir != "." else filename
                files.add(rel_path)
        return files, dirs

    def _file_digest(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _files_equal(self, left_path, right_path):
        try:
            if os.path.getsize(left_path) != os.path.getsize(right_path):
                return False
            return self._file_digest(left_path) == self._file_digest(right_path)
        except OSError:
            return False

    def _backup_file_for_sync(self, target_root, rel_path, rollback_dir, backed_up_files):
        if rel_path in backed_up_files:
            return
        source_path = os.path.join(target_root, rel_path)
        if not os.path.isfile(source_path):
            return
        backup_path = os.path.join(rollback_dir, "files", rel_path)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(source_path, backup_path)
        backed_up_files.add(rel_path)

    def _rollback_incremental_sync(self, target_dir, rollback_dir, created_files, backed_up_files, removed_dirs):
        for rel_path in sorted(created_files, reverse=True):
            target_path = os.path.join(target_dir, rel_path)
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except OSError:
                    pass
        for rel_path in sorted(backed_up_files):
            backup_path = os.path.join(rollback_dir, "files", rel_path)
            target_path = os.path.join(target_dir, rel_path)
            if os.path.exists(backup_path):
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(backup_path, target_path)
        for rel_dir in sorted(removed_dirs):
            os.makedirs(os.path.join(target_dir, rel_dir), exist_ok=True)

    def _create_export_progress_dialog(self, total_steps):
        dialog = QProgressDialog("正在准备导出...", None, 0, max(1, int(total_steps)), self)
        dialog.setWindowTitle("导出中")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        QApplication.processEvents()
        return dialog

    def _advance_export_progress(self, dialog, progress_state, text=None, step=1, extra_total=0):
        if dialog is None or progress_state is None:
            return
        if extra_total:
            progress_state["total"] = max(
                progress_state.get("current", 0),
                progress_state.get("total", 0) + int(extra_total),
            )
            dialog.setMaximum(max(1, progress_state["total"]))
        if text:
            dialog.setLabelText(text)
        progress_state["current"] = min(
            progress_state.get("total", 0),
            progress_state.get("current", 0) + max(0, int(step)),
        )
        dialog.setValue(progress_state["current"])
        QApplication.processEvents()

    def _estimate_export_generation_steps(self):
        if not self.project:
            return 1

        total = 6
        for stage in self.project.stages:
            total += 1
            for scene in stage.scenes:
                total += 1
                total += max(1, len(scene.items))
        return max(1, total)

    def _estimate_tree_copy_steps(self, root_dir):
        if not root_dir or not os.path.isdir(root_dir):
            return 1
        files, dirs = self._scan_tree(root_dir)
        return max(1, 1 + len(files) + len(dirs))

    def _estimate_sync_steps(self, staging_dir, target_dir):
        staging_files, staging_dirs = self._scan_tree(staging_dir)
        target_files, target_dirs = self._scan_tree(target_dir)
        return max(
            1,
            len(target_files - staging_files)
            + len(staging_files)
            + len(target_dirs - staging_dirs)
            + 1,
        )

    def _copy_tree_with_progress(self, source_dir, target_dir, progress_callback=None):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        if progress_callback:
            progress_callback(f"正在准备资源目录: {os.path.basename(source_dir) or 'resource'}", 1)

        for current_dir, dirnames, filenames in os.walk(source_dir):
            rel_dir = os.path.relpath(current_dir, source_dir)
            dst_dir = target_dir if rel_dir == "." else os.path.join(target_dir, rel_dir)
            os.makedirs(dst_dir, exist_ok=True)
            if rel_dir != "." and progress_callback:
                progress_callback(f"正在创建目录: {rel_dir}", 1)
            for filename in filenames:
                source_path = os.path.join(current_dir, filename)
                target_path = os.path.join(dst_dir, filename)
                shutil.copy2(source_path, target_path)
                if progress_callback:
                    rel_path = filename if rel_dir == "." else os.path.join(rel_dir, filename)
                    progress_callback(f"正在复制资源: {rel_path}", 1)

    def _sync_project_dir_incremental(self, staging_dir, target_dir, progress_callback=None):
        target_files, target_dirs = self._scan_tree(target_dir)
        staging_files, staging_dirs = self._scan_tree(staging_dir)
        dirs_pending_removal = target_dirs - staging_dirs
        rollback_dir = tempfile.mkdtemp(prefix="label_export_sync_rollback_")
        created_files = set()
        changed_files = set()
        deleted_files = set()
        backed_up_files = set()
        removed_dirs = set()
        skipped_files = 0

        try:
            for rel_path in sorted(target_files - staging_files, reverse=True):
                target_path = os.path.join(target_dir, rel_path)
                if not os.path.exists(target_path):
                    continue
                self._backup_file_for_sync(target_dir, rel_path, rollback_dir, backed_up_files)
                os.remove(target_path)
                deleted_files.add(rel_path)
                if progress_callback:
                    progress_callback(f"正在删除旧文件: {rel_path}", 1)

            for rel_path in sorted(staging_files):
                source_path = os.path.join(staging_dir, rel_path)
                target_path = os.path.join(target_dir, rel_path)

                if os.path.isdir(target_path) and not os.path.islink(target_path):
                    nested_dirs = [
                        rel_dir for rel_dir in dirs_pending_removal
                        if rel_dir == rel_path or rel_dir.startswith(rel_path + os.sep)
                    ]
                    for rel_dir in sorted(nested_dirs, key=lambda path: path.count(os.sep), reverse=True):
                        dir_path = os.path.join(target_dir, rel_dir)
                        if not os.path.isdir(dir_path):
                            continue
                        try:
                            os.rmdir(dir_path)
                            removed_dirs.add(rel_dir)
                            if progress_callback:
                                progress_callback(f"正在移除旧目录: {rel_dir}", 1)
                        except OSError as exc:
                            raise OSError(f"无法用文件替换目录，目录非空：{dir_path}") from exc

                if os.path.isfile(target_path) and self._files_equal(source_path, target_path):
                    skipped_files += 1
                    if progress_callback:
                        progress_callback(f"正在检查文件: {rel_path}", 1)
                    continue

                target_exists_now = os.path.exists(target_path)
                if target_exists_now:
                    self._backup_file_for_sync(target_dir, rel_path, rollback_dir, backed_up_files)
                    changed_files.add(rel_path)
                else:
                    created_files.add(rel_path)

                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(source_path, target_path)
                if progress_callback:
                    action_text = "正在更新文件" if target_exists_now else "正在新增文件"
                    progress_callback(f"{action_text}: {rel_path}", 1)

            for rel_dir in sorted(dirs_pending_removal - removed_dirs, key=lambda path: path.count(os.sep), reverse=True):
                target_path = os.path.join(target_dir, rel_dir)
                try:
                    os.rmdir(target_path)
                    removed_dirs.add(rel_dir)
                except OSError:
                    pass
                finally:
                    if progress_callback:
                        progress_callback(f"正在清理目录: {rel_dir}", 1)

            return {
                "created": len(created_files),
                "changed": len(changed_files),
                "deleted": len(deleted_files),
                "skipped": skipped_files,
            }
        except Exception:
            self._rollback_incremental_sync(target_dir, rollback_dir, created_files, backed_up_files, removed_dirs)
            raise
        finally:
            shutil.rmtree(rollback_dir, ignore_errors=True)

    # ==========================================
    # 4. 核心大脑：导出 Python 代码
    # ==========================================
    def export_project(self):
        if not self.project: return
        project_root_dir = self.get_project_root_dir()
        default_export_dir = os.path.join(project_root_dir, "customs_examples")
        os.makedirs(default_export_dir, exist_ok=True)
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", default_export_dir)
        if not export_dir:
            return
        original_project_name = self.project.name
        project_name = self.project.name
        project_dir = os.path.join(export_dir, project_name)
        preserved_handler_content = None
        imported_resource_source = self.imported_resource_dir if (
            self.imported_resource_dir and os.path.isdir(self.imported_resource_dir)
        ) else None
        export_resource_source = imported_resource_source
        export_temp_dir = None
        staging_project_dir = None
        renamed_backup_dir = None
        created_game_case_dir = None
        game_case_dir_existed = False
        project_dir_swapped = False
        progress_dialog = None
        progress_state = None
        try:
            existing_dir_strategy = None
            if os.path.exists(project_dir):
                msg = QMessageBox(self)
                msg.setWindowTitle("目录已存在")
                msg.setText("同级目录有重复的目录名，请选择处理方式。")
                replace_btn = msg.addButton("替换原工程", QMessageBox.ButtonRole.DestructiveRole)
                backup_btn = msg.addButton("备份原目录", QMessageBox.ButtonRole.ActionRole)
                rename_btn = msg.addButton("修改工程名", QMessageBox.ButtonRole.ActionRole)
                cancel_btn = msg.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == cancel_btn:
                    return
                if clicked == replace_btn:
                    existing_dir_strategy = "replace"
                    existing_resource_dir = os.path.join(project_dir, "resource")
                    if os.path.isdir(existing_resource_dir):
                        # 替换原工程时以导出瞬间的旧 resource 为准，避免导入后新增资源被误删。
                        export_resource_source = existing_resource_dir
                    if not imported_resource_source:
                        old_handler_path = os.path.join(project_dir, "resource", "SpecialSceneHandler.py")
                        if os.path.exists(old_handler_path):
                            with open(old_handler_path, "r", encoding="utf-8") as f:
                                preserved_handler_content = f.read()
                elif clicked == backup_btn:
                    existing_dir_strategy = "backup"
                    suffix = "_backup"
                    backup_dir = f"{project_dir}{suffix}"
                    idx = 1
                    while os.path.exists(backup_dir):
                        backup_dir = f"{project_dir}{suffix}{idx}"
                        idx += 1
                elif clicked == rename_btn:
                    new_name, ok = QInputDialog.getText(self, "修改工程名", "新工程名:", text=project_name)
                    if not ok or not new_name.strip():
                        return
                    self.project.name = new_name.strip()
                    project_name = self.project.name
                    project_dir = os.path.join(export_dir, project_name)
                    if os.path.exists(project_dir):
                        QMessageBox.critical(self, "导出失败", "修改后的工程名仍然冲突。")
                        return
            progress_total = self._estimate_export_generation_steps()
            progress_total += self._estimate_tree_copy_steps(export_resource_source) if export_resource_source else 1
            progress_total += 4
            progress_dialog = self._create_export_progress_dialog(progress_total)
            progress_state = {"current": 0, "total": progress_total}
            self._advance_export_progress(progress_dialog, progress_state, "正在创建导出暂存目录...", 0)
            export_temp_dir = tempfile.mkdtemp(prefix="label_export_")
            staging_project_dir = os.path.join(export_temp_dir, project_name)
            customs_game_examples_dir = os.path.join(project_root_dir, "customs_game_examples")
            os.makedirs(customs_game_examples_dir, exist_ok=True)
            scenes_dir = os.path.join(staging_project_dir, "scenes")
            templates_dir = os.path.join(staging_project_dir, "templates")
            resource_dir = os.path.join(staging_project_dir, "resource")
            os.makedirs(scenes_dir, exist_ok=True)
            os.makedirs(templates_dir, exist_ok=True)
            self._advance_export_progress(progress_dialog, progress_state, "已创建导出暂存目录", 1)
            if export_resource_source:
                self._copy_tree_with_progress(
                    export_resource_source,
                    resource_dir,
                    progress_callback=lambda text, step=1: self._advance_export_progress(
                        progress_dialog, progress_state, text, step
                    ),
                )
            else:
                os.makedirs(resource_dir, exist_ok=True)
                self._advance_export_progress(progress_dialog, progress_state, "已创建空资源目录", 1)
            file_path = os.path.join(staging_project_dir, "info.py")
            # 生成代码逻辑
            stage_dict = {}
            stage_info = {}
            special_area_names = set()

            def safe_filename(name):
                safe = name.strip()
                if not safe:
                    return "unnamed"
                for ch in [os.sep, "/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
                    safe = safe.replace(ch, "_")
                return safe

            def get_scene_pixmap(scene):
                if scene.pixmap and not scene.pixmap.isNull():
                    return scene.pixmap
                if scene.image_path and os.path.exists(scene.image_path):
                    pix = QPixmap(scene.image_path)
                    if not pix.isNull():
                        return pix
                return None

            def normalize_rect(rect, width, height):
                if width <= 0 or height <= 0:
                    return [0, 0, 0, 0]
                x1 = rect.x / width
                y1 = rect.y / height
                x2 = (rect.x + rect.w) / width
                y2 = (rect.y + rect.h) / height
                return [x1, y1, x2, y2]

            for index, stage in enumerate(self.project.stages):
                self._advance_export_progress(progress_dialog, progress_state, f"正在导出阶段: {stage.name}", 1)
                stage_dict[stage.name] = index == 0
                stage_entry = {
                    "groups": self._serialize_stage_groups(stage),
                    "scenes": {},
                }
                stage_safe_name = safe_filename(stage.name)
                scenes_stage_dir = os.path.join(scenes_dir, stage_safe_name)
                os.makedirs(scenes_stage_dir, exist_ok=True)
                template_stage_dir = os.path.join(templates_dir, stage_safe_name)
                os.makedirs(template_stage_dir, exist_ok=True)
                for scene_name, scene_versions in self._group_scenes_by_name(stage).items():
                    scene_safe_name = safe_filename(scene_name)
                    exported_versions = {}
                    has_multiple_versions = len(scene_versions) > 1
                    for scene in scene_versions:
                        self._advance_export_progress(
                            progress_dialog,
                            progress_state,
                            f"正在处理场景: {stage.name} / {scene.name}",
                            1,
                        )
                        scene_pixmap = get_scene_pixmap(scene)
                        scene_width = 0
                        scene_height = 0
                        if scene_pixmap:
                            scene_width = scene_pixmap.width()
                            scene_height = scene_pixmap.height()
                        elif scene.image_path and os.path.exists(scene.image_path):
                            temp_pix = QPixmap(scene.image_path)
                            if not temp_pix.isNull():
                                scene_width = temp_pix.width()
                                scene_height = temp_pix.height()
                        scene.image_width = scene_width
                        scene.image_height = scene_height
                        resolution_key = f"{scene_width}_{scene_height}"
                        name_suffix = f"__{resolution_key}" if has_multiple_versions else ""
                        scene_image_name = f"{scene_safe_name}{name_suffix}.png"
                        scene_image_path = os.path.join(scenes_stage_dir, scene_image_name)
                        if scene_pixmap:
                            scene_pixmap.save(scene_image_path)
                        elif scene.image_path and os.path.exists(scene.image_path):
                            shutil.copy(scene.image_path, scene_image_path)
                        scene_entry = {
                            "image": os.path.join("scenes", stage_safe_name, scene_image_name),
                            "width": scene_width,
                            "height": scene_height,
                            "areas": {},
                            "points": {},
                            "special_areas": {},
                        }
                        for item in scene.items:
                            self._advance_export_progress(
                                progress_dialog,
                                progress_state,
                                f"正在导出标注: {stage.name} / {scene.name} / {item.name}",
                                1,
                            )
                            rect_norm = normalize_rect(item.rect, scene_width, scene_height)
                            item_template_stem = f"{scene_safe_name}{name_suffix}__{safe_filename(item.name)}"
                            if item.item_type == "area":
                                scope_norm = normalize_rect(item.search_scope, scene_width, scene_height) if item.search_scope else [0, 0, 0, 0]
                                template_name = f"{item_template_stem}.png"
                                template_path = os.path.join(template_stage_dir, template_name)
                                if scene_pixmap:
                                    crop_rect = QRectF(item.rect.x, item.rect.y, item.rect.w, item.rect.h).toRect()
                                    cropped = scene_pixmap.copy(crop_rect)
                                    cropped.save(template_path)
                                scene_entry["areas"][item.name] = {
                                    "rect": rect_norm,
                                    "search_scope": scope_norm,
                                    "match_mode": item.match_mode or "gray",
                                    "template": os.path.join("templates", stage_safe_name, template_name),
                                }
                            elif item.item_type == "control":
                                scene_entry["points"][item.name] = {"rect": rect_norm}
                            elif item.item_type == "special_area":
                                special_area_names.add(item.name)
                                template_name = f"{item_template_stem}.png"
                                template_path = os.path.join(template_stage_dir, template_name)
                                if scene_pixmap:
                                    crop_rect = QRectF(item.rect.x, item.rect.y, item.rect.w, item.rect.h).toRect()
                                    cropped = scene_pixmap.copy(crop_rect)
                                    cropped.save(template_path)
                                scene_entry["special_areas"][item.name] = {
                                    "rect": rect_norm,
                                    "template": os.path.join("templates", stage_safe_name, template_name),
                                }
                        exported_versions[resolution_key] = scene_entry
                    if has_multiple_versions:
                        stage_entry["scenes"][scene_name] = {"resolutions": exported_versions}
                    else:
                        stage_entry["scenes"][scene_name] = next(iter(exported_versions.values()))
                stage_info[stage.name] = stage_entry

            code_lines = []
            code_lines.append("# -*- coding: utf-8 -*-")
            code_lines.append(f"# Auto-generated project: {project_name}")
            code_lines.append(f"PROJECT_NAME = {repr(project_name)}")
            code_lines.append("\nSTAGE_DICT = {")
            for key, value in stage_dict.items():
                code_lines.append(f"    {repr(key)}: {value},")
            code_lines.append("}")
            code_lines.append("\nSTAGE_INFO = {")
            for stage_name, data in stage_info.items():
                code_lines.append(f"    {repr(stage_name)}: {repr(data)},")
            code_lines.append("}")

            with open(file_path, "w", encoding='utf-8') as f:
                f.write("\n".join(code_lines))
            self._advance_export_progress(progress_dialog, progress_state, "正在生成 info.py", 1)
            if imported_resource_source:
                self.ensure_special_scene_handler(staging_project_dir, special_area_names)
            else:
                self.ensure_special_scene_handler(staging_project_dir, special_area_names, preserved_handler_content)
            self._advance_export_progress(progress_dialog, progress_state, "正在更新 SpecialSceneHandler.py", 1)
            created_game_case_dir = os.path.join(customs_game_examples_dir, project_name)
            game_case_dir_existed = os.path.exists(created_game_case_dir)
            os.makedirs(created_game_case_dir, exist_ok=True)
            sync_stats = None
            if existing_dir_strategy == "replace" and os.path.exists(project_dir):
                sync_steps = self._estimate_sync_steps(staging_project_dir, project_dir)
                self._advance_export_progress(
                    progress_dialog,
                    progress_state,
                    "正在计算增量替换计划...",
                    0,
                    extra_total=sync_steps,
                )
                sync_stats = self._sync_project_dir_incremental(
                    staging_project_dir,
                    project_dir,
                    progress_callback=lambda text, step=1: self._advance_export_progress(
                        progress_dialog, progress_state, text, step
                    ),
                )
                shutil.rmtree(staging_project_dir, ignore_errors=True)
                staging_project_dir = None
            elif existing_dir_strategy == "backup" and os.path.exists(project_dir):
                self._advance_export_progress(progress_dialog, progress_state, "正在备份并替换原工程...", 1)
                renamed_backup_dir = backup_dir
                os.rename(project_dir, renamed_backup_dir)
                shutil.move(staging_project_dir, project_dir)
                staging_project_dir = None
                project_dir_swapped = True
            else:
                self._advance_export_progress(progress_dialog, progress_state, "正在写入导出结果...", 1)
                shutil.move(staging_project_dir, project_dir)
                staging_project_dir = None
                project_dir_swapped = True
            self._advance_export_progress(progress_dialog, progress_state, "导出完成", 0)
            if progress_dialog and progress_state:
                progress_dialog.setValue(progress_state["total"])
                QApplication.processEvents()
            if sync_stats:
                QMessageBox.information(
                    self,
                    "成功",
                    f"项目已导出至 {project_dir}\n"
                    f"新增 {sync_stats['created']} 个文件，更新 {sync_stats['changed']} 个文件，"
                    f"删除 {sync_stats['deleted']} 个文件，跳过 {sync_stats['skipped']} 个未改动文件。",
                )
            else:
                QMessageBox.information(self, "成功", f"项目已导出至 {project_dir}")
        except Exception as exc:
            if staging_project_dir and os.path.exists(staging_project_dir):
                shutil.rmtree(staging_project_dir, ignore_errors=True)
            if project_dir_swapped and os.path.exists(project_dir):
                shutil.rmtree(project_dir, ignore_errors=True)
            if renamed_backup_dir and os.path.exists(renamed_backup_dir) and not os.path.exists(project_dir):
                os.rename(renamed_backup_dir, project_dir)
            if created_game_case_dir and not game_case_dir_existed and os.path.isdir(created_game_case_dir):
                try:
                    os.rmdir(created_game_case_dir)
                except OSError:
                    pass
            self.project.name = original_project_name
            QMessageBox.critical(self, "导出失败", f"导出失败，已恢复导出前状态。\n\n{exc}")
        finally:
            if progress_dialog is not None:
                progress_dialog.close()
            if export_temp_dir and os.path.isdir(export_temp_dir):
                shutil.rmtree(export_temp_dir, ignore_errors=True)
    def load_project_from_dir(self, import_dir):
        import_dir = os.path.abspath(os.fspath(import_dir))
        py_path = os.path.join(import_dir, "info.py")
        if not os.path.exists(py_path):
            raise FileNotFoundError("未找到 info.py。")
        data_env = {}
        with open(py_path, "r", encoding="utf-8") as f:
            exec(f.read(), data_env)
        project_name = data_env.get("PROJECT_NAME") or os.path.basename(import_dir.rstrip(os.sep))
        stage_info = data_env.get("STAGE_INFO", {})
        if not isinstance(stage_info, dict):
            raise ValueError("项目脚本缺少 STAGE_INFO。")
        new_project = ProjectData(name=project_name)
        for stage_name, stage_data in stage_info.items():
            stage = StageData(id=str(random.randint(1000, 9999)), name=stage_name)
            stage_control_names = {}
            scenes = stage_data.get("scenes", {}) if isinstance(stage_data, dict) else {}
            for scene_name, scene_data in scenes.items():
                scene_versions = scene_data.get("resolutions", {}) if isinstance(scene_data, dict) else {}
                if not scene_versions:
                    scene_versions = {"": scene_data}
                for _, scene_version_data in scene_versions.items():
                    scene = self._import_scene_version(import_dir, stage_name, scene_name, scene_version_data, stage_control_names)
                    stage.scenes.append(scene)
            stage.groups = self._deserialize_stage_groups(stage_data.get("groups", {}), stage) if isinstance(stage_data, dict) else default_stage_groups()
            stage.active_group_name = DEFAULT_GROUP_NAME
            new_project.stages.append(stage)
        self.project = new_project
        resource_dir = os.path.join(import_dir, "resource")
        self.imported_resource_dir = resource_dir if os.path.isdir(resource_dir) else None
        self.current_stage = None
        self.set_current_work_stage(None)
        self.current_scene = None
        self.update_tree_view()
        if self.project.stages:
            first_stage = self.project.stages[0]
            self.current_stage = first_stage
            self.set_current_work_stage(first_stage)
            if first_stage.scenes:
                first_scene = first_stage.scenes[0]
                self.current_scene = first_scene
                self.select_data_in_tree(first_scene)
            else:
                self.select_data_in_tree(first_stage)
        self.btn_add_stage.setEnabled(True)
        self.status_label.setText(f"工程 {project_name} 已导入。")
        return project_name

    def import_project(self):
        import_dir = QFileDialog.getExistingDirectory(self, "选择导入目录")
        if not import_dir:
            return
        try:
            self.load_project_from_dir(import_dir)
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", f"无法导入项目：\n{exc}")

    def _import_scene_version(self, import_dir, stage_name, scene_name, scene_data, stage_control_names):
        scene = SceneData(id=str(random.randint(1000, 9999)), name=scene_name)
        image_rel = scene_data.get("image", "")
        image_path = os.path.join(import_dir, image_rel) if image_rel else ""
        scene.image_path = image_path
        if image_path and os.path.exists(image_path):
            pix = QPixmap(image_path)
            if not pix.isNull():
                scene.pixmap = pix
        width = scene_data.get("width", 0) or (scene.pixmap.width() if scene.pixmap else 0)
        height = scene_data.get("height", 0) or (scene.pixmap.height() if scene.pixmap else 0)
        scene.image_width = int(width)
        scene.image_height = int(height)

        def denormalize_rect(rect_norm):
            if not rect_norm or width <= 0 or height <= 0:
                return RectData(0, 0, 0, 0)
            x1, y1, x2, y2 = rect_norm
            return RectData(x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height)

        for name, area_data in scene_data.get("areas", {}).items():
            rect = denormalize_rect(area_data.get("rect"))
            scope = denormalize_rect(area_data.get("search_scope"))
            item = ItemData(
                id=str(random.randint(10000, 99999)),
                name=name,
                item_type="area",
                rect=rect,
                search_scope=scope,
                match_mode=area_data.get("match_mode", "gray"),
            )
            scene.items.append(item)
        for name, point_data in scene_data.get("points", {}).items():
            previous_scene_name = stage_control_names.get(name)
            if previous_scene_name is not None and previous_scene_name != scene_name:
                raise ValueError(f"阶段内控点名称重复: {stage_name} -> {name}")
            stage_control_names[name] = scene_name
            rect = denormalize_rect(point_data.get("rect"))
            item = ItemData(
                id=str(random.randint(10000, 99999)),
                name=name,
                item_type="control",
                rect=rect
            )
            scene.items.append(item)
        for name, sp_data in scene_data.get("special_areas", {}).items():
            rect = denormalize_rect(sp_data.get("rect"))
            item = ItemData(
                id=str(random.randint(10000, 99999)),
                name=name,
                item_type="special_area",
                rect=rect
            )
            scene.items.append(item)
        return scene
# ==========================================
# 5. 启动入口
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AutoStudioWindow()
    window.show()
    sys.exit(app.exec())
