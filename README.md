# auto_game 使用指南

这是一套面向游戏自动化开发的工程，包含三部分能力：

1. 标注工具：把游戏画面里的按钮、特征图、特殊区域可视化标出来。
2. 自动化引擎：按“阶段 -> 场景 -> 标注项”的配置识别当前画面，并驱动点击、滑动、长按等操作。
3. 启动与归档工具：支持通过 `main.py` 或 `launcher.py` 启动任务，并自动归档运行时的帧图和日志。

如果你是第一次接触这个工程，建议先看“5 分钟上手”，再看“标注工具”和“脚本开发”两部分。

## 1. 先理解整体原理

这个工程的核心思路是：

1. 先用标注工具把游戏资源导出成 `info.py + scenes + templates + resource/SpecialSceneHandler.py`。
2. 自动化引擎运行时会加载 `info.py`，按当前阶段识别画面中的区域、控点、特殊区域。
3. 你在 `customs_game_examples/<project_case>/<target_case>.py` 中编写 `on_stage(w)` 逻辑。
4. 引擎每一帧都会回调一次 `on_stage(w)`，你在里面根据 `w.get_info(...)` 的结果决定点什么、切什么阶段、何时停止。

可以把它理解成：

- `customs_examples` 负责“眼睛”和“地图”
- `customs_game_examples` 负责“脑子”
- `testcases` / `main.py` / `launcher.py` 负责“怎么启动”

## 2. 5 分钟上手

### 2.1 最短路径

1. 准备好 Python 环境，并确保设备已连接，`hdc` 可用。
2. 运行标注工具：

```bash
python aw/autogame/tools/Label.py
```

3. 新建项目，完成阶段/场景/区域/控点/特殊区域标注。
4. 导出项目，导出结果会进入：

```text
aw/autogame/customs_examples/<project_case>/
```

5. 在下面这个目录创建你的业务逻辑脚本：

```text
aw/autogame/customs_game_examples/<project_case>/<target_case>.py
```

6. 在脚本里实现 `on_stage(w)`。
7. 如果要走完整 testcase 流程，再写一个 `testcases/.../*.py` 启动脚本。
8. 运行：

```bash
python main.py
```

或者：

```bash
python launcher.py
```

### 2.2 你至少要记住两个名字

- `project_case`：标注导出的资源工程名
- `target_case`：你编写的自动化逻辑脚本名

它们必须一一对应。例如：

```python
project_case = "Auto_PUBG_ALL"
target_case = "auto_pubg"
```

引擎就是靠这两个名字，去找到：

- `aw/autogame/customs_examples/Auto_PUBG_ALL/info.py`
- `aw/autogame/customs_game_examples/Auto_PUBG_ALL/auto_pubg.py`

## 3. 目录结构怎么理解

请尽量保持现有目录结构，不要随意改层级。这样后续只提交你改动的工程目录和脚本目录即可。

```text
auto_game/
├── launcher.py
├── main.py
├── testcases/
├── aw/
│   └── autogame/
│       ├── customs_examples/
│       │   └── <project_case>/
│       │       ├── info.py
│       │       ├── scenes/
│       │       ├── templates/
│       │       └── resource/
│       │           └── SpecialSceneHandler.py
│       ├── customs_game_examples/
│       │   └── <project_case>/
│       │       └── <target_case>.py
│       ├── tools/
│       │   ├── Label.py
│       │   ├── GameAutomator.py
│       │   ├── GameFrameWorker.py
│       │   └── Utils.py
│       └── temp/
```

### 3.1 各目录职责

- `customs_examples/<project_case>`：标注导出的资源目录
- `customs_game_examples/<project_case>`：你自己写的业务逻辑
- `testcases`：测试入口，适合接 SP、日志抓取、性能分析、自动启动游戏等流程
- `launcher.py`：图形化启动器，适合手工选择 testcase、配置运行次数、查看实时预览
- `main.py`：轻量入口，适合直接运行或联调
- `aw/autogame/temp`：运行时日志、帧图和归档结果

## 4. 标注工具怎么用

标注工具入口：

```bash
python aw/autogame/tools/Label.py
```

### 4.1 三层核心概念

- 阶段 `Stage`：长流程节点，例如“开始游戏阶段”“跑图阶段”“开车阶段”
- 场景 `Scene`：阶段内的具体画面，例如“游戏场景”“驾驶”“弹窗”
- 标注项 `Item`：场景内的具体对象

标注项分三类：

- 区域 `Area`：用于模板匹配，判断某个特征是否出现
- 控点 `Control`：用于点击、长按、拖拽、双指操作的交互位置
- 特殊区域 `Special Area`：用于读取更复杂的信息，例如小地图位置、方向、障碍物、OCR 内容等

### 4.2 常用操作

- 新建项目：菜单栏 `文件 -> 新建项目`
- 导入项目：菜单栏 `文件 -> 导入项目`
- 导出项目：菜单栏 `文件 -> 导出项目`
- 抓图：从已连接设备实时抓取
- 导入图片：使用本地图片做标注

### 4.3 标注建议

#### Area

- 用来判断“某个特征是否出现”
- 适合标按钮图标、提示文本、状态图标、固定 UI 元素
- 尽量框选特征明显、背景干净的区域
- 如果目标只会出现在局部位置，建议设置 `Search Scope`，可以提高识别速度和准确率

#### Control

- 用来执行交互
- 通常框按钮中心区域即可
- 自动化时会取中心点进行点击或滑动

#### Special Area

- 用来做“是否出现”之外的逻辑
- 例如解析位置、方位、障碍物、颜色统计、OCR 结果
- 导出时会自动在 `resource/SpecialSceneHandler.py` 里生成对应函数
- 你只需要在这些函数里补业务逻辑

### 4.4 快捷键和画布交互

当前工具支持这些快捷操作：

- `Ctrl + A`：添加区域
- `Ctrl + C`：添加控点
- `Ctrl + S`：添加特殊区域
- `Ctrl + 鼠标滚轮`：缩放画布
- 鼠标右键：打开当前对象菜单
- 可直接拖动标注框：调整位置

### 4.5 导出后会生成什么

导出后会在 `aw/autogame/customs_examples/<project_case>/` 下生成：

- `info.py`：核心配置，包含 `STAGE_INFO`
- `scenes/`：原始场景截图
- `templates/`：Area 和 Special Area 裁剪出的模板图
- `resource/SpecialSceneHandler.py`：特殊区域处理函数

最重要的是：导出后不需要手工移动文件，直接保持当前结构即可。

## 5. 自动化脚本怎么写

你真正写业务逻辑的地方在：

```text
aw/autogame/customs_game_examples/<project_case>/<target_case>.py
```

脚本通常会导出一个 `on_stage(w)`，引擎每一帧都会调用它。

### 5.1 最小脚本结构

```python
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aw.autogame.tools.GameFrameWorker import FrameWorker


def on_stage(w: "FrameWorker"):
    if w.current_stage == "登录界面":
        if w.get_info("确认按钮"):
            w.click("确认按钮")
            time.sleep(2)
            w.refresh_frame()

    elif w.current_stage == "大厅":
        if w.get_info("进入战斗"):
            w.click("进入战斗")
            w.change_stage("战斗中")

        if w.get_info("退出"):
            w.stop()
```

### 5.2 `on_stage(w)` 的执行方式

你可以把它理解成一个“每帧决策函数”：

1. 当前帧先做图像识别
2. 引擎把识别结果放到 `w.stage_info`
3. 然后调用一次 `on_stage(w)`
4. 你根据当前阶段和识别结果做操作

所以脚本写法应尽量遵循：

- 先判断 `w.current_stage`
- 再判断 `w.get_info(...)`
- 执行动作后 `sleep + w.refresh_frame()`
- 需要切流程时调用 `w.change_stage(...)`

## 6. 常用 API 速查

这些接口都来自 `FrameWorker`，是写脚本时最常用的一层。

### 6.1 流程控制

- `w.current_stage`
  读取当前激活阶段名

- `w.change_stage("阶段名")`
  切换逻辑阶段

- `w.refresh_frame()`
  强制刷新一帧，重新识别

- `w.stop()`
  停止整个自动化流程

### 6.2 识别接口

- `w.get_info("名称")`
  读取当前阶段下对应标注项的识别结果

返回值不是单纯布尔值：

- 如果匹配成功，通常会返回位置或识别结果
- 如果失败，通常返回 `False`

最常见写法是：

```python
if w.get_info("确认按钮"):
    ...
```

### 6.3 交互接口

- `w.click(btn_name, x_bias=0, y_bias=0)`
  单次点击

- `w.tap_single(btn, wait=100, dura=500, x_bias=0, y_bias=0)`
  单指拖动/长按/滑动

- `w.tap_double(btn1, btn2, wait=100, dura=500, x1_bias=0, y1_bias=0, x2_bias=0, y2_bias=0)`
  双指操作

- `w.click_down(btn, x_bias=0, y_bias=0, dura=0)`
  按下不松开，适合摇杆、蓄力

### 6.4 日志分析接口

- `insert_logs(task_name, time_dura, *keywords)`
  记录一段性能分析区间，供后续 `analyze_txt(...)` 使用

例如：

```python
insert_logs("战斗测试", 10, "帧率", "插帧")
```

## 7. 特殊区域是怎么工作的

特殊区域不是简单判断“有没有”，而是“拿到区域后交给你自定义处理”。

例如：

- 小地图：提取人物位置
- 朝向区域：识别当前角度
- 前景区域：做障碍物检测
- OCR 区域：识别文字

导出项目后，标注工具会自动在：

```text
aw/autogame/customs_examples/<project_case>/resource/SpecialSceneHandler.py
```

里生成对应函数。你只需要补上处理逻辑即可。

这也是为什么：

- `Area` 更像“出现判断”
- `Special Area` 更像“数据提取”

## 8. testcase 怎么写

如果你只是做最简联调，可以先直接跑 `main.py`。

但如果你需要：

- 先启动游戏
- 接 SP
- 抓设备日志
- 做性能分析
- 跑完后做清理

那么建议写 `testcases`。

### 8.1 testcase 的基本要求

- `testcases` 里的类名必须和文件名一致
- 需要生成对应的 `.json` 文件
- 必须正确设置：

```python
project_case = "你的资源工程名"
target_case = "你的逻辑脚本名"

os.environ["TARGET_PROJECT_CASE"] = project_case
os.environ["TARGET_GAME_CASE"] = target_case
```

### 8.2 testcase 示例骨架

```python
import os
import time
from devicetest.core.test_case import TestCase
from hypium import UiDriver
from hypium.action.os_hypium.device_logger import DeviceLogger
from aw.autogame.tools.GameAutomator import GameAutomator
from aw.autogame.tools.Utils import analyze_txt

project_case = "your_project"
target_case = "your_logic"

os.environ["TARGET_PROJECT_CASE"] = project_case
os.environ["TARGET_GAME_CASE"] = target_case


class your_logic(TestCase):
    def __init__(self, controllers):
        self.TAG = self.__class__.__name__
        TestCase.__init__(self, self.TAG, controllers)
        self.tests = ["test_step"]
        self.driver = UiDriver(self.device1)
        self.automator = GameAutomator(driver=self.driver, logger=self.log)
        self.task_name = os.environ.get("TARGET_GAME_CASE")
        self.device_logger = DeviceLogger(self.driver)
        self.log_path = f"aw/autogame/temp/logs/{self.task_name}.txt"
        self.frame_path = "aw/autogame/temp/logs/process_save_frames"

    def test_step(self):
        self.device_logger.start_log(self.log_path)
        self.automator.start()
        self.device_logger.stop_log()
```

## 9. `main.py` 和 `launcher.py` 怎么选

### 9.1 `main.py`

适合：

- 快速联调
- 单次运行
- 命令行环境
- 开发时反复试逻辑

当前 `main.py` 支持两种方式：

- `AUTOGAME_MAIN_MODE=direct`：直接启动自动化，适合你已经手动打开游戏
- `AUTOGAME_MAIN_MODE=testcase`：通过 testcase 跑完整流程

例如：

```bash
python main.py
```

或者：

```bash
AUTOGAME_MAIN_MODE=testcase python main.py
```

### 9.2 `launcher.py`

适合：

- 手工选择 testcase
- 配置运行次数
- 安全温度/电量控制
- 观察实时帧图
- 查看实时输出
- 每轮结束自动归档日志和帧图

运行方式：

```bash
python launcher.py
```

当前 launcher 还有这些特点：

- 支持 `testcase` 模式和 `project_case / target_case` 直启模式
- 预览区可显示当前帧识别信息
- 可以打开“阶段标注叠加”开关，仅在预览中显示按钮/区域/特殊区域
- 每次运行结束会自动把本次产物归档到 `aw/autogame/temp/game_时间_第N次用例/`

## 10. 运行时文件会落在哪里

默认运行时目录在：

```text
aw/autogame/temp/
```

常见内容包括：

- `logs/process_temp_logs/`：实时可视化帧和对应 JSON
- `logs/process_save_frames/`：其他保存帧
- `logs/*.txt`：运行日志
- `results/<task_name>/`：分析结果
- `game_年月日时分秒_第N次用例/`：每次运行结束后的归档目录

## 11. 新手最容易踩的坑

### 11.1 名字对不上

最常见的问题就是：

- `project_case` 写错
- `target_case` 写错
- 目录名和文件名不一致

结果通常会表现为：

- 找不到 `info.py`
- 找不到业务逻辑模块
- 启动时报 `TARGET_PROJECT_CASE` / `TARGET_GAME_CASE` 相关错误

### 11.2 中文路径和命名

文档建议：

- 工程目录名用英文
- 脚本名用英文

这样最稳，尤其在导入、动态加载、跨平台运行时更少出问题。

### 11.3 点击后不刷新

很多新手脚本会这样写：

```python
w.click("确定")
```

然后马上继续判断下一步，结果还是旧画面。

建议改成：

```python
w.click("确定")
time.sleep(1)
w.refresh_frame()
```

### 11.4 Area 框太大

如果 Area 带了太多背景，容易：

- 误识别
- 漏识别
- 识别变慢

优先框“最稳定、最有辨识度”的图案。

### 11.5 Search Scope 不合理

- 太大：速度慢，误判增加
- 太小：目标稍微偏一点就匹配不到

经验上应让 scope 覆盖“这个特征可能出现的活动范围”，不要比整个屏幕还大，除非它真的会全屏漂移。

### 11.6 特殊区域函数没补

标注工具只会帮你生成函数框架，不会自动实现逻辑。

如果某个 Special Area 识别不出内容，先看：

- `resource/SpecialSceneHandler.py` 里是否已经写了函数
- 函数名是否和导出的区域名对应

### 11.7 `hdc` 不在 PATH

如果抓图、设备交互、launcher 安全检查失败，先确认：

```bash
hdc list targets
```

能够正常返回设备信息。

尤其在 Windows 下，必须确保 `hdc.exe` 在系统 `PATH` 中。

## 12. 推荐开发顺序

如果你是新手，建议按这个顺序做：

1. 先完成一个最小项目导出，只做 1 个阶段、1 个场景、1 个按钮
2. 写一个只会点击这个按钮的 `on_stage(w)`
3. 用 `main.py` 跑通
4. 再加 testcase、日志抓取、性能分析
5. 最后再接复杂的特殊区域、小地图、路径规划、状态机

不要一上来就做大而全的自动化。这个工程本身是支持复杂逻辑的，但开发时最好先跑通最小闭环。

## 13. 一句话记住这套工程

这套工程的正确使用方式是：

1. 用标注工具产出资源
2. 用 `on_stage(w)` 写每帧决策逻辑
3. 用 `testcases` / `main.py` / `launcher.py` 选择启动方式
4. 用 `aw/autogame/temp` 里的日志、帧图和归档结果做调试和分析

如果你按这个思路来，基本不会走偏。
