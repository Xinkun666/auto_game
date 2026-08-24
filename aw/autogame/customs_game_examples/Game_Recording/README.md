# Game_Recording

这是一个不经过 Launcher 的华为 HOS 键盘录制入口。

## 第一次使用

1. 用标注工具加载：
   `aw/autogame/customs_examples/Game_Recording`
2. 新建一个阶段和一个场景，并导入当前手机画面。
3. 使用“控点”标注要操作的位置；控点名可以是 `w`、`a`、`s`、`d`，也可以是“前进”、“跳跃”等业务名称。
4. 导出时仍选择 `Game_Recording` 工程。

`q` 和 `e` 不再是录制开关，可以正常标成游戏控点。`w/a/s/d` 应标在同一个移动摇杆的上、左、下、右四个方向。

## 运行

在仓库根目录执行：

```bash
python aw/autogame/customs_game_examples/Game_Recording/main.py
```

默认使用 `auto` 动态兼容模式：先尝试设备已有 SO，启动不成功就继续尝试本地 `res/video` 中的其他候选；某个 SO 能启动但运行中断流，也会记录并切换下一个。只有所有候选都失败后，程序才最终停止。

启动后会先弹出“按键绑定与控点调整”窗口：

- 左侧选中标注控点后，直接按一下键盘按键完成绑定；
- 标注控点的名称不会自动当成键盘绑定；`KEY_BINDINGS` 中没有记录的控点会用红色显示“还没绑定”；
- 多场景可通过顶部的上一个/下一个按钮切换；
- 场景图上的蓝色控点可拖动，保存时会同步修改 `info.py` 的阶段场景和 `SCENE_POOL`；
- 上次的绑定保存在 `info.py` 的 `KEY_BINDINGS` 中，每次会自动加载。如果无需修改，直接点击“保存并进入录制”即可。

在该窗口点击保存之前不会启动 HOS 视频抓流；取消则直接结束本次运行。

保存后会打开一个统一窗口：

- `录制` 页：显示手机画面，可开启/关闭录制并保存新的记录；
- `回放` 页：选择历史记录并执行回放，同时复用录制页已经建立的 HOS 投屏与触控连接，不会再创建第二套 fport/HOScrcpy。开始回放后自动进入左右对比：左侧同步播放该条记录保存的 `video.mp4`，右侧显示当前手机的实时回放画面；若历史记录缺少或无法打开视频，右侧实时回放仍会正常继续。
- `对比` 页：加载已完整完成的历史回放记录，自动识别其对应的源录制；左侧播放回放视频，右侧播放源录制视频。两侧共用同一时间轴，播放、暂停、拖动进度和倍速（0.25x～3x）都会同步生效。

`start_record.py` 和 `start_replay.py` 仍保留，供旧脚本或只需要单项功能时使用；推荐日常使用统一的 `main.py`。

每个候选的启动结果和断流现场会写入运行日志及 `video_so_attempt_history`。

### 摇杆快捷标注

除了逐个标注 `w/a/s/d`，还可在同一场景标两个控点：`center` 和 `boundary`（这两个英文拼写就是标准写法，大小写不敏感）。`center` 是摇杆落指中心，`boundary` 是摇杆圆周上的任一点；两点距离就是摇杆半径。

按键绑定窗口会以 `center` 为圆心生成绿色的 `↑ ↓ ← → ↖ ↗ ↙ ↘` 八个虚拟方向。它们不是额外写入标注图的控点，可以只绑定需要使用的方向；保存后，录制和回放都会按已绑定方向使用同一中心与半径。`center` 和 `boundary` 自身无需绑定键盘按键。

如需对比设备原有版本：

```bash
python aw/autogame/customs_game_examples/Game_Recording/main.py --video-so reuse
```

也可以指定本地已存在的完整文件名：

```bash
python aw/autogame/customs_game_examples/Game_Recording/main.py \
  --video-so libscrcpy_server_unix_6.3.1-20260113.z.so
```

默认使用 HOS 触控。如需改用 `sendevent`：

```bash
python aw/autogame/customs_game_examples/Game_Recording/main.py \
  --touch-backend sendevent
```

程序会依次尝试设备端 `getevent -lp`、`getevent -p` 和 `/data/test/getevent -p`，自动识别触摸设备及 ABS 坐标范围。如果该手机无法自动探测，可以手动指定：

```bash
python aw/autogame/customs_game_examples/Game_Recording/main.py \
  --touch-backend sendevent \
  --sendevent-device event2 \
  --sendevent-max-x 10799 \
  --sendevent-max-y 23999
```

手动值必须以该手机 `getevent -p` 的实际结果为准，上面只是格式示例。`sendevent` 还需要 HDC Shell 具有写入 `/dev/input/eventX` 的权限；权限不足时错误会进入本次 `start_record.log` 和 `run_summary.json`。

录制窗口出现手机画面后：

- 在“录制名称”中可选输入名称；留空时继续使用默认时间戳目录
- 点击“开启录制”：开始录制，按钮同时变为“关闭录制”
- 点击“关闭录制”：停止并保存
- `q/e` 不再控制录制，可以在绑定界面中当作普通游戏键使用
- 中间按已标注的键位：控制手机并写入动作记录
- 按住任意已绑定控点（包括 `w/a/s/d`）时，每次新按一次键盘方向键，都会从该控点的初始位置执行一次约 0.12 秒的短滑动轨迹；长按方向键不会连续滑动

本次运行默认保存到：

`aw/autogame/customs_examples/Game_Recording/records/<时间戳>/`

开启录制后的子目录包括 `video.mp4`、`initial_view.png`、`action_raw.json`、`action_step.json` 和 `session.json`。如果填写了录制名称，该子目录就使用所填名称；同名目录不会被覆盖。

## 单独回放（兼容入口）

在仓库根目录执行：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_replay.py
```

启动后先弹出历史记录选择窗口，记录按录制时间从新到旧排列。选中后可查看初始画面、时长、动作数、视频帧数和结束原因；双击记录或点击“开始回放”后，程序才会连接手机。手机画面首帧到达后自动按原时间执行动作。

回放优先使用 `action_raw.json` 中精确的按下/松开及离散滑动事件；如果该文件为空或不存在，则使用 `action_step.json` 恢复键位状态。触控坐标优先使用该次录制 `session.json` 中的布局，并自动缩放到当前手机分辨率。离散滑动会按录制时的实际归一化坐标回放；如果含滑动的记录丢失了 `action_raw.json`，程序会拒绝不完整回放。回放完成、失败或中途取消时都会强制释放触点。

`start_replay.py` 与录制脚本一样支持 `--video-so`、`--touch-backend sendevent` 及手动 sendevent 设备参数。如果记录位于其他目录，可使用：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_replay.py \
  --records /path/to/records
```

## 断连处理和日志

`main.py` 每启动一次，都会先在 `records` 下创建一个唯一时间目录。本次运行的成功或失败日志、hilog、录制文件和诊断报告都只保存在该目录中。日志在脚本启动时就开始记录，不需要先开启录制。HOS 出现断连时：

- `auto` 模式不会重试已失败的 SO，而是切换到尚未尝试的候选；
- 所有 SO 都启动失败或运行中断流后，程序才最终停止；
- 程序从启动时就持续抓取 `hdc hilog`，避免手机掉线后无法补抓；
- 如果已经通过按钮开始录制，会先保存当前视频和动作，`session.json` 中的 `stop_reason` 为 `hos_disconnect`；
- 无论是否开始过录制，都会保存断连诊断和完整终端输出。
- 清理 HOS 连接前会先采集投屏进程、设备端视频端点和 `hdc fport` 状态，写入 `diagnostic.pre_cleanup_disconnect`。

手动关闭录制窗口、按 `Ctrl+C`、IDE 停止或普通异常退出时，hilog 抓取进程也会立即结束，不会留在后台继续写日志。

每次启动的目录结构：

- 本次运行目录：`records/<启动时间>/`
- 完整终端日志：`records/<启动时间>/start_record.log`
- HDC DEBUG 日志：`records/<启动时间>/hdc.log`（启动前执行 `hdc kill && hdc -l 5 start`，在 HOScrcpy 重新建立 fport 前开启；结束时从 `%TEMP%\\hdc.log` 归档）
- 实时 hilog：`records/<启动时间>/hilog.txt`
- 无论成功失败都生成：`records/<启动时间>/run_summary.json`
- 按钮产生的录制：`records/<启动时间>/recordings/<自定义名称或录制时间>/`
- 完整回放视频：`records/replays/<回放时间>/video.mp4`，同目录的 `replay_session.json` 会保存它对应的源录制目录和时长；未完整结束的回放不会保留在这里。
- 最终断连时额外生成：`records/<启动时间>/hos_disconnect.json`

如果断连时正在录制，`hilog.txt` 也会复制到当次录制子目录。如果 `hdc hilog` 无法启动，运行日志和 hilog 文件头部会记录失败原因；若 HDC DEBUG 原始日志无法归档，会额外写入 `hdc_capture_error.txt` 说明原因。
- 若断连时正在录制，录制目录内也会多一份 `hos_disconnect.json`

## 触控后端说明

该入口的视频流始终使用 HOScrcpy，触控可选 HOS 或 `sendevent`。当前键盘控制逻辑仍按单指摇杆设计，所以：

- `w/a/s/d` 开始时会先在摇杆中心落指，再滑动到标注方向；
- 新方向键会替换旧方向键，切换时必定执行“旧触点抬起 → 中心落指 → 滑动到新方向”；
- 非摇杆按钮从键盘按下到松开期间保持触点；短按仍然等价于普通点击；
- 按住任意已绑定按钮时，每次新按一次 `↑/↓/←/→` 都会先回到该按钮的初始控点，再用多个中间点完成一次约 0.12 秒的短滑动；默认距离为屏幕短边的 8%，键盘自动连发会被忽略；
- 手机上能看到触点移动、但游戏仍无反应时，通常是起点所在的按钮区域只接收点击而不接收拖拽；调整视角建议在游戏右侧空白视角区单独标注一个控点并绑定键位。
- 按住普通按钮期间会暂停摇杆触点，松开后才恢复仍在按住的摇杆方向；
- 这不是两个真实触点同时按下，要求严格多指并发的游戏操作仍需单独适配和真机验证。
