# Game_Recording

这是一个不经过 Launcher 的华为 HOS 键盘录制入口。

## 第一次使用

1. 用标注工具加载：
   `aw/autogame/customs_examples/Game_Recording`
2. 新建一个阶段和一个场景，并导入当前手机画面。
3. 使用“控点”标注要操作的位置；控点名可以是 `w`、`a`、`s`、`d`，也可以是“前进”、“跳跃”等业务名称。
4. 导出时仍选择 `Game_Recording` 工程。

`q` 和 `e` 已被录制开关占用，不要标成游戏控点。`w/a/s/d` 应标在同一个移动摇杆的上、左、下、右四个方向。

## 运行

在仓库根目录执行：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_record.py
```

默认使用 `auto` 动态兼容模式：先尝试设备已有 SO，启动不成功就继续尝试本地 `res/video` 中的其他候选；某个 SO 能启动但运行中断流，也会记录并切换下一个。只有所有候选都失败后，程序才最终停止。

启动后会先弹出“按键绑定与控点调整”窗口：

- 左侧选中标注控点后，直接按一下键盘按键完成绑定；
- 多场景可通过顶部的上一个/下一个按钮切换；
- 场景图上的蓝色控点可拖动，保存时会同步修改 `info.py` 的阶段场景和 `SCENE_POOL`；
- 上次的绑定保存在 `info.py` 的 `KEY_BINDINGS` 中，每次会自动加载。如果无需修改，直接点击“保存并进入录制”即可。

在该窗口点击保存之前不会启动 HOS 视频抓流；取消则直接结束本次运行。

每个候选的启动结果和断流现场会写入运行日志及 `video_so_attempt_history`。

如需对比设备原有版本：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_record.py --video-so reuse
```

也可以指定本地已存在的完整文件名：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_record.py \
  --video-so libscrcpy_server_unix_6.3.1-20260113.z.so
```

默认使用 HOS 触控。如需改用 `sendevent`：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_record.py \
  --touch-backend sendevent
```

程序会依次尝试设备端 `getevent -lp`、`getevent -p` 和 `/data/test/getevent -p`，自动识别触摸设备及 ABS 坐标范围。如果该手机无法自动探测，可以手动指定：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_record.py \
  --touch-backend sendevent \
  --sendevent-device event2 \
  --sendevent-max-x 10799 \
  --sendevent-max-y 23999
```

手动值必须以该手机 `getevent -p` 的实际结果为准，上面只是格式示例。`sendevent` 还需要 HDC Shell 具有写入 `/dev/input/eventX` 的权限；权限不足时错误会进入本次 `start_record.log` 和 `run_summary.json`。

录制窗口出现手机画面后：

- 按 `q`：开始录制
- 按 `e`：停止并保存
- 中间按已标注的键位：控制手机并写入动作记录

本次运行默认保存到：

`aw/autogame/customs_examples/Game_Recording/records/<时间戳>/`

按 `q` 后的录制子目录包括 `video.mp4`、`initial_view.png`、`action_raw.json`、`action_step.json` 和 `session.json`。

## 回放

在仓库根目录执行：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_replay.py
```

启动后先弹出历史记录选择窗口，记录按录制时间从新到旧排列。选中后可查看初始画面、时长、动作数、视频帧数和结束原因；双击记录或点击“开始回放”后，程序才会连接手机。手机画面首帧到达后自动按原时间执行动作。

回放优先使用 `action_raw.json` 中精确的按下/松开事件；如果该文件为空或不存在，则使用 `action_step.json` 恢复键位状态。触控坐标优先使用该次录制 `session.json` 中的布局，并自动缩放到当前手机分辨率。回放完成、失败或中途取消时都会强制释放触点。

`start_replay.py` 与录制脚本一样支持 `--video-so`、`--touch-backend sendevent` 及手动 sendevent 设备参数。如果记录位于其他目录，可使用：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_replay.py \
  --records /path/to/records
```

## 断连处理和日志

`start_record.py` 每启动一次，都会先在 `records` 下创建一个唯一时间目录。本次运行的成功或失败日志、hilog、录制文件和诊断报告都只保存在该目录中。日志在脚本启动时就开始记录，不需要先按 `q`。HOS 出现断连时：

- `auto` 模式不会重试已失败的 SO，而是切换到尚未尝试的候选；
- 所有 SO 都启动失败或运行中断流后，程序才最终停止；
- 程序从启动时就持续抓取 `hdc hilog`，避免手机掉线后无法补抓；
- 如果已经按 `q` 开始录制，会先保存当前视频和动作，`session.json` 中的 `stop_reason` 为 `hos_disconnect`；
- 无论是否按过 `q`，都会保存断连诊断和完整终端输出。
- 清理 HOS 连接前会先采集投屏进程、设备端视频端点和 `hdc fport` 状态，写入 `diagnostic.pre_cleanup_disconnect`。

每次启动的目录结构：

- 本次运行目录：`records/<启动时间>/`
- 完整终端日志：`records/<启动时间>/start_record.log`
- 实时 hilog：`records/<启动时间>/hilog.txt`
- 无论成功失败都生成：`records/<启动时间>/run_summary.json`
- 按 `q` 产生的录制：`records/<启动时间>/recordings/<录制时间>/`
- 最终断连时额外生成：`records/<启动时间>/hos_disconnect.json`

如果断连时正在录制，`hilog.txt` 也会复制到当次录制子目录。如果 `hdc hilog` 无法启动，运行日志和 hilog 文件头部会记录失败原因。
- 若断连时正在录制，录制目录内也会多一份 `hos_disconnect.json`

## 触控后端说明

该入口的视频流始终使用 HOScrcpy，触控可选 HOS 或 `sendevent`。当前键盘控制逻辑仍按单指摇杆设计，所以：

- `w/a/s/d` 开始时会先在摇杆中心落指，再滑动到标注方向；
- 新方向键会替换旧方向键，切换时必定执行“旧触点抬起 → 中心落指 → 滑动到新方向”；
- 按其他按钮时，会短暂松开摇杆、点击按钮，再恢复摇杆；
- 这不是两个真实触点同时按下，要求严格多指并发的游戏操作仍需单独适配和真机验证。
