# Game_Recording

这是一个不经过 Launcher 的华为 HOS 键盘录制入口。

## 第一次使用

1. 用标注工具加载：
   `aw/autogame/customs_examples/Game_Recording`
2. 新建一个阶段和一个场景，并导入当前手机画面。
3. 使用“控点”标注 `w`、`a`、`s`、`d`，也可以继续标注 `space`、`f`、`j`、`k` 等键。
4. 导出时仍选择 `Game_Recording` 工程。

`q` 和 `e` 已被录制开关占用，不要标成游戏控点。`w/a/s/d` 应标在同一个移动摇杆的上、左、下、右四个方向。

## 运行

在仓库根目录执行：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_record.py
```

默认使用 `auto` 动态兼容模式：先尝试设备已有 SO，启动不成功就继续尝试本地 `res/video` 中的其他候选；某个 SO 能启动但运行中断流，也会记录并切换下一个。只有所有候选都失败后，程序才最终停止。

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

录制窗口出现手机画面后：

- 按 `q`：开始录制
- 按 `e`：停止并保存
- 中间按已标注的键位：控制手机并写入动作记录

结果默认保存到：

`aw/autogame/customs_examples/Game_Recording/records/<时间戳>/`

每次结果包括 `video.mp4`、`initial_view.png`、`action_raw.json`、`action_step.json` 和 `session.json`。

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

## 华为单框架说明

该入口直接使用项目已有的 HOScrcpy 视频流和 HOS 触控通道。HOS 官方通道当前只暴露单指操作，所以：

- `w/a/s/d` 组合会合成为一个摇杆方向，可正常表达斜向移动；
- 按其他按钮时，会短暂松开摇杆、点击按钮，再恢复摇杆；
- 这不是两个真实触点同时按下，要求严格多指并发的游戏操作仍需单独适配和真机验证。
