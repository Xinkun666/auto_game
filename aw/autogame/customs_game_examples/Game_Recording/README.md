# Game_Recording

这是一个不经过 Launcher 的华为 HOS 键盘录制入口。

## 第一次使用

1. 用标注工具加载：
   `aw/autogame/customs_game_examples/Game_Recording`
2. 新建一个阶段和一个场景，并导入当前手机画面。
3. 使用“控点”标注 `w`、`a`、`s`、`d`，也可以继续标注 `space`、`f`、`j`、`k` 等键。
4. 导出时仍选择 `Game_Recording` 工程；启动脚本会自动读取 Label 导出的布局。

`q` 和 `e` 已被录制开关占用，不要标成游戏控点。`w/a/s/d` 应标在同一个移动摇杆的上、左、下、右四个方向。

## 运行

在仓库根目录执行：

```bash
python aw/autogame/customs_game_examples/Game_Recording/start_record.py
```

录制窗口出现手机画面后：

- 按 `q`：开始录制
- 按 `e`：停止并保存
- 中间按已标注的键位：控制手机并写入动作记录

结果默认保存到：

`aw/autogame/customs_examples/Game_Recording/records/<时间戳>/`

每次结果包括 `video.mp4`、`initial_view.png`、`action_raw.json`、`action_step.json` 和 `session.json`。

## 华为单框架说明

该入口直接使用项目已有的 HOScrcpy 视频流和 HOS 触控通道。HOS 官方通道当前只暴露单指操作，所以：

- `w/a/s/d` 组合会合成为一个摇杆方向，可正常表达斜向移动；
- 按其他按钮时，会短暂松开摇杆、点击按钮，再恢复摇杆；
- 这不是两个真实触点同时按下，要求严格多指并发的游戏操作仍需单独适配和真机验证。
