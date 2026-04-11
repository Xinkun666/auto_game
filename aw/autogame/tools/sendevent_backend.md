# Sendevent Backend 说明

当前工程里有两套触控后端：

- `uinput`
- `sendevent`

## 后端区别

`uinput` 继续沿用旧逻辑，直接使用业务层传入的坐标，不做额外的旋转换算。

`sendevent` 主要解决两件事：

- 把业务层坐标转换成设备物理触摸坐标
- 通过 `EventRecordV3_ohos` 回放事件文本

## 坐标转换

### 1. 静态控点

静态控点来自 `Label.py` 导出的 `info.py`。

`STAGE_INFO` 中每个 scene 已经带有 `width` 和 `height`，因此可以判断这套坐标是横屏记录还是竖屏记录。  
`sendevent` 模式下会结合当前设备 `rotation` 做转换。

入口在：

- [GameFrameWorker.py](/Users/liuxinkun/Downloads/auto_game/aw/autogame/tools/GameFrameWorker.py)
- [Utils.py](/Users/liuxinkun/Downloads/auto_game/aw/autogame/tools/Utils.py)

对应逻辑：

- `Controller._transform_button_pos()`
- `convert_scene_point_by_current_rotation()`

### 2. 动态识别点

动态识别点来自 `w.get_info(...)`。

这类坐标是相对于当前画面左上角的归一化坐标。  
`sendevent` 模式下会先映射到 `screen_size`，再结合当前 `rotation` 转到物理触摸坐标。

对应逻辑：

- `Controller._transform_runtime_point()`
- `convert_display_point_by_rotation()`

## Click 实现

当前 `SendEventController.click()` 没有继续复用新的多指轨迹控制器，而是单独走了一条“兼容旧 demo”的单指点击路径。

这样做的原因是：

- 你提供的 `send_event_demo2.py` 已经在设备上验证过可用
- 当前设备对单击事件的格式比较敏感
- 直接复用旧 demo 的单指回放文本更稳

当前 `click()` 的行为本质等价于旧 demo 里的：

```python
just_move(dut_handle, x, y, x, y, move_time_ms=duration_ms, down=True, up=True)
```

也就是：

1. 起点和终点是同一个点
2. 第一帧是 `down`
3. 中间是同点 contact 帧
4. 最后一帧是 `up`

对应工程内实现：

- `SendEventController.click()`
- `SendEventController._build_legacy_single_touch_text()`
- `gen_single_move_cmd_str2()`

## Tap 实现

`tap_single()` 和 `tap_double()` 仍然走 `MultiTouchController`。

这条路径适合：

- 长按
- 拖动
- 单指滑动
- 双指操作

如果后续你希望把多指和单击完全统一，也可以继续把 `MultiTouchController` 再往真实设备协议方向收敛。

## 当前结论

当前建议是：

- 普通点击：走 `SendEventController.click()`
- 连续轨迹或多指：走 `MultiTouchController`

这样单击稳定性和多指扩展性可以先同时保住。
