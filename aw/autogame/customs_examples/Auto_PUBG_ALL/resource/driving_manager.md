# DrivingManager 说明

本文档用于快速理解 `driving_manager.py` 中 `DrivingManager` 的结构、状态流转和调试入口。

## 1. 这个类负责什么

`DrivingManager` 负责整个开车阶段的行为控制，核心目标是：

1. 车辆出库。
2. 找到最近道路点并低速并入道路。
3. 沿道路持续行驶。
4. 根据 `white_angle` 和游戏时间估算圈中心，并把路径重规划到离圈最近的道路点。
5. 在开车过程中处理卡住、打转、避障、降速等情况。

可以把它理解成一个“开车阶段状态机 + 路径跟随器”。

## 2. 总体流程

`process(w)` 现在是一个“内部阶段机循环”。

它不是单纯按顺序一路 `return`，而是每帧按下面的模式运行：

1. 读取当前位置、车头朝向、速度、`white_angle`。
2. 处理死亡、卡住、打转、进圈重规划等全局状态。
3. 根据 `self.current_stage` 选择当前要执行的内部阶段。
4. 如果当前阶段只是完成了“状态切换”而没有真正执行动作，那么同一帧会继续进入下一个阶段。
5. 一旦某个阶段真正执行了点击/按下/双指动作，本轮结束，统一 `w.refresh_frame()`。

当前内部阶段有：

- `出库阶段`
- `规划道路阶段`
- `进路阶段`
- `道路对齐阶段`
- `巡航阶段`
- `避障阶段`
- `脱困阶段`

## 3. 你要重点记住的几个状态变量

### 出库相关

- `is_first_car`
  - `True`：还处于固定出库阶段。
  - `False`：跳过出库，直接进入后续道路逻辑。

### 并路相关

- `on_road`
  - `False`：还没正式并入道路。
  - `True`：已经完成道路点贴合和方向对齐，可以开始正式跑路。

- `road_entry_point`
  - 当前准备并入的最近道路点。

- `current_stage`
  - DrivingManager 自己的内部阶段状态。
  - `process(w)` 每次都会按这个阶段来调度逻辑。

### 路径相关

- `road_list`
  - 当前要跟随的路径点列表。
  - 并路阶段可能是一条去最近道路点的短路径。
  - 正式跑图阶段通常是一段环路路径或道路拓扑路径。

### 圈信息相关

- `stable_circle_angle`
  - 多帧稳定后的 `white_angle`。

- `circle_center`
  - 根据 `stable_circle_angle + 游戏时间` 估算出的圈中心。

- `circle_target_point`
  - 当前道路路径中，离圈中心最近的道路点。

### 行驶控制相关

- `auto_forward_enabled`
  - 当前是否认为“自动前进”已经打开。

- `stuck`
  - 短时卡死。

- `trapped`
  - 长时间在小范围打转。

## 4. 方法分组理解

### A. 主循环入口

- `process(w)`
  - 整个 DrivingManager 的总调度入口。
  - 内部会循环执行阶段切换，直到某个阶段真正做了动作，或者本轮没有更多阶段可推进。

### B. 阶段机逻辑

- `_sync_current_stage(...)`
  - 根据 `is_first_car`、`stuck`、`on_road`、`避障状态` 等全局条件，修正当前内部阶段。

- `_run_current_stage(...)`
  - 按 `self.current_stage` 分发到具体阶段逻辑。

- `_run_stage_exit_garage(...)`
- `_run_stage_plan_route(...)`
- `_run_stage_enter_road(...)`
- `_run_stage_align_road(...)`
- `_run_stage_cruise(...)`
- `_run_stage_avoidance(...)`
- `_run_stage_unstuck(...)`
  - 分别对应内部阶段。

### C. 出库逻辑

- `_handle_first_car_exit(w, direction)`
  - 固定规则出库。
  - 完成后只切到 `规划道路阶段`，不直接开始正式巡航。

### D. 并入道路逻辑

- `_run_stage_enter_road(w, context)`
  - 出库后第一步做的事情。
  - 找最近道路点，慢慢把车开过去。
  - 到达道路点附近后，切到 `道路对齐阶段`，不直接跑。

- `_run_stage_align_road(w, context)`
  - 站在道路点附近，先把车头调到道路方向。
  - 对齐完成后，切回 `规划道路阶段` 生成正式巡航路径。

- `_approach_target_slowly(...)`
  - 并路阶段的低速接近方法。
  - 这里会关闭自动前进，改成短促前进和小幅转向，避免冲过头。

- `_get_road_direction(point)`
  - 计算当前道路点附近道路的方向。
  - 用于“车到路边后，先把车头和道路方向对齐”。

### E. 路径规划逻辑

- `_plan_or_refresh_route(location)`
  - `规划道路阶段` 的统一入口。

- `_plan_route_to_nearest_road(location)`
  - 找最近道路点，生成一条接入道路的路径。

- `_plan_route_to_point(location, target_point)`
  - 用 `MapNavigator.plan_path()` 规划到指定目标点。

- `_plan_ring_route(location, target_point=None)`
  - 已经上路后，生成正式跑路路径。
  - 有圈目标时，规划到离圈最近的道路点。
  - 没有圈目标时，走默认环路。

- `_get_topology_path(location)`
  - 如果道路拓扑资源完整，则优先使用 `road_topology.py` 的拓扑路径。

### F. 圈逻辑

- `_update_circle_angle(angle)`
  - 对 `white_angle` 做多帧稳定化。

- `_should_replan_circle()`
  - 判断当前是否需要因为圈信息变化而重规划。

- `_estimate_circle_center(location)`
  - 根据当前位置、`stable_circle_angle` 和阶段距离，估算圈中心。

- `_replan_for_circle(location)`
  - 估算圈中心后，找到离圈最近的道路点，并把内部阶段切回 `规划道路阶段`。

### G. 正式道路跟随逻辑

- `_run_stage_cruise(w, context)`
  - 巡航阶段入口。
  - 先判断是否需要避障或重规划，再进入真正的道路跟随。

- `_follow_road(w, location, direction, speed)`
  - 正式道路跟随。
  - 会裁剪已走过的路径点、生成前方锚点、控制自动前进、做动态转向。

- `_trim_passed_waypoints(location)`
  - 清理已经经过的路径点。

- `_get_anchor_point(location, speed)`
  - 在前方一定距离生成动态锚点。
  - 车不是盯着当前最近点跑，而是盯着前方锚点跑。

- `_steer_towards_anchor(...)`
  - 根据锚点相对车头的偏左/偏右情况打方向。

- `_get_lookahead_distance(speed)`
  - 根据速度档位决定前视距离。

### H. 速度与自动前进

- `_maintain_auto_forward(w, speed)`
  - 管理“自动前进”状态。
  - 当 `speed == 3` 时，会主动取消自动前进，让车速降下来。

- `_cancel_auto_forward(w, reason=None)`
  - 关闭自动前进。

### I. 异常处理

- `_check_if_stuck(location)`
  - 检测短时卡死。

- `_check_if_trapped(location)`
  - 检测长时间打转。

- `_perform_unstuck_action(...)`
  - 车辆卡住时执行脱困动作。

- `_rid_forbidden(...)`
  - 前方不可通行时执行避障。

- `_handle_death(w)`
  - 检测到死亡后清状态并切阶段。

## 5. 调试时怎么快速下手

### 场景 1：只想调路上表现，不想跑出库

直接把：

```python
self.is_first_car = False
```

这样 `process()` 会跳过 `_handle_first_car_exit()`，直接进入“并路/道路巡航”逻辑。

### 场景 2：只想调“从车库贴到路边”这一步

重点看这几个方法：

- `_run_stage_enter_road`
- `_run_stage_align_road`
- `_approach_target_slowly`
- `_get_road_direction`

### 场景 3：只想调路上打方向

重点看：

- `_follow_road`
- `_get_anchor_point`
- `_steer_towards_anchor`
- `_get_steer_wait`

### 场景 4：只想调进圈重规划

重点看：

- `_update_circle_angle`
- `_should_replan_circle`
- `_estimate_circle_center`
- `_replan_for_circle`

## 6. 一眼判断当前卡在哪个阶段

看日志关键词即可：

- `按固定规则将车开出车库`
  - 说明还在出库。

- `当前内部阶段: 进路阶段`
  - 说明还没正式上路，正在接近道路点。

- `当前内部阶段: 道路对齐阶段`
  - 说明已经贴到道路点，但还在调车头角度。

- `阶段切换: 道路对齐阶段 -> 规划道路阶段`
  - 说明并路完成，准备生成正式巡航路径。

- `当前内部阶段: 巡航阶段`
  - 说明已经进入正式道路跟随。

- `动态修方向`
  - 说明当前正在按锚点做方向修正。

- `识别到进圈角度`
  - 说明 `white_angle` 已生效，并触发了进圈逻辑。

## 7. 推荐阅读顺序

如果你要快速读代码，建议按下面顺序看：

1. `process`
2. `_handle_first_car_exit`
3. `_approach_road_entry`
4. `_plan_route_to_nearest_road`
5. `_plan_ring_route`
6. `_follow_road`
7. `_steer_towards_anchor`
8. `_replan_for_circle`

这样最容易先建立整体结构，再看局部策略。
