# 房子外面到进屋业务逻辑追踪笔记

主题：和平精英 R 城搜房，从室外移动到确认进入房屋。

入口与主链路：
- `aw/autogame/customs_game_examples/Auto_PUBG_ALL/auto_pubg.py:470`：只有当前阶段是 `搜房阶段` 时，外层调度才启动搜房处理。
- `aw/autogame/customs_game_examples/Auto_PUBG_ALL/auto_pubg.py:471`：先处理 SP 录制启动。
- `aw/autogame/customs_game_examples/Auto_PUBG_ALL/auto_pubg.py:472`：检查死亡、排名、搜房计时完成等中断条件。
- `aw/autogame/customs_game_examples/Auto_PUBG_ALL/auto_pubg.py:476`：每帧调用 `searching_house_manager.process(w)`。
- `aw/autogame/customs_examples/Auto_PUBG_ALL/resource/control/house_search_manager.py:206`：父类 `process()` 做首帧第一人称、位置、方向准备。
- `aw/autogame/customs_examples/Auto_PUBG_ALL/resource/control/house_scene_search_manager.py:248`：子类 `searching_logic()` 基于 `house_scene` 和 R 城目标状态机推进。

计划步骤：
1. 搜房阶段入口与基础输入准备：启动 SP、确认不中断、切第一人称、读取稳定位置与方向。
2. 读取 `house_scene` 并处理特殊状态：已在室内、水中、落地位置未稳定等提前分支。
3. 判断是否靠近 R 城：远离 R 城时先规划到 R 城附近，未到搜单栋房屋阶段。
4. 选择并锁定下一栋 R 城房屋：从 R 城配置目标中按距离、朝向惩罚、失败次数打分。
5. 快速导航到房屋靠近点：自动前进、对齐方向、卡住检测、前方房体绕行。
6. 进入精细导航：距离进入 30 米内后停自动前进，分段摇杆推进到进门点。
7. 触发进门流程：到达房点或 near_door/near_wall 且离房体足够近时切 `SCENE_ENTRY`。
8. 执行 R 城 scene 进门：自由找门/窗/按钮，跳跃、开门、前推，并以 `house_scene=indoor` 确认进屋。
9. 进屋成功后的交接：调用室内旋转搜房；如果失败则临时跳过当前房点。

关键状态：
- `HOUSE_INDOOR = 0`
- `HOUSE_OUTDOOR = 1`
- `HOUSE_ROOFTOP = 2`
- `HOUSE_NEAR_DOOR = 3`
- `HOUSE_NEAR_WALL = 4`
- `HOUSE_NEAR_ENTRY_SCENES = {3, 4}`
- `STATUS_SCENE_ENTRY = "SCENE_ENTRY"`

用户已确认步骤：
- 第 1 步：进入搜房阶段后，外层先启动 SP、检查中断；随后 `HouseSearchManager.process()` 首帧刷新并切第一人称，读取有效 `location` 和 `direction`，再交给 `HouseSceneSearchManager.searching_logic()`。
- 第 2 步：`searching_logic()` 先读取并规范化 `house_scene`，处理结束计时、水中、已在室内、落地坐标未稳定等提前分支；只有这些分支都没命中，才继续走室外找房路线。
- 第 3 步：如果当前位置距离最近 R 城房点超过 `r_city_near_distance`，先进入 `ROUTE_TO_R_CITY` 区域导航，清空单房目标，规划到 R 城接入点的路线，沿 waypoint 对齐方向并自动前进。
- 第 4 步：到达 R 城附近且没有当前房屋目标时，从 `r_city_targets` 里排除已完成/失败超限目标，按 `距离 + 朝向惩罚 + 失败惩罚` 选最低分目标，并写入 `current_house_id`、`current_r_city_target`、`active_entry`，状态切到 `FAST_NAV`。
- 第 5 步：`FAST_NAV` 使用 `active_entry["location"]` 作为靠近点，对齐方向并开启自动前进；途中处理卡住、绕房、贴近其他房体时改锁目标，距离进入 `ENTRY_AUTO_FORWARD_DISTANCE` 后停止自动前进并切到 `PRECISE_NAV`。

未决问题：
- 用户是否只关心 R 城新搜房进门路径，还是也要旧 `HouseSearchManager` 普通房屋进门路径；当前代码主实例是 `HouseSceneSearchManager`，优先讲新路径。

本轮用户追问范围：
- 从“距离一个房子很近”开始讲到“进房”为止，优先从已锁定 R 城目标后的 `FAST_NAV -> PRECISE_NAV -> SCENE_ENTRY -> indoor` 近距离链路讲起。

近距离链路步骤：
1. 进入精细导航：`FAST_NAV` 中距离进门靠近点 `<= ENTRY_AUTO_FORWARD_DISTANCE` 时停自动前进并切 `PRECISE_NAV`。
2. 精细导航每帧先判断是否能触发进门：到靠近点 `<= house_arrival_distance`，或 `house_scene` 是 `near_door/near_wall` 且离房体 `<= early_entry_scene_distance`。
3. 未触发进门时，先对齐进门点，再按 fast/slow 模型小步前推，最多 4 小步，并用前后距离反馈更新移动模型。
4. 触发 `SCENE_ENTRY` 后，R 城目标走自由找入口：优先处理 indoor、跳跃、开门/关门按钮、stone_wall、门、窗等机会。
5. 点击开门或识别门已开后，重新修正进门方向，执行后拉/重锁门/短前推循环，直到 `house_scene=indoor` 或次数耗尽。
6. 一旦确认 indoor，调用 `start_searching()` 做室内旋转搜房，完成后标记当前 R 城房点完成并清空当前目标。
