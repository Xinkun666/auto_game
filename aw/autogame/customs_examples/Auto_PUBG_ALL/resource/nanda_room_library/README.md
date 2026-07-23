# 南大房型库部署目录

将南大最新 demo 的
`control_proxy/src/gametest_proxy/pubg_room_explore/room_library/rooms`
完整复制到本目录，最终结构必须是：

```text
nanda_room_library/
└── rooms/
    └── <room_id>/
        ├── metadata.json
        ├── actions/action_step.json
        ├── captures/
        └── templates/<template_id>/
            ├── segment.png
            └── mask.png
```

服务首次启动会计算每个模板的彩色和灰度 DINOv3 特征，并将缓存写入
`rooms/<room_id>/derived/autogame_embeddings/`。自动化启动前会检查全部有效模板，
缺少门窗结构缓存时用进程内的 SAM3 提取模板的 `door frame` 和 `window`，
并写入 `rooms/<room_id>/derived/autogame_structure/`。全部模板处理成功后才会
启动自动化；后续匹配和进程重启都会直接复用缓存。

游戏画面的 `building`、`door frame`、`window` 都通过搜房阶段的 `sam3`
special_area 按需获取，不需要启动 HTTP 匹配服务。`other` 仍是搜房阶段默认分组。

运行时只索引 `status != disabled`、存在 `action_step.json` 且至少有一个
有效模板的房型。匹配成功后仍会检查 `metadata.json` 中的
`replay.allow_actions`，为 `false` 时不会返回回放动作。
