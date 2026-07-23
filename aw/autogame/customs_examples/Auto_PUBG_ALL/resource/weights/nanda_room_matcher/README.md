# 南大房型匹配权重

本目录只保存部署约定，小文件配置进入 Git，大权重由服务器单独放置。

必须形成以下结构：

```text
nanda_room_matcher/
├── dinov3_vitl16/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors
└── rgb_mlp_struct_v7.pkl
```

- `model.safetensors` 必须是真实 DINOv3 ViT-L/16 权重，不能是 Git LFS 指针。
- `rgb_mlp_struct_v7.pkl` 使用南大最新 demo 的同名文件。
- 匹配运行时只做本地离线加载，不会从 Hugging Face 或 ModelScope 自动下载权重。

房型模板与回放不放在这里，放到同级资源目录的
`nanda_room_library/rooms/<room_id>/`。

把房型匹配依赖安装到实际运行 auto_game 的同一个环境：

```bash
python -m pip install -r requirements_nanda_room_matcher.txt
```

不需要启动任何 HTTP 服务。人物完成门前对准后，搜房进程会按需取得
`building`、`door frame`、`window` 三类 `sam3_tiny` 分割结果，并在当前进程
完成 DINOv3 + MLP 匹配。DINOv3、MLP 和房型索引只加载一次，之后持续复用。
