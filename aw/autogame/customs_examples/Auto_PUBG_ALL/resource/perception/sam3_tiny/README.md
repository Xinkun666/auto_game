# Auto_PUBG_ALL 本地 SAM3

该目录提供 `Auto_PUBG_ALL` 项目专属的 EfficientSAM3 本地推理实现。模型与
自动化逻辑运行在同一个 Python 进程中，不使用 host、port、ZMQ 或独立服务。

## 部署

1. 使用 Python 3.10 或更高版本的环境启动整个 `auto_game` 工程；
   当前已按 Python 3.10.20 服务器环境放宽版本限制。
2. 安装本目录依赖：

   ```bash
   python -m pip install -r aw/autogame/customs_examples/Auto_PUBG_ALL/resource/perception/sam3_tiny/requirements.txt
   ```

3. 将 TinyViT 权重放到：

   ```text
   aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/sam3_tiny/efficientsam3_tinyvit.pt
   ```

4. 在 Label 中框选一个“特殊区域”，名称填写 `sam3`。运行到对应阶段后，后台会
   裁剪该区域并直接调用本地模型。

默认模型参数为 TinyViT 11m、MobileCLIP-S0、context length 16，默认文本提示词
为 `building`。如需部署时覆盖，可设置：

- `AUTOGAME_SAM3_CHECKPOINT`
- `AUTOGAME_SAM3_PROMPT`
- `AUTOGAME_SAM3_DEVICE`
- `AUTOGAME_SAM3_CONFIDENCE_THRESHOLD`
- `AUTOGAME_SAM3_MIN_MASK_AREA_RATIO`

返回值通过 `w.get_info("sam3")` 获取，其中 `found` 表示是否找到目标；
`bbox_xyxy_local` 是相对于 Label 所框 SAM3 区域的局部坐标。

`runtime/` 中的第三方 SAM3 源码按其随附 `LICENSE` 分发。
