# Auto_PUBG_ALL 本地 SAM3

该目录提供 `Auto_PUBG_ALL` 项目专属的两套 EfficientSAM3 本地推理实现。
模型与自动化逻辑运行在同一个 Python 进程中，不使用 host、port、ZMQ 或
独立服务。

- `version=0`（默认）：TV-M，TinyViT 11m + MobileCLIP-S0。
- `version=1`：EV-M，EfficientViT B1 + MobileCLIP-S0。

未配置 `version` 时始终按 `0` 处理，保持原有行为不变。

## 部署

1. 使用 Python 3.10 或更高版本的环境启动整个 `auto_game` 工程；
   当前已按 Python 3.10.20 服务器环境放宽版本限制。
2. 安装本目录依赖：

   ```bash
   python -m pip install -r aw/autogame/customs_examples/Auto_PUBG_ALL/resource/perception/sam3_tiny/requirements.txt
   ```

3. 将所需权重放到：

   ```text
   aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/sam3_tiny/efficientsam3_tinyvit.pt
   aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/sam3_tiny/efficientsam3_efficientvit.pt
   ```

4. 在 Label 中框选一个“特殊区域”，名称填写 `sam3`，并在该
   special_area 配置中填写 `seg_name`，并可填写 `version`。运行到对应
   阶段后，后台会裁剪该区域，并将 `seg_name` 作为文本提示词传给所选
   本地模型。例如：

   ```python
   "sam3": {
       "rect": [0.0, 0.0, 1.0, 1.0],
       "seg_name": "building",
       "version": 1,
   }
   ```

两套实现的文本编码器均为 MobileCLIP-S0、context length 16。未配置
`seg_name` 时，默认文本提示词为 `building`。如需部署时覆盖，可设置：

- `AUTOGAME_SAM3_CHECKPOINT`
- `AUTOGAME_SAM3_V1_CHECKPOINT`
- `AUTOGAME_SAM3_PROMPT`
- `AUTOGAME_SAM3_DEVICE`
- `AUTOGAME_SAM3_CONFIDENCE_THRESHOLD`
- `AUTOGAME_SAM3_MIN_MASK_AREA_RATIO`

返回值通过 `w.get_info("sam3")` 获取，其中 `found` 表示是否找到目标；
`bbox_xyxy_local` 是相对于 Label 所框 SAM3 区域的局部坐标。

`runtime/` 中的第三方 SAM3 源码按其随附 `LICENSE` 分发。
