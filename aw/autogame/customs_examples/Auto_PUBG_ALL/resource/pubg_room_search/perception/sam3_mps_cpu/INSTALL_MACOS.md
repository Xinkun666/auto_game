# macOS Apple Silicon 安装说明

这个目录是 SAM3 的本地 macOS/MPS 补丁版，用于在 `game_test` Conda 环境中做图片推理。它基于 SAM3 issue/PR 里社区提供的 macOS 兼容方案，并在本地补了几处修改：

- 非 CUDA 平台下避免强依赖 Triton
- 自动选择 MPS/CPU，减少硬编码 `cuda` 路径
- 放宽 NumPy 2.x 和新版 `ftfy` 的依赖元数据
- 用 `importlib.resources` 替代 `pkg_resources`
- `Sam3Processor` 默认按 `CUDA -> MPS -> CPU` 选择设备

当前这套安装方式主要面向 Apple Silicon 上的图片推理。视频和 tracking 路径仍然不是这套补丁的主要目标。

## 目录结构

从仓库根目录看，相关路径是：

```bash
control_proxy/src/gametest_proxy/pubg_room_explore/sam3_mps_cpu
control_proxy/src/gametest_proxy/pubg_room_explore/models/sam3.pt
```

模型权重默认放在：

```bash
control_proxy/src/gametest_proxy/pubg_room_explore/models/sam3.pt
```

## 安装

先激活项目环境：

```bash
conda activate game_test
```

在仓库根目录执行本地安装：

```bash
python -m pip install --force-reinstall --no-deps \
  ./control_proxy/src/gametest_proxy/pubg_room_explore/sam3_mps_cpu
```

这里故意使用 `--no-deps`。如果让 pip 自动解析上游依赖，它可能会继续尝试安装不适合 macOS arm64 的包，尤其是 `decord`。当前 `game_test` 环境已经包含本项目需要的主要运行依赖。

如果是在一个全新的环境里安装，先准备这些核心依赖：

```bash
python -m pip install \
  torch torchvision \
  numpy "ftfy>=6.1.1,<6.4" regex iopath timm pycocotools \
  opencv-python scipy pillow huggingface-hub
```

除非明确需要测试视频读取，否则不要安装 `eva-decord`。它可能会和本仓库已有的 PyAV/FFmpeg 库产生重复加载警告。

## 验证 MPS

请在普通终端里验证 MPS。某些沙箱环境会把 Apple Silicon 上的 MPS 误报为不可用。

```bash
conda run -n game_test python -c 'import torch; print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available()); print(torch.ones(1, device="mps"))'
```

正常情况下会看到类似输出：

```text
True
True
tensor([1.], device='mps:0')
```

## Smoke Test

在仓库根目录执行：

```bash
conda run -n game_test python \
  control_proxy/src/gametest_proxy/pubg_room_explore/sam3_mps_cpu/scripts/smoke_macos.py \
  --device mps --skip-forward
```

正常结果应包含：

```text
Smoke test PASSED
Model is on device: mps:0
```

## 加载本地权重

使用本地 `sam3.pt`，不要从 Hugging Face 下载：

```bash
conda run -n game_test python -c '
from sam3.model_builder import build_sam3_image_model

ckpt = "control_proxy/src/gametest_proxy/pubg_room_explore/models/sam3.pt"
model = build_sam3_image_model(
    device="mps",
    load_from_HF=False,
    checkpoint_path=ckpt,
    enable_inst_interactivity=False,
    compile=False,
)
print(type(model), next(model.parameters()).device)
'
```

正常结果应类似：

```text
<class 'sam3.model.sam3_image.Sam3Image'> mps:0
```

## 最小图片推理验证

推理前建议设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`，让 MPS 暂不支持的算子自动回退到 CPU。

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n game_test python -c '
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ckpt = "control_proxy/src/gametest_proxy/pubg_room_explore/models/sam3.pt"
model = build_sam3_image_model(
    device="mps",
    load_from_HF=False,
    checkpoint_path=ckpt,
    enable_inst_interactivity=False,
    compile=False,
)
processor = Sam3Processor(model, device="mps")

image = Image.new("RGB", (512, 512), "white")
draw = ImageDraw.Draw(image)
draw.rectangle((120, 120, 390, 390), fill="gray")

state = processor.set_image(image)
state = processor.set_text_prompt("gray square", state)

print("boxes", tuple(state["boxes"].shape))
print("masks", tuple(state["masks"].shape))
print("scores", tuple(state["scores"].shape))
print("device", processor.device)
'
```

正常结果应类似：

```text
boxes (1, 4)
masks (1, 1, 512, 512)
scores (1,)
device mps
```

## 注意事项

- `sam3` 可能仍显示版本号 `0.1.0`，这是上游分支的版本元数据，不代表它是 PyPI 上那个空包。
- PyPI 的 `sam3==0.1.3` 在当前 Mac 路径上仍会遇到 `decord` 和 `triton` 相关问题。
- 这个目录是本地 macOS 补丁的来源。修改补丁后，需要重新从本目录执行安装命令。
