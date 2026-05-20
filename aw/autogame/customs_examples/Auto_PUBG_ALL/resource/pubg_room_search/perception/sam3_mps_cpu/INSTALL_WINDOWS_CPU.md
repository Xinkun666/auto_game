# Windows CPU 安装说明

这份文档说明如何在 Windows 上尝试使用当前本地补丁版 SAM3 跑图片推理的 CPU 路径。

注意：Windows CPU 路径目前属于实验性用法。本仓库已在 macOS Apple Silicon 的 MPS 路径上做过验证，但没有在 Windows 机器上实际验证。下面步骤的目标是绕开上游 `triton`、`decord` 和硬编码 `cuda` 的问题，让 image inference 走 CPU。

## 适用范围

适用：

- 图片推理
- `build_sam3_image_model(...)`
- `Sam3Processor(...)`
- `processor.set_image(...)`
- `processor.set_text_prompt(...)`

不作为主要目标：

- 视频推理
- tracking
- 训练
- Triton/perflib 相关路径

CPU 能跑不代表速度理想。SAM3 权重约 3.2G，默认输入分辨率较高，CPU 推理会比较慢。

## 目录结构

从仓库根目录看，相关路径是：

```powershell
control_proxy\src\gametest_proxy\pubg_room_explore\sam3_mps_cpu
control_proxy\src\gametest_proxy\pubg_room_explore\models\sam3.pt
```

模型权重默认放在：

```powershell
control_proxy\src\gametest_proxy\pubg_room_explore\models\sam3.pt
```

## 安装

先激活项目环境：

```powershell
conda activate game_test
```

安装 PyTorch CPU 版本：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

安装图片推理所需依赖：

```powershell
python -m pip install numpy "ftfy>=6.1.1,<6.4" regex iopath timm pycocotools opencv-python scipy pillow huggingface-hub
```

从仓库根目录安装本地补丁版 SAM3：

```powershell
python -m pip install --force-reinstall --no-deps .\control_proxy\src\gametest_proxy\pubg_room_explore\sam3_mps_cpu
```

这里故意使用 `--no-deps`。如果让 pip 自动解析上游依赖，它可能会继续尝试安装不适合当前平台或不需要的包，尤其是 `decord` 和 `triton`。CPU 图片推理不需要这两个包。

## 验证 CPU

```powershell
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.ones(1, device='cpu'))"
```

没有 NVIDIA CUDA 的 Windows 机器上，正常结果应类似：

```text
cuda False
tensor([1.])
```

## 加载本地权重

使用本地 `sam3.pt`，不要从 Hugging Face 下载：

```powershell
python -c "from sam3.model_builder import build_sam3_image_model; ckpt='control_proxy/src/gametest_proxy/pubg_room_explore/models/sam3.pt'; model=build_sam3_image_model(device='cpu', load_from_HF=False, checkpoint_path=ckpt, enable_inst_interactivity=False, compile=False); print(type(model), next(model.parameters()).device)"
```

正常结果应类似：

```text
<class 'sam3.model.sam3_image.Sam3Image'> cpu
```

## 最小图片推理验证

```powershell
python -c "from PIL import Image, ImageDraw; from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor; ckpt='control_proxy/src/gametest_proxy/pubg_room_explore/models/sam3.pt'; model=build_sam3_image_model(device='cpu', load_from_HF=False, checkpoint_path=ckpt, enable_inst_interactivity=False, compile=False); processor=Sam3Processor(model, device='cpu'); image=Image.new('RGB', (512, 512), 'white'); draw=ImageDraw.Draw(image); draw.rectangle((120, 120, 390, 390), fill='gray'); state=processor.set_image(image); state=processor.set_text_prompt('gray square', state); print('boxes', tuple(state['boxes'].shape)); print('masks', tuple(state['masks'].shape)); print('scores', tuple(state['scores'].shape)); print('device', processor.device)"
```

正常结果应类似：

```text
boxes (1, 4)
masks (1, 1, 512, 512)
scores (1,)
device cpu
```

## 注意事项

- `sam3` 可能仍显示版本号 `0.1.0`，这是上游分支的版本元数据，不代表它是 PyPI 上那个空包。
- 不建议直接安装 PyPI 的 `sam3==0.1.3` 来跑 Windows CPU。它仍然有 `decord`、`triton` 和硬编码 `cuda` 相关问题。
- Windows CPU 不需要 `PYTORCH_ENABLE_MPS_FALLBACK=1`，那是 macOS MPS 专用环境变量。
- 如果安装 `pycocotools` 失败，可以先确认 Python 版本和 wheel 支持情况，或改用 Conda/预编译 wheel 来源。
- 这个目录是本地补丁的来源。修改补丁后，需要重新从本目录执行安装命令。
