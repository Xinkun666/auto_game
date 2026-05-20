# Remote Brain

这个目录是 `auto_game` 的远程大脑雏形：

- 本地：只负责抓图、上传帧、执行服务器返回的动作。
- 服务器：负责图像识别、阶段状态、业务决策，返回 `click/tap_single/change_stage` 等动作。

## 目录

```text
remote_brain/
  protocol.py   # 图像 base64 编解码和 JSON 安全转换
  server.py     # 内网服务器启动入口
  client.py     # 本地上传帧、接收动作、执行动作的客户端
```

## 服务器启动

服务器上需要有完整 `auto_game` 仓库，而不是只有 `remote_brain` 目录，因为服务端会复用：

- `aw/autogame/tools/GameSceneHandler.py`
- `aw/autogame/customs_game_examples/Auto_PUBG_ALL/auto_pubg.py`
- `aw/autogame/customs_examples/Auto_PUBG_ALL/resource/*`
- 所有模型权重文件

启动：

```bash
cd /path/to/auto_game
python3 -m remote_brain.server --host 0.0.0.0 --port 8765
```

健康检查：

```bash
curl http://服务器IP:8765/health
```

返回：

```json
{"ok":true,"message":"remote brain alive"}
```

## 本地客户端用法

### 通过 launcher 启动

本地启动器现在有两种后端：

- `本地`：默认模式，保持原来的本机识别和决策。
- `服务端`：本地只抓图和执行动作，识别与逻辑在服务器运行。

使用服务端模式：

1. 先在服务器启动服务：

```bash
cd /path/to/auto_game
python3 -m remote_brain.server --host 0.0.0.0 --port 8765
```

2. 本地启动 launcher：

```bash
python3 launcher.py
```

3. 在 `运行后端` 里选择 `服务端`，填写：

```text
http://服务器IP:8765
```

4. 其他 project_case / target_case / testcases 选择方式不变，点击启动即可。

launcher 会把下面两个环境变量传给子进程：

```bash
AUTOGAME_RUN_BACKEND=remote
AUTOGAME_REMOTE_BRAIN_URL=http://服务器IP:8765
```

如果选择 `本地`，则不会走上传帧逻辑。

先单独验证能否创建会话：

```python
from remote_brain.client import RemoteBrainClient

client = RemoteBrainClient("http://服务器IP:8765")
client.start_session(screen=(2832, 1316), current_stage="关闭弹窗阶段")
```

上传一帧：

```python
response = client.tick(
    frame_rgb=frame,
    current_stage="跑图阶段",
    frame_id=1,
    screen=(2832, 1316),
)
print(response["actions"])
```

执行动作：

```python
from remote_brain.client import execute_actions

execute_actions(worker, response["actions"])
```

## FrameWorker 接入

远程大脑已经接入到本地核心循环：

```text
aw/autogame/tools/GameFrameWorker.py
```

本地模式仍然执行：

```python
self.stage_info = self.stage_resolver.process_frame(self.frame, self.current_stage)
self.on_stage_logic(self)
```

服务端模式执行：

```python
response = self.remote_brain.tick_worker(self, execute=True)
```

也可以不经过 launcher，直接用环境变量切到服务端：

```bash
export AUTOGAME_RUN_BACKEND=remote
export AUTOGAME_REMOTE_BRAIN_URL=http://服务器IP:8765
```

## 注意

1. 服务端第一次收到 `/session/start` 时会加载模型，可能会慢。
2. 当前服务端按单项目设计，默认 `Auto_PUBG_ALL/auto_pubg`。
3. `refresh_frame` 在远程模式里是 no-op。服务器无法直接抓下一帧，所以本地会等下一次 tick 再上传新帧。
4. 如果服务器没有 HDC，没关系。服务端会用客户端传入的 `screen.width/screen.height` patch 掉分辨率读取。
5. 图片默认用 JPG 85 压缩。若模板匹配不稳定，可把客户端 `image_format="png"`。
