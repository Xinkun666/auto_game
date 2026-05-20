# SAM3 Standalone Server

This directory can run the SAM3 segmentation server without importing the rest of
`control_proxy`. It only needs this `sam3/` directory and the sibling
`sam3_mps_cpu/` patched SAM3 source tree.

Use this directory when the proxy runtime or room-library tooling needs the
remote `sam3` backend. `gametest_proxy.main` does not start this server
automatically; the current default replay exploration path expects it to be
running separately.

If you are already working inside this monorepo and have run
`cd control_proxy && poetry install --extras pubg`, the supported repository
entry is:

```bash
cd control_proxy/src
poetry run python -m gametest_proxy.pubg_room_explore.sam3.server
```

The rest of this README focuses on the standalone `sam3/` deployment path.

By default, the repository module entry resolves the checkpoint at
`pubg_room_explore/models/sam3.pt` relative to this package's parent directory.
If you deploy these files elsewhere, pass `--checkpoint_path` explicitly.

## Install

From this directory:

```bash
python -m pip install -r requirements-server.txt
```

`requirements-server.txt` installs:

- this server package in editable mode
- `../sam3_mps_cpu` in editable mode
- runtime dependencies declared in `pyproject.toml`

It does not download model weights. The checkpoint file is a separate runtime
asset.

For platform-specific PyTorch builds, install the correct `torch` and
`torchvision` wheels first, then run the command above.

## Run

Fake mode does not load model weights:

```bash
python server.py --mode fake --host 0.0.0.0 --port <port>
```

SAM3 mode loads the local patched SAM3 package in this process:

```bash
python server.py
```

After installation, the console script is also available:

```bash
gametest-sam3-server
```

## Backend Contract

`Sam3Segmenter` supports three backends:

- `sam3`: remote client mode. The current process sends requests to a SAM3 server.
- `sam3_local`: local in-process mode. The current process loads and runs SAM3.
- `fake`: returns the input image and an all-ones mask.

The server itself uses `sam3_local` internally when launched with `--mode sam3`.
Client code uses remote inference by default with `backend="sam3"`.
Local inference uses `device="auto"` by default, which selects CUDA, then MPS, then CPU.
