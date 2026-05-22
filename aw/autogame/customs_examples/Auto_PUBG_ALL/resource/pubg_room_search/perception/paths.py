from __future__ import annotations

import os
from pathlib import Path


PERCEPTION_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PERCEPTION_DIR.parents[1]
MODELS_DIR = PERCEPTION_DIR / "models"
YOLO_WEIGHTS_DIR = MODELS_DIR / "yolo"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def yolo_detect_model_path() -> Path:
    return _env_path("AUTOGAME_PUBG_YOLO_DETECT_MODEL", YOLO_WEIGHTS_DIR / "26x_det.pt")


def yolo_classify_model_path() -> Path:
    return _env_path("AUTOGAME_PUBG_YOLO_CLASSIFY_MODEL", YOLO_WEIGHTS_DIR / "26x_cls.pt")


def looks_like_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        return path.read_text(encoding="utf-8", errors="ignore").startswith(
            "version https://git-lfs.github.com/spec/v1"
        )
    except OSError:
        return False


def require_real_model_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} 模型不存在: {path}")
    if looks_like_lfs_pointer(path):
        raise FileNotFoundError(
            f"{label} 模型仍是 Git LFS 指针，不是真实权重文件: {path}"
        )
    return path
