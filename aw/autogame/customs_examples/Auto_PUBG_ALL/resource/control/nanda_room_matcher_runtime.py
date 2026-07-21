"""auto_game 内置的南大 DINOv3 + MLP 房型匹配运行时。

该模块只消费已经由 ``sam3_tiny`` 生成的房屋立面和 building mask，
不会启动或再次调用南大 SAM3。实现保留南大最新版的核心链路：

1. DINOv3 masked-patch pooling 提取彩色/灰度房屋特征；
2. 对房型模板做向量召回并展开候选房型的全部模板；
3. 使用 ``rgb_mlp_struct_v7.pkl`` 做 MLP 重排；
4. 使用南大阈值拒绝弱匹配，再返回房型绑定的回放 DSL。

南大当前 MLP 的特征契约包含门窗结构向量。auto_game 的方案明确不再
额外调用 SAM3 提取门窗，因此这里保留相同特征维度并传入零结构向量，
同时保留南大“无强结构时 DINO 至少 0.65”的拒绝规则。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import pickle
import re
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np


LOGGER = logging.getLogger("NandaIntegratedRoomMatcher")

EMBEDDING_DIM = 1024
MIN_MASKED_PATCHES = 4
ROOM_SHORTLIST_TEMPLATE_MULTIPLIER = 4
STRUCTURE_VECTOR_MAX_ROWS = 4
STRUCTURE_VECTOR_CHANNELS = 4
STRUCTURE_VECTOR_GLOBAL_DIM = 6
STRUCTURE_VECTOR_DIM = (
    STRUCTURE_VECTOR_GLOBAL_DIM
    + STRUCTURE_VECTOR_MAX_ROWS * 3 * STRUCTURE_VECTOR_CHANNELS
)

# 与南大最新版 control_proxy/pubg_room_explore/sam3/config.py 保持一致。
FACADE_TOP_K = 16
FACADE_STRUCT_SCORE_ALPHA = 0.30
FACADE_LOW_MLP_CONFIDENCE_SCORE = 0.05
FACADE_NO_MATCH_MIN_SCORE = 0.25
FACADE_NO_MATCH_MIN_DINO_SCORE = 0.55
FACADE_NO_MATCH_STRICT_DINO_SCORE = 0.65
FACADE_NO_MATCH_STRONG_STRUCTURE_SCORE = 0.45
FACADE_NO_MATCH_MIN_MARGIN = 0.0

RUNTIME_CACHE_VERSION = 1
INPUT_PREPROCESS_VERSION = "dino_input_no_black_holes_v2"


def _resource_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class NandaMatcherAssetPaths:
    """进程内房型匹配所需的三个独立资产入口。"""

    dino_model_dir: Path
    mlp_model_path: Path
    room_library_path: Path

    @classmethod
    def auto_game_defaults(cls) -> "NandaMatcherAssetPaths":
        resource_root = _resource_root()
        weight_root = resource_root / "weights" / "nanda_room_matcher"
        return cls(
            dino_model_dir=weight_root / "dinov3_vitl16",
            mlp_model_path=weight_root / "rgb_mlp_struct_v7.pkl",
            room_library_path=resource_root / "nanda_room_library",
        )

    @classmethod
    def from_nanda_project(cls, project_root: Path) -> "NandaMatcherAssetPaths":
        package_root = (
            project_root.expanduser().resolve()
            / "control_proxy"
            / "src"
            / "gametest_proxy"
            / "pubg_room_explore"
        )
        return cls(
            dino_model_dir=(package_root / "img_similarity" / "dinov3_vitl16"),
            mlp_model_path=package_root / "models" / "rgb_mlp_struct_v7.pkl",
            room_library_path=package_root / "room_library",
        )

    def resolved(self) -> "NandaMatcherAssetPaths":
        return NandaMatcherAssetPaths(
            dino_model_dir=self.dino_model_dir.expanduser().resolve(),
            mlp_model_path=self.mlp_model_path.expanduser().resolve(),
            room_library_path=self.room_library_path.expanduser().resolve(),
        )

    def validate(self) -> None:
        paths = self.resolved()
        required_dino_files = (
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
        )
        missing = [
            str(paths.dino_model_dir / name)
            for name in required_dino_files
            if not (paths.dino_model_dir / name).is_file()
        ]
        if not paths.mlp_model_path.is_file():
            missing.append(str(paths.mlp_model_path))
        rooms_dir = paths.room_library_path / "rooms"
        if not rooms_dir.is_dir():
            missing.append(str(rooms_dir))
        if missing:
            raise FileNotFoundError("南大房型匹配缺少资产: " + ", ".join(missing))
        _ensure_real_model_file(
            paths.dino_model_dir / "model.safetensors", "DINOv3 权重",
        )
        _ensure_real_model_file(paths.mlp_model_path, "南大 MLP 权重")

    def to_jsonable(self) -> Dict[str, str]:
        paths = self.resolved()
        return {
            "dino_model_dir": str(paths.dino_model_dir),
            "mlp_model_path": str(paths.mlp_model_path),
            "room_library_path": str(paths.room_library_path),
        }


def _ensure_real_model_file(path: Path, label: str) -> None:
    with path.open("rb") as model_file:
        header = model_file.read(256)
    if path.stat().st_size < 1024 or header.startswith(
        b"version https://git-lfs.github.com/spec/"
    ):
        raise RuntimeError(f"{label}不是实际模型文件（可能仍是 Git LFS 指针）: {path}")


def _natural_key(text: str) -> List[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def _normalize_embedding(value: np.ndarray) -> np.ndarray:
    embedding = np.asarray(value, dtype="float32").reshape(-1)
    if embedding.size != EMBEDDING_DIM:
        raise ValueError(
            f"DINOv3 特征维度异常: got={embedding.size}, expected={EMBEDDING_DIM}"
        )
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("DINOv3 返回了无效的零特征")
    return np.ascontiguousarray(embedding / norm, dtype="float32")


def _image_md5(image: Optional[np.ndarray]) -> Optional[str]:
    if image is None:
        return None
    digest = hashlib.md5()
    digest.update(repr(image.shape).encode("utf-8"))
    digest.update(str(image.dtype).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _normalize_mask(
    mask: Optional[np.ndarray], shape_hw: Tuple[int, int]
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    normalized = np.asarray(mask)
    if normalized.ndim == 3:
        normalized = np.any(normalized != 0, axis=2).astype(np.uint8)
    if normalized.ndim != 2:
        raise ValueError(f"房屋 mask 维度异常: {normalized.shape}")
    if normalized.shape != shape_hw:
        normalized = cv2.resize(
            normalized.astype(np.uint8),
            (shape_hw[1], shape_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (normalized > 0).astype(np.uint8) * 255


def _fill_mask_holes(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None or not np.any(mask):
        return mask
    mask_bool = mask.astype(bool)
    inverse = (~mask_bool).astype(np.uint8)
    count, labels = cv2.connectedComponents(inverse, connectivity=8)
    if count <= 1:
        return mask
    border_labels = set(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    filled = mask_bool.copy()
    for label in range(1, count):
        if label not in border_labels:
            filled[labels == label] = True
    return filled.astype(np.uint8) * 255


def _build_masked_dino_input(
    base_image: np.ndarray, mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if base_image.ndim != 3 or base_image.shape[2] != 3:
        raise ValueError(f"DINO 输入必须是 HxWx3 BGR 图像: {base_image.shape}")
    aligned_mask = _fill_mask_holes(_normalize_mask(mask, base_image.shape[:2]))
    if aligned_mask is None:
        return base_image.copy(), None
    output = np.zeros_like(base_image)
    mask_bool = aligned_mask.astype(bool)
    output[mask_bool] = base_image[mask_bool]
    return output, aligned_mask


def _gray_bgr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class DinoV3FeatureExtractor:
    """从 auto_game 指定目录离线加载 DINOv3，不访问模型网络。"""

    def __init__(self, model_dir: Path, device: Optional[str] = None):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError(
                "DINOv3 匹配环境缺少 torch/transformers；"
                "请安装 requirements_nanda_room_matcher.txt"
            ) from exc

        self.torch = torch
        self.model_dir = model_dir.expanduser().resolve()
        self.device = self._resolve_device(device)
        LOGGER.info("加载 DINOv3：path=%s device=%s", self.model_dir, self.device)
        transformers_logging.disable_progress_bar()
        self.processor = AutoImageProcessor.from_pretrained(
            str(self.model_dir), use_fast=True, local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            str(self.model_dir), local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: Optional[str]) -> str:
        if requested:
            return str(requested)
        if self.torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(self.torch, "backends", None), "mps", None)
        if mps is not None and bool(mps.is_available()):
            return "mps"
        return "cpu"

    @staticmethod
    def _patch_mask_from_image_mask(
        mask: np.ndarray, input_height: int, input_width: int, patch_size: int,
    ) -> np.ndarray:
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.ndim == 3:
            mask_bool = np.any(mask_bool, axis=2)
        grid_h = input_height // patch_size
        grid_w = input_width // patch_size
        token_h = grid_h * patch_size
        token_w = grid_w * patch_size
        resized = cv2.resize(
            mask_bool.astype(np.uint8),
            (token_w, token_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        return (
            resized.reshape(grid_h, patch_size, grid_w, patch_size)
            .any(axis=(1, 3))
            .reshape(-1)
        )

    def extract(
        self, image_bgr: np.ndarray, mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0]
        if mask is not None:
            pixel_values = inputs["pixel_values"]
            patch_size = int(getattr(self.model.config, "patch_size", 16))
            patch_mask = self._patch_mask_from_image_mask(
                mask,
                input_height=int(pixel_values.shape[-2]),
                input_width=int(pixel_values.shape[-1]),
                patch_size=patch_size,
            )
            prefix_tokens = 1 + int(
                getattr(self.model.config, "num_register_tokens", 0)
            )
            patch_tokens = outputs.last_hidden_state[:, prefix_tokens:]
            if (
                int(patch_mask.sum()) >= MIN_MASKED_PATCHES
                and patch_tokens.shape[1] == patch_mask.shape[0]
            ):
                selected = self.torch.from_numpy(patch_mask).to(
                    device=patch_tokens.device, dtype=self.torch.bool,
                )
                embedding = patch_tokens[:, selected, :].mean(dim=1)
        return _normalize_embedding(embedding.detach().cpu().numpy())


@dataclass(frozen=True)
class RoomMatchFeatureSpec:
    feature_set: str = "rgb"
    feature_mode: str = "pair_12d"

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "RoomMatchFeatureSpec":
        spec = cls(
            feature_set=str(metadata.get("feature_set", "rgb")),
            feature_mode=str(metadata.get("feature_mode", "pair_12d")),
        )
        if spec.feature_set not in {"rgb", "gray", "rgb_gray"}:
            raise ValueError(f"不支持的 MLP feature_set: {spec.feature_set}")
        if spec.feature_mode not in {
            "pair_12d",
            "embedding_concat",
            "embedding_concat_structure",
        }:
            raise ValueError(f"不支持的 MLP feature_mode: {spec.feature_mode}")
        return spec

    @property
    def uses_gray_embedding(self) -> bool:
        return self.feature_set in {"gray", "rgb_gray"}

    @property
    def uses_structure(self) -> bool:
        return self.feature_mode == "embedding_concat_structure"

    def expected_n_features(self) -> int:
        if self.feature_mode == "pair_12d":
            return 12 if self.feature_set == "rgb_gray" else 6
        embedding_branches = 4 if self.feature_set == "rgb_gray" else 2
        result = embedding_branches * EMBEDDING_DIM
        if self.uses_structure:
            result += STRUCTURE_VECTOR_DIM * 3
        return result


def _pair_geometry_features(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = _normalize_embedding(first)
    second = _normalize_embedding(second)
    diff = first - second
    product = first * second
    return np.asarray(
        [
            float(np.dot(first, second)),
            float(np.linalg.norm(diff)),
            float(np.mean(np.abs(diff))),
            float(np.max(np.abs(diff))),
            float(np.mean(product)),
            float(np.std(product)),
        ],
        dtype="float32",
    )


def _build_mlp_features(
    spec: RoomMatchFeatureSpec,
    query_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    query_gray: Optional[np.ndarray],
    candidate_gray: Optional[np.ndarray],
) -> np.ndarray:
    if spec.feature_set == "gray":
        if query_gray is None or candidate_gray is None:
            raise ValueError("MLP 需要灰度 DINO 特征，但灰度特征不可用")
        base_query, base_candidate = query_gray, candidate_gray
    else:
        base_query, base_candidate = query_rgb, candidate_rgb

    if spec.feature_mode == "pair_12d":
        features = [_pair_geometry_features(base_query, base_candidate)]
        if spec.feature_set == "rgb_gray":
            if query_gray is None or candidate_gray is None:
                raise ValueError("rgb_gray MLP 缺少灰度 DINO 特征")
            features.append(_pair_geometry_features(query_gray, candidate_gray))
        return np.concatenate(features).astype("float32")

    parts = [
        _normalize_embedding(base_query),
        _normalize_embedding(base_candidate),
    ]
    if spec.feature_set == "rgb_gray":
        if query_gray is None or candidate_gray is None:
            raise ValueError("rgb_gray MLP 缺少灰度 DINO 特征")
        parts.extend(
            [_normalize_embedding(query_gray), _normalize_embedding(candidate_gray)]
        )
    if spec.uses_structure:
        # 不再额外调用 SAM3 门窗结构；与南大 None 结构的向量表达完全一致。
        zero_structure = np.zeros((STRUCTURE_VECTOR_DIM,), dtype="float32")
        parts.extend([zero_structure, zero_structure, zero_structure])
    return np.concatenate(parts).astype("float32")


class MlpRoomReranker:
    def __init__(self, model_path: Path):
        self.model_path = model_path.expanduser().resolve()
        try:
            with self.model_path.open("rb") as model_file:
                payload = pickle.load(model_file)
        except ImportError as exc:
            raise RuntimeError(
                "MLP 权重依赖与当前环境不一致；请安装 " "requirements_nanda_room_matcher.txt"
            ) from exc
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            metadata = payload.get("metadata")
            self.metadata = metadata if isinstance(metadata, dict) else {}
            self.positive_class_index = int(payload.get("positive_class_index", 1))
        else:
            self.model = payload
            self.metadata = {}
            self.positive_class_index = 1
        self.feature_spec = RoomMatchFeatureSpec.from_metadata(self.metadata)
        expected = self.feature_spec.expected_n_features()
        actual = self._model_n_features(self.model)
        metadata_features = self.metadata.get("n_features_in")
        if actual is not None and actual != expected:
            raise ValueError(
                f"MLP 输入维度不匹配: model={actual}, runtime={expected}, "
                f"path={self.model_path}"
            )
        if metadata_features is not None and int(metadata_features) != expected:
            raise ValueError(
                f"MLP metadata 输入维度不匹配: metadata={metadata_features}, "
                f"runtime={expected}"
            )
        LOGGER.info(
            "加载南大 MLP：path=%s feature_set=%s feature_mode=%s n_features=%s",
            self.model_path,
            self.feature_spec.feature_set,
            self.feature_spec.feature_mode,
            expected,
        )

    @staticmethod
    def _model_n_features(model: Any) -> Optional[int]:
        value = getattr(model, "n_features_in_", None)
        if value is None and hasattr(model, "named_steps"):
            final = model.named_steps.get("clf")
            value = getattr(final, "n_features_in_", None)
        return None if value is None else int(value)

    def predict(self, features: np.ndarray) -> float:
        features = np.asarray(features, dtype="float32").reshape(1, -1)
        if not np.all(np.isfinite(features)):
            raise ValueError("MLP 输入包含非有限数值")
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(features)).reshape(-1)
            index = max(0, min(self.positive_class_index, probabilities.size - 1))
            return float(probabilities[index])
        if hasattr(self.model, "decision_function"):
            raw = float(
                np.asarray(self.model.decision_function(features)).reshape(-1)[0]
            )
            return float(1.0 / (1.0 + math.exp(-raw)))
        if hasattr(self.model, "predict"):
            return float(np.asarray(self.model.predict(features)).reshape(-1)[0])
        raise TypeError(f"不支持的 MLP 模型类型: {type(self.model)!r}")


@dataclass
class RoomTemplate:
    room_id: str
    template_id: str
    template_path: Path
    mask_path: Optional[Path]
    action_path: Path
    metadata: Dict[str, Any]
    rgb_embedding: np.ndarray
    gray_embedding: Optional[np.ndarray]


def _iter_room_dirs(room_library_path: Path) -> Iterable[Path]:
    rooms_dir = room_library_path / "rooms"
    for path in sorted(rooms_dir.iterdir(), key=lambda value: _natural_key(value.name)):
        if path.is_dir() and not path.name.startswith("."):
            yield path


def _load_room_metadata(room_dir: Path) -> Dict[str, Any]:
    metadata_path = room_dir / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取房型 metadata: {metadata_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"房型 metadata 必须是对象: {metadata_path}")
    replay = payload.get("replay")
    if not isinstance(replay, dict) or not isinstance(
        replay.get("allow_actions"), bool
    ):
        raise ValueError(f"房型 metadata.replay.allow_actions 无效: {metadata_path}")
    return payload


def _find_capture(room_dir: Path, template_id: str) -> Optional[Path]:
    for suffix in (".jpeg", ".jpg", ".png"):
        candidate = room_dir / "captures" / f"{template_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _locate_source_crop(
    source: np.ndarray, segment: np.ndarray, mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if source.shape[:2] == segment.shape[:2]:
        return source.copy()
    if source.shape[0] < segment.shape[0] or source.shape[1] < segment.shape[1]:
        return None
    if mask is None or mask.shape != segment.shape[:2] or not np.any(mask):
        return None
    visible = mask.astype(bool) & np.any(segment != 0, axis=2)
    if not np.any(visible):
        visible = mask.astype(bool)
    try:
        scores = cv2.matchTemplate(
            cv2.cvtColor(source, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY),
            cv2.TM_CCORR_NORMED,
            mask=visible.astype(np.uint8) * 255,
        )
    except cv2.error:
        return None
    if scores.size == 0:
        return None
    _, best_score, _, best_location = cv2.minMaxLoc(scores)
    if not np.isfinite(best_score) or best_score < 0.80:
        return None
    x1, y1 = best_location
    height, width = segment.shape[:2]
    return source[y1 : y1 + height, x1 : x1 + width].copy()


def _load_template_dino_input(
    room_dir: Path, template_dir: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    segment_path = template_dir / "segment.png"
    segment = cv2.imread(str(segment_path), cv2.IMREAD_COLOR)
    if segment is None:
        raise FileNotFoundError(f"无法读取房型模板: {segment_path}")
    mask_path = template_dir / "mask.png"
    raw_mask = (
        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_path.is_file()
        else None
    )
    mask = _normalize_mask(raw_mask, segment.shape[:2])

    derived_path = room_dir / "derived" / "dino_inputs" / f"{template_dir.name}.png"
    derived = (
        cv2.imread(str(derived_path), cv2.IMREAD_COLOR)
        if derived_path.is_file()
        else None
    )
    if derived is not None and derived.shape[:2] == segment.shape[:2]:
        return derived, _fill_mask_holes(mask)

    source_path = _find_capture(room_dir, template_dir.name)
    source = (
        cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source_path is not None
        else None
    )
    crop = _locate_source_crop(source, segment, mask) if source is not None else None
    base = crop if crop is not None else segment
    return _build_masked_dino_input(base, mask)


class IntegratedNandaRoomMatcher:
    """持久加载 DINOv3、MLP 和房型索引的独立运行时。"""

    def __init__(
        self, asset_paths: NandaMatcherAssetPaths, *, device: Optional[str] = None,
    ):
        self.asset_paths = asset_paths.resolved()
        self.asset_paths.validate()
        self.extractor = DinoV3FeatureExtractor(
            self.asset_paths.dino_model_dir, device=device,
        )
        self.reranker = MlpRoomReranker(self.asset_paths.mlp_model_path)
        self.templates: List[RoomTemplate] = []
        self.templates_by_room: Dict[str, List[RoomTemplate]] = {}
        self.template_matrix: Optional[np.ndarray] = None
        self.faiss_index = None
        self._load_room_library()

    def _cache_identity(self, variant: str) -> Dict[str, Any]:
        weights = self.asset_paths.dino_model_dir / "model.safetensors"
        return {
            "runtime_cache_version": RUNTIME_CACHE_VERSION,
            "preprocess_version": INPUT_PREPROCESS_VERSION,
            "model_bytes": int(weights.stat().st_size),
            "variant": variant,
        }

    def _load_or_compute_embedding(
        self,
        room_dir: Path,
        template_id: str,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        *,
        variant: str,
    ) -> np.ndarray:
        cache_dir = room_dir / "derived" / "autogame_embeddings"
        cache_path = cache_dir / f"{template_id}.{variant}.npy"
        metadata_path = cache_dir / f"{template_id}.{variant}.json"
        expected = {
            **self._cache_identity(variant),
            "image_md5": _image_md5(image),
            "mask_md5": _image_md5(mask),
        }
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cached_metadata == expected:
                return _normalize_embedding(
                    np.load(str(cache_path), allow_pickle=False)
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass

        embedding = self.extractor.extract(image, mask)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp.npy")
            np.save(str(temporary), embedding)
            temporary.replace(cache_path)
            metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
            metadata_tmp.write_text(
                json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            metadata_tmp.replace(metadata_path)
        except OSError as exc:
            LOGGER.warning("无法写入 DINO 模板缓存 %s: %s", cache_path, exc)
        return embedding

    def _load_room_library(self) -> None:
        started = perf_counter()
        room_count = 0
        for room_dir in _iter_room_dirs(self.asset_paths.room_library_path):
            metadata = _load_room_metadata(room_dir)
            if metadata.get("status") == "disabled":
                continue
            action_path = room_dir / "actions" / "action_step.json"
            templates_dir = room_dir / "templates"
            if not action_path.is_file() or not templates_dir.is_dir():
                continue
            room_templates: List[RoomTemplate] = []
            template_dirs = sorted(
                (
                    path
                    for path in templates_dir.iterdir()
                    if path.is_dir()
                    and not path.name.startswith(".")
                    and not path.name.endswith("_structure_overlay")
                    and (path / "segment.png").is_file()
                ),
                key=lambda value: _natural_key(value.name),
            )
            for template_dir in template_dirs:
                try:
                    image, mask = _load_template_dino_input(room_dir, template_dir)
                    rgb_embedding = self._load_or_compute_embedding(
                        room_dir, template_dir.name, image, mask, variant="rgb",
                    )
                    gray_embedding = None
                    if self.reranker.feature_spec.uses_gray_embedding:
                        gray_embedding = self._load_or_compute_embedding(
                            room_dir,
                            template_dir.name,
                            _gray_bgr(image),
                            mask,
                            variant="gray",
                        )
                    record = RoomTemplate(
                        room_id=room_dir.name,
                        template_id=template_dir.name,
                        template_path=template_dir / "segment.png",
                        mask_path=(
                            template_dir / "mask.png"
                            if (template_dir / "mask.png").is_file()
                            else None
                        ),
                        action_path=action_path,
                        metadata=metadata,
                        rgb_embedding=rgb_embedding,
                        gray_embedding=gray_embedding,
                    )
                    room_templates.append(record)
                    self.templates.append(record)
                except Exception as exc:
                    LOGGER.warning(
                        "跳过无法索引的房型模板 room=%s template=%s: %s",
                        room_dir.name,
                        template_dir.name,
                        exc,
                    )
            if room_templates:
                self.templates_by_room[room_dir.name] = room_templates
                room_count += 1

        if not self.templates:
            raise RuntimeError(f"房型库没有可索引模板: {self.asset_paths.room_library_path}")
        self.template_matrix = np.ascontiguousarray(
            np.stack([item.rgb_embedding for item in self.templates]), dtype="float32",
        )
        try:
            import faiss

            index = faiss.IndexFlatIP(self.template_matrix.shape[1])
            index.add(self.template_matrix)
            self.faiss_index = index
            index_backend = "faiss"
        except ImportError:
            self.faiss_index = None
            index_backend = "numpy"
        LOGGER.info(
            "房型库索引完成：rooms=%d templates=%d backend=%s elapsed=%.2fs",
            room_count,
            len(self.templates),
            index_backend,
            perf_counter() - started,
        )

    def health_payload(self) -> Dict[str, Any]:
        return {
            "assets": self.asset_paths.to_jsonable(),
            "room_count": len(self.templates_by_room),
            "template_count": len(self.templates),
            "dino_device": self.extractor.device,
            "mlp_feature_set": self.reranker.feature_spec.feature_set,
            "mlp_feature_mode": self.reranker.feature_spec.feature_mode,
            "structure_mode": "disabled_zero_vector",
        }

    def _retrieve(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        room_limit = min(FACADE_TOP_K, len(self.templates_by_room))
        template_limit = min(
            max(room_limit * ROOM_SHORTLIST_TEMPLATE_MULTIPLIER, room_limit),
            len(self.templates),
        )
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(
                np.ascontiguousarray(query_embedding.reshape(1, -1), dtype="float32"),
                template_limit,
            )
            pairs = zip(indices[0], scores[0])
        else:
            all_scores = np.dot(self.template_matrix, query_embedding)
            order = np.argsort(-all_scores)[:template_limit]
            pairs = ((index, all_scores[index]) for index in order)

        retrieved = []
        for index, score in pairs:
            if int(index) < 0:
                continue
            template = self.templates[int(index)]
            retrieved.append(
                {
                    "room_id": template.room_id,
                    "template": template,
                    "dino_score": float(score),
                }
            )
        retrieved.sort(key=lambda value: value["dino_score"], reverse=True)
        shortlisted_rooms: List[str] = []
        for candidate in retrieved:
            if candidate["room_id"] not in shortlisted_rooms:
                shortlisted_rooms.append(candidate["room_id"])
            if len(shortlisted_rooms) >= room_limit:
                break

        expanded = []
        for room_id in shortlisted_rooms:
            for template in self.templates_by_room[room_id]:
                expanded.append(
                    {
                        "room_id": room_id,
                        "template": template,
                        "dino_score": float(
                            np.dot(query_embedding, template.rgb_embedding)
                        ),
                    }
                )
        expanded.sort(key=lambda value: value["dino_score"], reverse=True)
        return expanded

    @staticmethod
    def _candidate_sort_key(candidate: Mapping[str, Any]) -> Tuple[float, ...]:
        return (
            float(candidate["total_score"]),
            float(candidate.get("room_best_dino_score", candidate["dino_score"])),
            float(candidate["structure_score"]),
            float(candidate["mlp_score"]),
        )

    def match(
        self,
        segmented_bgr: np.ndarray,
        cropped_mask: np.ndarray,
        cropped_bgr: np.ndarray,
    ) -> Tuple[Optional[str], Optional[Path], Dict[str, Any]]:
        started = perf_counter()
        query_image, query_mask = _build_masked_dino_input(
            cropped_bgr
            if cropped_bgr.shape[:2] == segmented_bgr.shape[:2]
            else segmented_bgr,
            cropped_mask,
        )
        query_rgb = self.extractor.extract(query_image, query_mask)
        query_gray = None
        if self.reranker.feature_spec.uses_gray_embedding:
            query_gray = self.extractor.extract(_gray_bgr(query_image), query_mask)

        candidates = self._retrieve(query_rgb)
        ranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            template = candidate["template"]
            features = _build_mlp_features(
                self.reranker.feature_spec,
                query_rgb,
                template.rgb_embedding,
                query_gray,
                template.gray_embedding,
            )
            mlp_score = self.reranker.predict(features)
            mlp_rank_score = max(
                0.0, float(mlp_score) - FACADE_LOW_MLP_CONFIDENCE_SCORE
            )
            structure_score = 0.0
            total_score = mlp_rank_score + FACADE_STRUCT_SCORE_ALPHA * structure_score
            ranked.append(
                {
                    **candidate,
                    "mlp_score": float(mlp_score),
                    "mlp_rank_score": mlp_rank_score,
                    "structure_score": structure_score,
                    "total_score": total_score,
                }
            )

        best_by_room: Dict[str, Dict[str, Any]] = {}
        best_dino_by_room: Dict[str, float] = {}
        for candidate in ranked:
            room_id = candidate["room_id"]
            best_dino_by_room[room_id] = max(
                best_dino_by_room.get(room_id, -1.0), float(candidate["dino_score"]),
            )
            previous = best_by_room.get(room_id)
            if previous is None or self._candidate_sort_key(
                candidate
            ) > self._candidate_sort_key(previous):
                best_by_room[room_id] = candidate
        room_candidates = []
        for room_id, candidate in best_by_room.items():
            room_candidates.append(
                {**candidate, "room_best_dino_score": best_dino_by_room[room_id],}
            )
        room_candidates.sort(key=self._candidate_sort_key, reverse=True)

        thresholds = {
            "min_dino_score": FACADE_NO_MATCH_MIN_DINO_SCORE,
            "strict_dino_score": FACADE_NO_MATCH_STRICT_DINO_SCORE,
            "strong_structure_score": FACADE_NO_MATCH_STRONG_STRUCTURE_SCORE,
            "min_total_score": FACADE_NO_MATCH_MIN_SCORE,
            "min_top2_margin": FACADE_NO_MATCH_MIN_MARGIN,
            "low_mlp_confidence_score": FACADE_LOW_MLP_CONFIDENCE_SCORE,
        }
        debug: Dict[str, Any] = {
            "thresholds": thresholds,
            "structure_mode": "disabled_zero_vector",
            "candidate_count": len(ranked),
            "room_candidate_count": len(room_candidates),
            "elapsed_ms": (perf_counter() - started) * 1000.0,
            "decision": {"status": "no_match"},
        }
        if not room_candidates:
            debug["no_match_reason"] = "empty_room_candidate_pool"
            return None, None, debug

        chosen = room_candidates[0]
        margin = (
            float(chosen["total_score"] - room_candidates[1]["total_score"])
            if len(room_candidates) > 1
            else float("inf")
        )
        template = chosen["template"]
        decision = {
            "status": "matched",
            "room_id": chosen["room_id"],
            "template_path": str(template.template_path),
            "dino_score": chosen["dino_score"],
            "room_best_dino_score": chosen["room_best_dino_score"],
            "mlp_score": chosen["mlp_score"],
            "mlp_rank_score": chosen["mlp_rank_score"],
            "structure_score": chosen["structure_score"],
            "total_score": chosen["total_score"],
            "top2_margin": margin,
            "room_status": template.metadata.get("status"),
            "replay_allow_actions": bool(
                (template.metadata.get("replay") or {}).get("allow_actions", False)
            ),
            "replay_disabled_reason": (template.metadata.get("replay") or {}).get(
                "disabled_reason"
            ),
        }
        debug["decision"] = decision
        debug["top2_margin"] = margin
        debug["top_candidates"] = [
            {
                "room_id": value["room_id"],
                "dino_score": value["dino_score"],
                "room_best_dino_score": value["room_best_dino_score"],
                "mlp_score": value["mlp_score"],
                "total_score": value["total_score"],
                "template_path": str(value["template"].template_path),
            }
            for value in room_candidates[:5]
        ]

        gate_dino_score = float(chosen["room_best_dino_score"])
        no_match_reason = None
        if gate_dino_score < FACADE_NO_MATCH_MIN_DINO_SCORE:
            no_match_reason = "dino_score_below_threshold"
        elif (
            gate_dino_score < FACADE_NO_MATCH_STRICT_DINO_SCORE
            and float(chosen["structure_score"])
            < FACADE_NO_MATCH_STRONG_STRUCTURE_SCORE
        ):
            no_match_reason = "dino_score_below_strict_threshold_without_structure"
        elif float(chosen["total_score"]) < FACADE_NO_MATCH_MIN_SCORE:
            no_match_reason = "total_score_below_threshold"
        elif FACADE_NO_MATCH_MIN_MARGIN > 0 and margin < FACADE_NO_MATCH_MIN_MARGIN:
            no_match_reason = "top2_margin_below_threshold"

        if no_match_reason:
            debug["no_match_reason"] = no_match_reason
            debug["decision"]["status"] = "no_match"
            LOGGER.info(
                "房型匹配被拒绝：reason=%s room=%s dino=%.4f mlp=%.4f total=%.4f",
                no_match_reason,
                chosen["room_id"],
                gate_dino_score,
                chosen["mlp_score"],
                chosen["total_score"],
            )
            return None, None, debug

        LOGGER.info(
            "房型匹配成功：room=%s dino=%.4f mlp=%.4f total=%.4f margin=%s",
            chosen["room_id"],
            gate_dino_score,
            chosen["mlp_score"],
            chosen["total_score"],
            "inf" if not np.isfinite(margin) else f"{margin:.4f}",
        )
        return chosen["room_id"], template.action_path, debug


__all__ = [
    "IntegratedNandaRoomMatcher",
    "NandaMatcherAssetPaths",
]
