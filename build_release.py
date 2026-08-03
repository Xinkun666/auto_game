import argparse
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


APP_NAME = "AutoGameLauncher"
REPO_ROOT = Path(__file__).resolve().parent
BUILD_DIR = REPO_ROOT / "build"
DIST_DIR = REPO_ROOT / "dist" / APP_NAME
SPEC_FILE = REPO_ROOT / f"{APP_NAME}.spec"
INTERNAL_DIR = DIST_DIR / "_internal"
PUBG_RESOURCE_DEST = Path(
    "aw/autogame/customs_examples/Auto_PUBG_ALL/resource"
)
NANDA_CONFIG_FILE = REPO_ROOT / "aw/autogame/customs_examples/Auto_PUBG_ALL/config.json"
NANDA_PROJECT_ENV = "AUTOGAME_NANDA_PROJECT_ROOT"
NANDA_DINO_DEST = PUBG_RESOURCE_DEST / "weights/nanda_room_matcher/dinov3_vitl16"
NANDA_MLP_DEST = PUBG_RESOURCE_DEST / "weights/nanda_room_matcher/rgb_mlp_struct_v7.pkl"
NANDA_ROOMS_DEST = PUBG_RESOURCE_DEST / "nanda_room_library/rooms"

OPTIONAL_PACKAGES = (
    "xdevice",
    "devicetest",
    "hypium",
    "aosp",
    "ohos",
    "ultralytics",
    "lap",
)
OPTIONAL_HIDDEN_IMPORTS = (
    "_core",
    "aosp.drivers.android",
    "ohos.drivers.cpp_driver",
    "ultralytics",
    "lap",
)
NANDA_REQUIRED_DISTRIBUTIONS = (
    ("torch", "torch", (2, 0)),
    ("transformers", "transformers", (5, 0)),
    ("safetensors", "safetensors", (0, 4, 5)),
    ("sklearn", "scikit-learn", (1, 7, 2)),
    ("numpy", "numpy", (2, 0)),
)
NANDA_COLLECT_ALL_PACKAGES = (
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("sklearn", "scikit-learn"),
)
NANDA_OPTIONAL_COLLECT_ALL_PACKAGES = (
    ("faiss", "faiss-cpu"),
)
REQUIRED_HIDDEN_IMPORTS = (
    "aw.autogame.tools.ProcessUtils",
    "aw.autogame.stream_client.hos_sdk",
    "aw.autogame.stream_client.hos_sdk.HosRemoteConfig",
    "aw.autogame.stream_client.hos_sdk.HosRemoteDevice",
    "aw.autogame.stream_client.hos_sdk.ScreenCapCallback",
    "aw.autogame.stream_client.hos_sdk.communication.proto.scrcpy_pb2",
    "aw.autogame.stream_client.hos_sdk.communication.proto.scrcpy_pb2_grpc",
)
MODEL_WEIGHT_FILES = (
    "aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/direction_ctc.pt",
    "aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/best.pt",
    "aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/speed_classifier.pt",
    "aw/autogame/customs_examples/Auto_PUBG_ALL/resource/weights/scene_best_model.pth",
)


@dataclass(frozen=True)
class ReleaseAsset:
    source: Path
    pyinstaller_dest: str
    runtime_dest: Optional[Path]

    def add_data_arg(self) -> str:
        separator = ";" if os.name == "nt" else ":"
        try:
            source = self.source.relative_to(REPO_ROOT)
        except ValueError:
            source = self.source
        return f"{source}{separator}{self.pyinstaller_dest}"


def _asset(source: str, pyinstaller_dest: Optional[str] = None, runtime_dest: Optional[str] = None) -> ReleaseAsset:
    return ReleaseAsset(
        source=REPO_ROOT / source,
        pyinstaller_dest=pyinstaller_dest or source,
        runtime_dest=Path(runtime_dest if runtime_dest is not None else source),
    )


def _external_asset(source: Path, destination: Path) -> ReleaseAsset:
    return ReleaseAsset(
        source=source.expanduser().resolve(),
        pyinstaller_dest=destination.as_posix(),
        runtime_dest=destination,
    )


def nanda_house_search_enabled() -> bool:
    try:
        payload = json.loads(NANDA_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to read PUBG config for release: {NANDA_CONFIG_FILE}: {exc}") from exc
    config = payload.get("nanda_house_search", {})
    value = config.get("enabled", False) if isinstance(config, dict) else False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _nanda_package_root(project_root: Path) -> Optional[Path]:
    project_root = project_root.expanduser().resolve()
    candidates = (
        project_root,
        project_root / "control_proxy/src/gametest_proxy/pubg_room_explore",
    )
    for candidate in candidates:
        if (
            (candidate / "img_similarity/dinov3_vitl16").is_dir()
            and (candidate / "models/rgb_mlp_struct_v7.pkl").is_file()
            and (candidate / "room_library/rooms").is_dir()
        ):
            return candidate
    return None


def resolve_nanda_package_root(explicit_project_root: Optional[Path] = None) -> Path:
    if explicit_project_root is not None:
        package_root = _nanda_package_root(explicit_project_root)
        if package_root is None:
            raise SystemExit(
                "The explicit Nanda project root is incomplete: "
                f"{explicit_project_root.expanduser()}"
            )
        return package_root

    local_resource_root = REPO_ROOT / PUBG_RESOURCE_DEST
    local_root = local_resource_root / "weights/nanda_room_matcher"
    if (
        (local_root / "dinov3_vitl16").is_dir()
        and (local_root / "rgb_mlp_struct_v7.pkl").is_file()
        and (local_resource_root / "nanda_room_library/rooms").is_dir()
    ):
        return local_resource_root

    configured_root = os.environ.get(NANDA_PROJECT_ENV, "").strip()
    if configured_root:
        package_root = _nanda_package_root(Path(configured_root))
        if package_root is None:
            raise SystemExit(
                f"{NANDA_PROJECT_ENV} points to an incomplete Nanda project: "
                f"{configured_root}"
            )
        return package_root

    default_root = REPO_ROOT.parent / "pubg_test-main"
    package_root = _nanda_package_root(default_root)
    if package_root is not None:
        return package_root

    raise SystemExit(
        "Nanda house search is enabled, but its release assets were not found.\n"
        "Provide --nanda-project-root or set AUTOGAME_NANDA_PROJECT_ROOT.\n"
        f"Default location checked:\n  - {default_root}"
    )


def nanda_runtime_assets(project_root: Optional[Path] = None) -> list[ReleaseAsset]:
    if not nanda_house_search_enabled():
        return []

    package_root = resolve_nanda_package_root(project_root)
    if package_root == REPO_ROOT / PUBG_RESOURCE_DEST:
        dino_source = package_root / "weights/nanda_room_matcher/dinov3_vitl16"
        mlp_source = package_root / "weights/nanda_room_matcher/rgb_mlp_struct_v7.pkl"
        rooms_source = package_root / "nanda_room_library/rooms"
    else:
        dino_source = package_root / "img_similarity/dinov3_vitl16"
        mlp_source = package_root / "models/rgb_mlp_struct_v7.pkl"
        rooms_source = package_root / "room_library/rooms"
    return [
        _external_asset(dino_source, NANDA_DINO_DEST),
        _external_asset(mlp_source, NANDA_MLP_DEST),
        _external_asset(rooms_source, NANDA_ROOMS_DEST),
    ]


def required_runtime_assets(nanda_project_root: Optional[Path] = None) -> list[ReleaseAsset]:
    assets = [
        _asset("testcases/pubg/pubg_full_flow"),
        _asset("aw/autogame/common"),
        _asset("aw/autogame/tools"),
        _asset("aw/autogame/config"),
        _asset("aw/autogame/stream_client"),
        _asset("aw/autogame/customs_examples/Auto_PUBG_ALL"),
        _asset("aw/autogame/customs_game_examples/Auto_PUBG_ALL"),
    ]
    assets.extend(nanda_runtime_assets(nanda_project_root))
    return assets


def pyinstaller_data_assets(nanda_project_root: Optional[Path] = None) -> list[ReleaseAsset]:
    return required_runtime_assets(nanda_project_root)


def _module_exists(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _metadata_exists(package_name: str) -> bool:
    try:
        importlib_metadata.distribution(package_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return True


def _path_text(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _print_header(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def _is_real_model_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as model_file:
        return not model_file.read(256).startswith(b"version https://git-lfs.github.com/spec/")


def validate_nanda_assets(assets: Iterable[ReleaseAsset]) -> None:
    by_destination = {asset.runtime_dest: asset.source for asset in assets}
    dino_dir = by_destination.get(NANDA_DINO_DEST)
    mlp_path = by_destination.get(NANDA_MLP_DEST)
    rooms_dir = by_destination.get(NANDA_ROOMS_DEST)
    if dino_dir is None or mlp_path is None or rooms_dir is None:
        raise SystemExit("Nanda release asset mapping is incomplete.")

    missing = []
    for file_name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        path = dino_dir / file_name
        if not path.is_file():
            missing.append(path)
    if not mlp_path.is_file():
        missing.append(mlp_path)
    if not rooms_dir.is_dir():
        missing.append(rooms_dir)
    if missing:
        lines = "\n".join(f"  - {_path_text(path)}" for path in missing)
        raise SystemExit(f"Nanda release asset is missing:\n{lines}")

    model_path = dino_dir / "model.safetensors"
    invalid_models = [
        path for path in (model_path, mlp_path) if not _is_real_model_file(path)
    ]
    if invalid_models:
        lines = "\n".join(f"  - {_path_text(path)}" for path in invalid_models)
        raise SystemExit(
            "Nanda release contains an empty model or Git LFS pointer; fetch the real files first:\n"
            f"{lines}"
        )

    if not any(rooms_dir.glob("*/metadata.json")):
        raise SystemExit(f"Nanda room library has no room metadata: {_path_text(rooms_dir)}")


def validate_source_assets(assets: Optional[list[ReleaseAsset]] = None) -> None:
    assets = assets if assets is not None else required_runtime_assets()
    missing = [asset.source for asset in assets if not asset.source.exists()]
    if missing:
        lines = "\n".join(f"  - {_path_text(path)}" for path in missing)
        raise SystemExit(f"Required release asset is missing:\n{lines}")
    if nanda_house_search_enabled():
        validate_nanda_assets(assets)


def warn_missing_model_weights() -> None:
    missing = [REPO_ROOT / path for path in MODEL_WEIGHT_FILES if not (REPO_ROOT / path).exists()]
    if not missing:
        return

    print()
    print("[WARN] Model weights are not present in this checkout.")
    print("       The release can still be built, but runtime perception will need these files:")
    for path in missing:
        print(f"       - {_path_text(path)}")


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def nanda_dependency_problems() -> list[str]:
    problems = []
    for module_name, distribution_name, minimum in NANDA_REQUIRED_DISTRIBUTIONS:
        if not _module_exists(module_name):
            problems.append(f"missing module {module_name} ({distribution_name})")
            continue
        try:
            installed = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            problems.append(f"missing metadata {distribution_name}")
            continue
        if _version_tuple(installed) < minimum:
            required = ".".join(str(part) for part in minimum)
            problems.append(f"{distribution_name}>={required} required, found {installed}")
    return problems


def validate_build_environment() -> None:
    if not nanda_house_search_enabled():
        return
    problems = nanda_dependency_problems()
    if problems:
        lines = "\n".join(f"  - {problem}" for problem in problems)
        raise SystemExit(
            "The selected Python environment cannot build the enabled Nanda runtime:\n"
            f"{lines}\nInstall requirements_nanda_room_matcher.txt first."
        )


def _collect_package(command: list[str], module_name: str, distribution_name: str) -> None:
    command.extend(["--collect-all", module_name])
    command.extend(["--collect-submodules", module_name])
    if _metadata_exists(distribution_name):
        command.extend(["--copy-metadata", distribution_name])


def build_pyinstaller_command(
    assets: Optional[list[ReleaseAsset]] = None,
) -> list[str]:
    assets = assets if assets is not None else pyinstaller_data_assets()
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        APP_NAME,
        "--windowed",
        "--onedir",
        "--clean",
    ]

    skipped_packages = []
    for package in OPTIONAL_PACKAGES:
        if not _module_exists(package):
            skipped_packages.append(package)
            continue
        command.extend(["--collect-all", package])
        command.extend(["--collect-submodules", package])
        if _metadata_exists(package):
            command.extend(["--copy-metadata", package])

    if nanda_house_search_enabled():
        for module_name, distribution_name in NANDA_COLLECT_ALL_PACKAGES:
            if _module_exists(module_name):
                _collect_package(command, module_name, distribution_name)
        for module_name, distribution_name in NANDA_OPTIONAL_COLLECT_ALL_PACKAGES:
            if _module_exists(module_name):
                _collect_package(command, module_name, distribution_name)
        if _metadata_exists("torch"):
            command.extend(["--copy-metadata", "torch"])

    command.extend(["--collect-submodules", "aw"])
    command.extend(["--collect-submodules", "aw.autogame.stream_client.hos_sdk"])

    for module_name in REQUIRED_HIDDEN_IMPORTS:
        command.extend(["--hidden-import", module_name])

    for module_name in OPTIONAL_HIDDEN_IMPORTS:
        if _module_exists(module_name):
            command.extend(["--hidden-import", module_name])

    for asset in assets:
        command.extend(["--add-data", asset.add_data_arg()])

    command.append(str(REPO_ROOT / "launcher.py"))

    if skipped_packages:
        print("[INFO] Optional packages not installed; not collecting:", ", ".join(skipped_packages))

    return command


def terminate_existing_launcher() -> None:
    if os.name != "nt":
        return
    for process_name in (f"{APP_NAME}.exe", "AutoGameLauncherDebug.exe"):
        subprocess.run(
            ["taskkill", "/F", "/IM", process_name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def clean_previous_build() -> None:
    for path in (BUILD_DIR, DIST_DIR):
        if path.exists():
            shutil.rmtree(path)
    if SPEC_FILE.exists():
        SPEC_FILE.unlink()


def _ignore_runtime_copy(dir_name: str, names: Iterable[str]) -> set[str]:
    ignored = set()
    for name in names:
        lower_name = name.lower()
        if name == "__pycache__" or lower_name in {".ds_store", ".pytest_cache"}:
            ignored.add(name)
        elif lower_name.endswith((".pyc", ".pyo")):
            ignored.add(name)
    return ignored


def copy_runtime_assets(assets: Optional[list[ReleaseAsset]] = None) -> None:
    assets = assets if assets is not None else required_runtime_assets()
    for asset in assets:
        if asset.runtime_dest is None:
            continue
        target = DIST_DIR / asset.runtime_dest
        target.parent.mkdir(parents=True, exist_ok=True)
        if asset.source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(asset.source, target, ignore=_ignore_runtime_copy)
        else:
            shutil.copy2(asset.source, target)

    (DIST_DIR / "aw" / "autogame" / "temp" / "logs" / "process_temp_logs").mkdir(parents=True, exist_ok=True)


def run_pyinstaller(command: list[str]) -> None:
    if not _module_exists("PyInstaller"):
        raise SystemExit(
            "PyInstaller is not installed in the selected Python environment. "
            "Install it with: python -m pip install pyinstaller"
        )

    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def verify_release_output() -> None:
    internal_root = INTERNAL_DIR if INTERNAL_DIR.exists() else DIST_DIR
    checks = [
        (APP_NAME, _first_existing((DIST_DIR / f"{APP_NAME}.exe", DIST_DIR / APP_NAME))),
        ("testcase", DIST_DIR / "testcases" / "pubg" / "pubg_full_flow" / "auto_pubg.py"),
        ("root ProcessUtils.py", DIST_DIR / "aw" / "autogame" / "tools" / "ProcessUtils.py"),
        ("root config.json", DIST_DIR / "aw" / "autogame" / "config" / "config.json"),
        ("root customs_examples Auto_PUBG_ALL", DIST_DIR / "aw" / "autogame" / "customs_examples" / "Auto_PUBG_ALL" / "info.py"),
        ("root customs_game_examples Auto_PUBG_ALL", DIST_DIR / "aw" / "autogame" / "customs_game_examples" / "Auto_PUBG_ALL" / "auto_pubg.py"),
        ("internal ProcessUtils.py", internal_root / "aw" / "autogame" / "tools" / "ProcessUtils.py"),
        ("house entry summary", DIST_DIR / "aw" / "autogame" / "customs_examples" / "Auto_PUBG_ALL" / "resource" / "house_entry" / "house_entries_summary.json"),
        ("map mask", DIST_DIR / "aw" / "autogame" / "customs_examples" / "Auto_PUBG_ALL" / "resource" / "map" / "hpjy_mask.tif"),
    ]
    if nanda_house_search_enabled():
        checks.extend([
            ("Nanda DINOv3 model", DIST_DIR / NANDA_DINO_DEST / "model.safetensors"),
            ("Nanda MLP model", DIST_DIR / NANDA_MLP_DEST),
            ("Nanda room library", DIST_DIR / NANDA_ROOMS_DEST),
        ])

    missing = []
    for label, path in checks:
        if path is None or not path.exists():
            missing.append(label)

    if missing:
        lines = "\n".join(f"  - {label}" for label in missing)
        raise SystemExit(f"Release verification failed; missing:\n{lines}")


def print_dry_run(command: list[str]) -> None:
    _print_header("Dry run")
    print("Repository:", REPO_ROOT)
    print("Output:", DIST_DIR)
    print()
    print("PyInstaller command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the AutoGameLauncher release package.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the command without building.")
    parser.add_argument("--skip-clean", action="store_true", help="Keep existing build/dist files.")
    parser.add_argument(
        "--nanda-project-root",
        type=Path,
        help=(
            "Path to the Nanda demo repository or pubg_room_explore directory. "
            "Defaults to AUTOGAME_NANDA_PROJECT_ROOT, then ../pubg_test-main."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    os.chdir(REPO_ROOT)
    assets = required_runtime_assets(args.nanda_project_root)
    validate_source_assets(assets)
    warn_missing_model_weights()
    command = build_pyinstaller_command(assets)

    if args.dry_run:
        print_dry_run(command)
        if not _module_exists("PyInstaller"):
            print()
            print("[WARN] PyInstaller is not installed in this Python environment.")
        problems = nanda_dependency_problems() if nanda_house_search_enabled() else []
        if problems:
            print()
            print("[WARN] Nanda build dependencies are not ready:")
            for problem in problems:
                print(f"       - {problem}")
        return 0

    validate_build_environment()

    _print_header("Closing old launcher process")
    terminate_existing_launcher()

    if not args.skip_clean:
        _print_header("Cleaning previous build")
        clean_previous_build()

    _print_header("Running PyInstaller")
    run_pyinstaller(command)

    _print_header("Copying runtime assets to exe root")
    copy_runtime_assets(assets)

    _print_header("Verifying release output")
    verify_release_output()

    print()
    print("Release package is ready:")
    print(f"  {_path_text(DIST_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
