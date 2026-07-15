import argparse
import importlib.metadata as importlib_metadata
import importlib.util
import os
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


def required_runtime_assets() -> list[ReleaseAsset]:
    return [
        _asset("restart.bat", ".", "restart.bat"),
        _asset("testcases/pubg/pubg_full_flow"),
        _asset("aw/autogame/tools"),
        _asset("aw/autogame/config"),
        _asset("aw/autogame/stream_client"),
        _asset("aw/autogame/customs_examples/Auto_PUBG_ALL"),
        _asset("aw/autogame/customs_game_examples/Auto_PUBG_ALL"),
    ]


def pyinstaller_data_assets() -> list[ReleaseAsset]:
    return required_runtime_assets()


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


def validate_source_assets() -> None:
    missing = [asset.source for asset in required_runtime_assets() if not asset.source.exists()]
    if missing:
        lines = "\n".join(f"  - {_path_text(path)}" for path in missing)
        raise SystemExit(f"Required release asset is missing:\n{lines}")


def warn_missing_model_weights() -> None:
    missing = [REPO_ROOT / path for path in MODEL_WEIGHT_FILES if not (REPO_ROOT / path).exists()]
    if not missing:
        return

    print()
    print("[WARN] Model weights are not present in this checkout.")
    print("       The release can still be built, but runtime perception will need these files:")
    for path in missing:
        print(f"       - {_path_text(path)}")


def build_pyinstaller_command() -> list[str]:
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

    command.extend(["--collect-submodules", "aw"])
    command.extend(["--collect-submodules", "aw.autogame.stream_client.hos_sdk"])

    for module_name in REQUIRED_HIDDEN_IMPORTS:
        command.extend(["--hidden-import", module_name])

    for module_name in OPTIONAL_HIDDEN_IMPORTS:
        if _module_exists(module_name):
            command.extend(["--hidden-import", module_name])

    for asset in pyinstaller_data_assets():
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


def copy_runtime_assets() -> None:
    for asset in required_runtime_assets():
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
        ("restart.bat", DIST_DIR / "restart.bat"),
        ("testcase", DIST_DIR / "testcases" / "pubg" / "pubg_full_flow" / "auto_pubg.py"),
        ("root ProcessUtils.py", DIST_DIR / "aw" / "autogame" / "tools" / "ProcessUtils.py"),
        ("root config.json", DIST_DIR / "aw" / "autogame" / "config" / "config.json"),
        ("root customs_examples Auto_PUBG_ALL", DIST_DIR / "aw" / "autogame" / "customs_examples" / "Auto_PUBG_ALL" / "info.py"),
        ("root customs_game_examples Auto_PUBG_ALL", DIST_DIR / "aw" / "autogame" / "customs_game_examples" / "Auto_PUBG_ALL" / "auto_pubg.py"),
        ("internal ProcessUtils.py", internal_root / "aw" / "autogame" / "tools" / "ProcessUtils.py"),
        ("house entry summary", DIST_DIR / "aw" / "autogame" / "customs_examples" / "Auto_PUBG_ALL" / "resource" / "house_entry" / "house_entries_summary.json"),
        ("map mask", DIST_DIR / "aw" / "autogame" / "customs_examples" / "Auto_PUBG_ALL" / "resource" / "map" / "hpjy_mask.tif"),
    ]

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
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    os.chdir(REPO_ROOT)
    validate_source_assets()
    warn_missing_model_weights()
    command = build_pyinstaller_command()

    if args.dry_run:
        print_dry_run(command)
        if not _module_exists("PyInstaller"):
            print()
            print("[WARN] PyInstaller is not installed in this Python environment.")
        return 0

    _print_header("Closing old launcher process")
    terminate_existing_launcher()

    if not args.skip_clean:
        _print_header("Cleaning previous build")
        clean_previous_build()

    _print_header("Running PyInstaller")
    run_pyinstaller(command)

    _print_header("Copying runtime assets to exe root")
    copy_runtime_assets()

    _print_header("Verifying release output")
    verify_release_output()

    print()
    print("Release package is ready:")
    print(f"  {_path_text(DIST_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
