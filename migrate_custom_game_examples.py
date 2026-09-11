#!/usr/bin/env python3
"""把旧 customs_game_examples 用例复制到对应项目的 scripts 目录。"""

from pathlib import Path
import shutil
import tempfile


REPO_ROOT = Path(__file__).resolve().parent
IGNORE_FILES = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store")


def find_cases(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    )


def copy_scripts(source_dir: Path, scripts_dir: Path) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".scripts-migration-", dir=scripts_dir.parent
    ) as temp_dir:
        staged_dir = Path(temp_dir) / "scripts"
        shutil.copytree(source_dir, staged_dir, ignore=IGNORE_FILES)
        staged_dir.rename(scripts_dir)


def migrate(repo_root: Path = REPO_ROOT) -> int:
    projects_root = repo_root / "aw" / "autogame" / "customs_examples"
    legacy_root = repo_root / "aw" / "autogame" / "customs_game_examples"
    if not projects_root.is_dir():
        print(f"[失败] 未找到项目目录：{projects_root}")
        return 1
    if not legacy_root.is_dir():
        print(f"[完成] 未找到旧用例目录，无需迁移：{legacy_root}")
        return 0

    copied = skipped = failed = 0
    projects = sorted(
        path
        for path in projects_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    for project_dir in projects:
        scripts_dir = project_dir / "scripts"
        legacy_dir = legacy_root / project_dir.name
        if scripts_dir.exists():
            print(f"[跳过] {project_dir.name}：scripts 已存在")
            skipped += 1
            continue
        if not legacy_dir.is_dir():
            print(f"[跳过] {project_dir.name}：没有旧同名用例目录")
            skipped += 1
            continue

        cases = find_cases(legacy_dir)
        if not cases:
            print(f"[跳过] {project_dir.name}：旧目录中没有 Python 用例")
            skipped += 1
            continue
        try:
            copy_scripts(legacy_dir, scripts_dir)
        except OSError as exc:
            print(f"[失败] {project_dir.name}：{exc}")
            failed += 1
            continue
        print(f"[迁移] {project_dir.name}：复制 {len(cases)} 个用例到 {scripts_dir}")
        copied += 1

    print(f"[完成] 迁移 {copied} 个项目，跳过 {skipped} 个，失败 {failed} 个")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(migrate())
