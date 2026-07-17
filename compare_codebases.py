#!/usr/bin/env python3
"""Compare two source trees and write a human-readable plain-text report.

The script is dependency-free and is intended for comparing a clean/base copy
of a repository with a copy that was modified on a server.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_IGNORED_NAMES = {
    ".DS_Store",
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "node_modules",
}
DEFAULT_IGNORED_PATTERNS = ("*.pyc", "*.pyo", "*.log", "*.tmp")
PUBG_SCOPED_PARENT_NAMES = {
    "customs_examples",
    "customs_game_examples",
}
PUBG_SCOPED_PROJECT_NAME = "Auto_PUBG_ALL"

@dataclass(frozen=True)
class TextFile:
    size: int
    digest: str
    text: str | None
    encoding: str | None

    @property
    def sha256(self) -> str:
        return self.digest


@dataclass(frozen=True)
class ChangeBlock:
    kind: str
    base_start: int
    base_end: int
    server_start: int
    server_end: int
    base_lines: tuple[str, ...]
    server_lines: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较两份代码目录，并生成按文件归类的中文纯文本差异报告。"
    )
    parser.add_argument("base_dir", type=Path, help="基准代码目录（例如从 GitHub 新拉取的代码）")
    parser.add_argument("server_dir", type=Path, help="服务器改动后的代码目录")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("代码差异报告.txt"),
        help="报告保存路径，固定生成 .txt；默认：./代码差异报告.txt",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="额外忽略的相对路径 glob，可重复使用，例如 --exclude 'outputs/**'",
    )
    parser.add_argument(
        "--include-default-ignored",
        action="store_true",
        help="同时比较 .git、缓存、日志等默认忽略项",
    )
    return parser.parse_args(argv)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_directories(base_dir: Path, server_dir: Path) -> tuple[Path, Path]:
    base_dir = base_dir.expanduser().resolve()
    server_dir = server_dir.expanduser().resolve()
    for label, directory in (("基准目录", base_dir), ("服务器目录", server_dir)):
        if not directory.is_dir():
            raise ValueError(f"{label}不存在或不是目录：{directory}")
    if base_dir == server_dir:
        raise ValueError("基准目录和服务器目录不能是同一个目录。")
    if _is_relative_to(base_dir, server_dir) or _is_relative_to(server_dir, base_dir):
        raise ValueError("两个待比较目录不能互相嵌套，请把它们放在并列目录中。")
    return base_dir, server_dir


def normalize_output_path(output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".txt":
        output = output.with_suffix(".txt")
    return output


def should_ignore(
    relative_path: Path,
    patterns: Sequence[str],
    include_default_ignored: bool,
) -> bool:
    relative_posix = relative_path.as_posix()
    parts = relative_path.parts
    if (
        len(parts) >= 4
        and parts[:2] == ("aw", "autogame")
        and parts[2] in PUBG_SCOPED_PARENT_NAMES
        and parts[3] != PUBG_SCOPED_PROJECT_NAME
    ):
        return True
    if not include_default_ignored:
        if any(part in DEFAULT_IGNORED_NAMES for part in relative_path.parts):
            return True
        if any(fnmatch.fnmatch(relative_path.name, pattern) for pattern in DEFAULT_IGNORED_PATTERNS):
            return True
    return any(
        fnmatch.fnmatch(relative_posix, pattern)
        or fnmatch.fnmatch(relative_path.name, pattern)
        for pattern in patterns
    )


def collect_files(
    root: Path,
    patterns: Sequence[str],
    include_default_ignored: bool,
    ignored_absolute_paths: set[Path],
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for current_root, dir_names, file_names in os.walk(root):
        current = Path(current_root)
        relative_current = current.relative_to(root)
        dir_names[:] = sorted(
            name
            for name in dir_names
            if not should_ignore(
                relative_current / name,
                patterns,
                include_default_ignored,
            )
        )
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(root)
            if path.resolve() in ignored_absolute_paths:
                continue
            if should_ignore(relative, patterns, include_default_ignored):
                continue
            if path.is_file():
                files[relative.as_posix()] = path
    return files


def looks_binary(raw: bytes) -> bool:
    sample = raw[:8192]
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    control_bytes = sum(byte < 32 and byte not in (9, 10, 12, 13) for byte in sample)
    return control_bytes / len(sample) > 0.10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_are_equal(base_path: Path, server_path: Path) -> bool:
    if base_path.stat().st_size != server_path.stat().st_size:
        return False
    with base_path.open("rb") as base_file, server_path.open("rb") as server_file:
        while True:
            base_chunk = base_file.read(1024 * 1024)
            server_chunk = server_file.read(1024 * 1024)
            if base_chunk != server_chunk:
                return False
            if not base_chunk:
                return True


def read_text_file(path: Path) -> TextFile:
    size = path.stat().st_size
    with path.open("rb") as file:
        sample = file.read(8192)
        if looks_binary(sample):
            return TextFile(
                size=size,
                digest=file_sha256(path),
                text=None,
                encoding=None,
            )
        raw = sample + file.read()
    digest = hashlib.sha256(raw).hexdigest()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return TextFile(
                size=size,
                digest=digest,
                text=raw.decode(encoding),
                encoding=encoding,
            )
        except UnicodeDecodeError:
            pass
    return TextFile(size=size, digest=digest, text=None, encoding=None)


def build_change_blocks(base_text: str, server_text: str) -> list[ChangeBlock]:
    from difflib import SequenceMatcher

    base_lines = base_text.splitlines()
    server_lines = server_text.splitlines()
    matcher = SequenceMatcher(None, base_lines, server_lines, autojunk=False)
    changes: list[ChangeBlock] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        kind = {"insert": "added", "delete": "deleted", "replace": "modified"}[tag]
        changes.append(
            ChangeBlock(
                kind=kind,
                base_start=i1 + 1,
                base_end=i2,
                server_start=j1 + 1,
                server_end=j2,
                base_lines=tuple(base_lines[i1:i2]),
                server_lines=tuple(server_lines[j1:j2]),
            )
        )
    return changes


def line_range(start: int, end: int) -> str:
    if end < start:
        return f"第 {start} 行之前"
    if start == end:
        return f"第 {start} 行"
    return f"第 {start}-{end} 行"


def text_block(lines: Iterable[str]) -> list[str]:
    return ["----- 内容开始 -----", *lines, "----- 内容结束 -----"]


def append_text_change_report(
    report: list[str],
    base_file: TextFile,
    server_file: TextFile,
) -> None:
    assert base_file.text is not None and server_file.text is not None
    blocks = build_change_blocks(base_file.text, server_file.text)
    if not blocks:
        report.extend(
            [
                "文件字节发生变化，但逐行文本相同。可能只改了换行符、文件编码或末尾换行。",
                "",
            ]
        )
        return

    grouped = {
        kind: [block for block in blocks if block.kind == kind]
        for kind in ("added", "deleted", "modified")
    }
    headings = {
        "added": "新增了哪些",
        "deleted": "删除了哪些",
        "modified": "改动了哪些",
    }
    for kind in ("added", "deleted", "modified"):
        kind_blocks = grouped[kind]
        report.extend([headings[kind], "-" * 40, ""])
        if not kind_blocks:
            report.extend(["无。", ""])
            continue
        for index, block in enumerate(kind_blocks, start=1):
            if kind == "added":
                report.append(f"[{index}] 服务器代码 {line_range(block.server_start, block.server_end)}：")
                report.append("")
                report.extend(text_block(block.server_lines))
            elif kind == "deleted":
                report.append(f"[{index}] 基准代码 {line_range(block.base_start, block.base_end)}：")
                report.append("")
                report.extend(text_block(block.base_lines))
            else:
                report.append(
                    f"[{index}] 基准代码 {line_range(block.base_start, block.base_end)} "
                    f"改为服务器代码 {line_range(block.server_start, block.server_end)}："
                )
                report.extend(["", "改动前：", ""])
                report.extend(text_block(block.base_lines))
                report.extend(["", "改动后：", ""])
                report.extend(text_block(block.server_lines))
            report.append("")


def append_added_or_deleted_file(report: list[str], file: TextFile, action: str) -> None:
    if file.text is None:
        report.extend(
            [
                f"{action}二进制文件，大小 {file.size} 字节，SHA-256：{file.sha256}。",
                "",
            ]
        )
        return
    lines = file.text.splitlines()
    report.extend(
        [
            f"{action}整个文本文件，共 {len(lines)} 行，编码：`{file.encoding}`。",
            "",
        ]
    )
    if lines:
        report.extend(text_block(lines))
        report.append("")


def find_exact_renames(
    base_only: set[str],
    server_only: set[str],
    base_files: dict[str, Path],
    server_files: dict[str, Path],
) -> list[tuple[str, str]]:
    base_by_size: dict[int, list[str]] = {}
    server_by_size: dict[int, list[str]] = {}
    for relative_path in sorted(base_only):
        size = base_files[relative_path].stat().st_size
        base_by_size.setdefault(size, []).append(relative_path)
    for relative_path in sorted(server_only):
        size = server_files[relative_path].stat().st_size
        server_by_size.setdefault(size, []).append(relative_path)

    base_by_hash: dict[str, list[str]] = {}
    server_by_hash: dict[str, list[str]] = {}
    for size in sorted(base_by_size.keys() & server_by_size.keys()):
        for relative_path in base_by_size[size]:
            digest = file_sha256(base_files[relative_path])
            base_by_hash.setdefault(digest, []).append(relative_path)
        for relative_path in server_by_size[size]:
            digest = file_sha256(server_files[relative_path])
            server_by_hash.setdefault(digest, []).append(relative_path)

    renames: list[tuple[str, str]] = []
    for digest in sorted(base_by_hash.keys() & server_by_hash.keys()):
        old_paths = base_by_hash[digest]
        new_paths = server_by_hash[digest]
        for old_path, new_path in zip(old_paths, new_paths):
            renames.append((old_path, new_path))
            base_only.remove(old_path)
            server_only.remove(new_path)
    return renames


def generate_report(
    base_dir: Path,
    server_dir: Path,
    output: Path,
    patterns: Sequence[str],
    include_default_ignored: bool,
) -> tuple[str, dict[str, int]]:
    output = output.expanduser().resolve()
    ignored_output = {output}
    base_files = collect_files(base_dir, patterns, include_default_ignored, ignored_output)
    server_files = collect_files(server_dir, patterns, include_default_ignored, ignored_output)

    base_paths = set(base_files)
    server_paths = set(server_files)
    base_only = base_paths - server_paths
    server_only = server_paths - base_paths
    common = base_paths & server_paths
    renames = find_exact_renames(base_only, server_only, base_files, server_files)

    modified: list[tuple[str, TextFile, TextFile]] = []
    unchanged_count = 0
    for relative_path in sorted(common):
        if files_are_equal(base_files[relative_path], server_files[relative_path]):
            unchanged_count += 1
        else:
            base_file = read_text_file(base_files[relative_path])
            server_file = read_text_file(server_files[relative_path])
            modified.append((relative_path, base_file, server_file))

    stats = {
        "modified": len(modified),
        "added": len(server_only),
        "deleted": len(base_only),
        "renamed": len(renames),
        "unchanged": unchanged_count,
    }
    report = [
        "代码差异总报告",
        "=" * 80,
        "",
        f"生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"基准代码：{base_dir}",
        f"服务器代码：{server_dir}",
        f"修改文件：{stats['modified']} 个",
        f"新增文件：{stats['added']} 个",
        f"删除文件：{stats['deleted']} 个",
        f"移动/重命名文件：{stats['renamed']} 个",
        f"内容未变文件：{stats['unchanged']} 个",
        "",
        "范围规则：customs_examples 和 customs_game_examples "
        "目录下只比较 Auto_PUBG_ALL，其他项目不计入报告。",
        "",
        "说明：“改动”表示原位置的一段文本被另一段文本替换；"
        "纯插入和纯删除分别列入“新增”和“删除”。",
        "",
    ]

    if not any(stats[key] for key in ("modified", "added", "deleted", "renamed")):
        report.extend(["两份目录没有发现内容差异。", ""])

    if modified:
        report.extend(["修改的文件", "=" * 80, ""])
        for relative_path, base_file, server_file in modified:
            report.extend([relative_path, "-" * 80, ""])
            if base_file.text is None or server_file.text is None:
                report.extend(
                    [
                        "二进制文件内容发生变化。",
                        "",
                        f"改动前：{base_file.size} 字节，SHA-256：{base_file.sha256}",
                        f"改动后：{server_file.size} 字节，SHA-256：{server_file.sha256}",
                        "",
                    ]
                )
            else:
                append_text_change_report(report, base_file, server_file)

    if server_only:
        report.extend(["新增的文件", "=" * 80, ""])
        for relative_path in sorted(server_only):
            report.extend([relative_path, "-" * 80, ""])
            append_added_or_deleted_file(
                report,
                read_text_file(server_files[relative_path]),
                "新增",
            )

    if base_only:
        report.extend(["删除的文件", "=" * 80, ""])
        for relative_path in sorted(base_only):
            report.extend([relative_path, "-" * 80, ""])
            append_added_or_deleted_file(
                report,
                read_text_file(base_files[relative_path]),
                "删除",
            )

    if renames:
        report.extend(["移动或重命名的文件", "=" * 80, ""])
        for old_path, new_path in sorted(renames):
            report.append(f"{old_path} → {new_path}（文件内容未变）")
        report.append("")

    return "\n".join(report).rstrip() + "\n", stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        base_dir, server_dir = validate_directories(args.base_dir, args.server_dir)
        output = normalize_output_path(args.output)
        report, stats = generate_report(
            base_dir=base_dir,
            server_dir=server_dir,
            output=output,
            patterns=args.exclude,
            include_default_ignored=args.include_default_ignored,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1

    total_changes = sum(stats[key] for key in ("modified", "added", "deleted", "renamed"))
    print(f"报告已生成：{output}")
    print(
        "差异汇总："
        f"修改 {stats['modified']}，新增 {stats['added']}，"
        f"删除 {stats['deleted']}，移动/重命名 {stats['renamed']}，"
        f"合计 {total_changes} 个文件。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
