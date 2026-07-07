#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_IMAGE = (
    REPO_ROOT
    / "aw"
    / "autogame"
    / "customs_examples"
    / "Auto_PUBG_ALL"
    / "resource"
    / "map"
    / "hpjg.png"
)
DEFAULT_ENTRIES_JSON = (
    REPO_ROOT
    / "aw"
    / "autogame"
    / "customs_examples"
    / "Auto_PUBG_ALL"
    / "resource"
    / "house_entry"
    / "house_entries_summary.json"
)
DEFAULT_OUTPUT_IMAGE = Path(__file__).resolve().parent / "hpjg_entry_points_with_dirs.png"


@dataclass(frozen=True)
class EntryPoint:
    entry_id: str
    house_id: str
    entry_index: int
    location: tuple[int, int]
    entry_dir: float
    source_image: str


def load_house_entries(entries_json: Path) -> list[EntryPoint]:
    entries_json = Path(entries_json)
    data = json.loads(entries_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"entries json must be an object: {entries_json}")

    points: list[EntryPoint] = []
    for house_id, entries in sorted(data.items(), key=_house_sort_key):
        if not isinstance(entries, list):
            continue
        for entry_index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue
            location = entry.get("location")
            if not isinstance(location, list) or len(location) < 2:
                continue
            if "direction" not in entry:
                continue
            x, y = int(round(float(location[0]))), int(round(float(location[1])))
            entry_dir = float(entry["direction"])
            points.append(
                EntryPoint(
                    entry_id=f"house_{house_id}_entry_{entry_index}",
                    house_id=str(house_id),
                    entry_index=entry_index,
                    location=(x, y),
                    entry_dir=entry_dir,
                    source_image=str(entry.get("image") or ""),
                )
            )
    return points


def render_entry_map(
    map_image: Path,
    entries_json: Path,
    output_image: Path,
    *,
    point_radius: int = 0,
) -> Path:
    map_image = Path(map_image)
    entries_json = Path(entries_json)
    output_image = Path(output_image)
    if not map_image.is_file():
        raise FileNotFoundError(f"map image not found: {map_image}")
    if not entries_json.is_file():
        raise FileNotFoundError(f"entries json not found: {entries_json}")

    entries = load_house_entries(entries_json)
    if not entries:
        raise ValueError(f"no entry points loaded from: {entries_json}")

    image = Image.open(map_image).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    out_of_bounds: list[EntryPoint] = []

    for entry in entries:
        print(
            f"{entry.entry_id} location={entry.location} entry_dir={int(round(entry.entry_dir))}"
        )
        x, y = entry.location
        if x < 0 or y < 0 or x >= image.width or y >= image.height:
            out_of_bounds.append(entry)
            continue
        _draw_entry_marker(
            draw,
            entry,
            point_radius=point_radius,
        )

    result = Image.alpha_composite(image, overlay).convert("RGBA")
    output_image.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_image)

    if out_of_bounds:
        preview = ", ".join(f"{entry.entry_id}@{entry.location}" for entry in out_of_bounds[:8])
        suffix = "" if len(out_of_bounds) <= 8 else f", ... {len(out_of_bounds) - 8} more"
        print(f"[WARN] skipped out-of-bounds entries: {preview}{suffix}")
    print(f"[OK] wrote {output_image} with {len(entries) - len(out_of_bounds)}/{len(entries)} entry points")
    return output_image


def _draw_entry_marker(
    draw: ImageDraw.ImageDraw,
    entry: EntryPoint,
    *,
    point_radius: int,
) -> None:
    x, y = entry.location

    if point_radius <= 0:
        draw.point((x, y), fill=(255, 64, 64, 255))
    else:
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=(255, 64, 64, 255),
            outline=(255, 255, 255, 255),
            width=1,
        )


def _house_sort_key(item: tuple[str, object]) -> tuple[int, object]:
    house_id = str(item[0])
    if house_id.isdigit():
        return (0, int(house_id))
    return (1, house_id)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark PUBG house entry points on hpjg.png and print each entry_dir.",
    )
    parser.add_argument("--map-image", type=Path, default=DEFAULT_MAP_IMAGE)
    parser.add_argument("--entries-json", type=Path, default=DEFAULT_ENTRIES_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_IMAGE)
    parser.add_argument("--point-radius", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        render_entry_map(
            args.map_image,
            args.entries_json,
            args.output,
            point_radius=args.point_radius,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
