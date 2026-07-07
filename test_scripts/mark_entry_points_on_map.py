#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageDraw, ImageFont


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


def entry_direction_vector(entry_dir: float) -> tuple[float, float]:
    angle = math.radians(float(entry_dir))
    x = round(math.sin(angle), 10)
    y = round(-math.cos(angle), 10)
    return (0.0 if x == -0.0 else x, 0.0 if y == -0.0 else y)


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
    arrow_length: int = 3,
    point_radius: int = 0,
    label_entries: bool = False,
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
    font = _load_font(11)
    out_of_bounds: list[EntryPoint] = []

    for entry in entries:
        x, y = entry.location
        if x < 0 or y < 0 or x >= image.width or y >= image.height:
            out_of_bounds.append(entry)
            continue
        _draw_entry_marker(draw, entry, arrow_length=arrow_length, point_radius=point_radius, font=font, label=label_entries)

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
    arrow_length: int,
    point_radius: int,
    font: ImageFont.ImageFont,
    label: bool,
) -> None:
    x, y = entry.location
    vector_x, vector_y = entry_direction_vector(entry.entry_dir)
    end_x = x + vector_x * arrow_length
    end_y = y + vector_y * arrow_length

    draw.line((x, y, end_x, end_y), fill=(255, 218, 64, 255), width=1)
    _draw_arrow_head(draw, end_x, end_y, vector_x, vector_y)
    if point_radius <= 0:
        draw.point((x, y), fill=(255, 64, 64, 255))
    else:
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=(255, 64, 64, 255),
            outline=(255, 255, 255, 255),
            width=1,
        )

    if label:
        label_text = f"h{entry.house_id}e{entry.entry_index} entry_dir={int(round(entry.entry_dir))}"
        _draw_label(draw, label_text, x + point_radius + 3, y + point_radius + 3, font)


def _draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    tip_x: float,
    tip_y: float,
    vector_x: float,
    vector_y: float,
    *,
    size: int = 1,
) -> None:
    angle = math.atan2(vector_y, vector_x)
    left = angle + math.radians(150)
    right = angle - math.radians(150)
    points = [
        (tip_x, tip_y),
        (tip_x + math.cos(left) * size, tip_y + math.sin(left) * size),
        (tip_x + math.cos(right) * size, tip_y + math.sin(right) * size),
    ]
    draw.polygon(points, fill=(255, 218, 64, 255))


def _draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0, 170),
    )
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _house_sort_key(item: tuple[str, object]) -> tuple[int, object]:
    house_id = str(item[0])
    if house_id.isdigit():
        return (0, int(house_id))
    return (1, house_id)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark PUBG house entry points and entry_dir arrows on hpjg.png.",
    )
    parser.add_argument("--map-image", type=Path, default=DEFAULT_MAP_IMAGE)
    parser.add_argument("--entries-json", type=Path, default=DEFAULT_ENTRIES_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_IMAGE)
    parser.add_argument("--arrow-length", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=0)
    parser.add_argument("--labels", action="store_true", help="show hXeY entry_dir labels")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        render_entry_map(
            args.map_image,
            args.entries_json,
            args.output,
            arrow_length=args.arrow_length,
            point_radius=args.point_radius,
            label_entries=args.labels,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
