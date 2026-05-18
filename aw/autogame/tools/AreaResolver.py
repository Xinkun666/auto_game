# -*- coding: utf-8 -*-


def get_rect_by_anchor(image_width, image_height, area_config):
    """
    Resolve an anchored area config to pixel coordinates: [x1, y1, x2, y2].
    """
    anchor = area_config.get("anchor")
    offset = area_config.get("offset", {})
    size = area_config.get("size", {})

    area_width = int(size["width"])
    area_height = int(size["height"])

    if anchor == "top_left":
        x1 = int(offset.get("left", 0))
        y1 = int(offset.get("top", 0))
        x2 = x1 + area_width
        y2 = y1 + area_height
    elif anchor == "top_right":
        x2 = int(image_width) - int(offset.get("right", 0))
        y1 = int(offset.get("top", 0))
        x1 = x2 - area_width
        y2 = y1 + area_height
    elif anchor == "bottom_left":
        x1 = int(offset.get("left", 0))
        y2 = int(image_height) - int(offset.get("bottom", 0))
        x2 = x1 + area_width
        y1 = y2 - area_height
    elif anchor == "bottom_right":
        x2 = int(image_width) - int(offset.get("right", 0))
        y2 = int(image_height) - int(offset.get("bottom", 0))
        x1 = x2 - area_width
        y1 = y2 - area_height
    else:
        raise ValueError(f"Unsupported anchor type: {anchor}")

    return [x1, y1, x2, y2]


def resolve_area_rect(image_width, image_height, area_config):
    """
    Resolve a new anchor config or legacy normalized rect config to pixels.
    """
    if "anchor" in area_config:
        return get_rect_by_anchor(image_width, image_height, area_config)

    if "rect" in area_config:
        x1_norm, y1_norm, x2_norm, y2_norm = area_config["rect"]
        return [
            int(float(x1_norm) * image_width),
            int(float(y1_norm) * image_height),
            int(float(x2_norm) * image_width),
            int(float(y2_norm) * image_height),
        ]

    raise ValueError("area_config must contain either 'anchor' or 'rect'")


def infer_anchor_config_from_rect(origin_width, origin_height, rect):
    """
    Infer an anchor config from a legacy normalized rect if the whole rect stays
    inside one screen quadrant. Return None when the rect crosses any centerline.
    """
    x1_norm, y1_norm, x2_norm, y2_norm = [float(v) for v in rect]

    in_left = x1_norm < 0.5 and x2_norm < 0.5
    in_right = x1_norm > 0.5 and x2_norm > 0.5
    in_top = y1_norm < 0.5 and y2_norm < 0.5
    in_bottom = y1_norm > 0.5 and y2_norm > 0.5

    if not ((in_left or in_right) and (in_top or in_bottom)):
        return None

    x1 = int(x1_norm * origin_width)
    y1 = int(y1_norm * origin_height)
    x2 = int(x2_norm * origin_width)
    y2 = int(y2_norm * origin_height)
    size = {
        "width": max(0, x2 - x1),
        "height": max(0, y2 - y1),
    }

    if in_top and in_left:
        return {
            "anchor": "top_left",
            "offset": {"left": x1, "top": y1},
            "size": size,
        }
    if in_top and in_right:
        return {
            "anchor": "top_right",
            "offset": {"right": int(origin_width) - x2, "top": y1},
            "size": size,
        }
    if in_bottom and in_left:
        return {
            "anchor": "bottom_left",
            "offset": {"left": x1, "bottom": int(origin_height) - y2},
            "size": size,
        }
    if in_bottom and in_right:
        return {
            "anchor": "bottom_right",
            "offset": {"right": int(origin_width) - x2, "bottom": int(origin_height) - y2},
            "size": size,
        }

    return None


def resolve_area_rect_for_frame(
    frame_width,
    frame_height,
    area_config,
    screen_width=None,
    screen_height=None,
    origin_width=None,
    origin_height=None,
):
    """
    Resolve an area to the requested target coordinate system.

    Area configs are defined in the scene annotation coordinate system. Runtime
    first resolves them against the real screen size, then scales the resolved
    real-screen rect to the requested target size. For vision crops the target
    is the resized frame; for touch points the target is the real screen size.
    """
    has_screen_size = screen_width and screen_height and int(screen_width) > 0 and int(screen_height) > 0
    has_origin_size = origin_width and origin_height and int(origin_width) > 0 and int(origin_height) > 0

    if has_screen_size:
        target_width = int(screen_width)
        target_height = int(screen_height)

        if has_origin_size and int(origin_width) == target_width and int(origin_height) == target_height:
            x1, y1, x2, y2 = resolve_area_rect(origin_width, origin_height, area_config)
        elif has_origin_size and "anchor" not in area_config and "rect" in area_config:
            inferred_config = infer_anchor_config_from_rect(
                int(origin_width),
                int(origin_height),
                area_config["rect"],
            )
            if inferred_config:
                x1, y1, x2, y2 = get_rect_by_anchor(target_width, target_height, inferred_config)
            else:
                x1, y1, x2, y2 = resolve_area_rect(target_width, target_height, area_config)
        else:
            x1, y1, x2, y2 = resolve_area_rect(target_width, target_height, area_config)

        scale_x = float(frame_width) / float(target_width)
        scale_y = float(frame_height) / float(target_height)
        return [
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        ]

    return resolve_area_rect(frame_width, frame_height, area_config)
