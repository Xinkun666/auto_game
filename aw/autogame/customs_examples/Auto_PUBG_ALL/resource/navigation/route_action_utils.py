"""Helpers for normalizing recorded route actions."""

from typing import Any, Dict, List


def parse_route_to_dicts(route: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    final_actions = []
    pending_angle = 0.0
    index = 0

    while index < len(route):
        step = route[index]
        if step.get("item_type") == "ACTION":
            action = step.get("action", {})
            method = action.get("method")
            args = action.get("args", {})

            if method == "view_slide":
                angle = args.get("angle", 0.0)
                direction = args.get("direction")
                if direction == "LEFT":
                    pending_angle += angle
                elif direction == "RIGHT":
                    pending_angle -= angle
            elif method == "move_press":
                if abs(pending_angle) > 1e-5:
                    final_actions.append(
                        {
                            "action": "view_slide",
                            "direction": "LEFT" if pending_angle > 0 else "RIGHT",
                            "angle": round(abs(pending_angle), 2),
                        }
                    )
                    pending_angle = 0.0

                interval = 0.0
                if index + 1 < len(route):
                    next_step = route[index + 1]
                    if next_step.get("item_type") == "INTERVAL":
                        interval = next_step.get("interval", 0.0)
                final_actions.append(
                    {
                        "action": "move_press",
                        "init_angle": args.get("init_angle"),
                        "interval": round(interval, 4),
                    }
                )
        index += 1

    if abs(pending_angle) > 1e-5:
        final_actions.append(
            {
                "action": "view_slide",
                "direction": "LEFT" if pending_angle > 0 else "RIGHT",
                "angle": round(abs(pending_angle), 2),
            }
        )
    return final_actions
