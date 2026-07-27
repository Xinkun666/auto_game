"""Backward-compatible imports for the former navigation utility grab bag.

New code should import from the owning module directly:

- device resolution/shell: ``aw.autogame.tools.Utils``
- geometry and stability: ``navigation_geometry``
- configured car routes: ``car_route_utils``
- image calibration: ``navigation_vision``
- recorded actions: ``route_action_utils``
"""

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.car_route_utils import (
    CAR_POINT_CONFIG,
    find_path,
    generate_shortest_path,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.device_config_utils import (
    get_auto_brightness,
    get_brightness,
    get_buttons,
    get_dms_rotation_mode,
    get_rois,
    parse_tuple_str,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_geometry import (
    analyze_distance,
    calculate_angle,
    calculate_move_count_old as calculate_move_count,
    extract_keys,
    find_nearest_point,
    get_distance,
    get_relative_sector,
    is_location_stagnant,
    round_to_nearest_5,
    stable_angle,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.navigation_vision import (
    correct_mini_map_roi,
    correct_speed_roi,
    detect_angel,
    extract_color_centers,
    find_boundaries,
    get_fast_running_status,
    hex_to_rgb,
)
from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.navigation.route_action_utils import (
    parse_route_to_dicts,
)
from aw.autogame.tools.Utils import (
    _parse_screen_resolution,
    get_resolution,
    get_wh,
    run_shell,
)


__all__ = [
    "CAR_POINT_CONFIG",
    "_parse_screen_resolution",
    "analyze_distance",
    "calculate_angle",
    "calculate_move_count",
    "correct_mini_map_roi",
    "correct_speed_roi",
    "detect_angel",
    "extract_color_centers",
    "extract_keys",
    "find_boundaries",
    "find_nearest_point",
    "find_path",
    "generate_shortest_path",
    "get_auto_brightness",
    "get_brightness",
    "get_buttons",
    "get_distance",
    "get_dms_rotation_mode",
    "get_fast_running_status",
    "get_relative_sector",
    "get_resolution",
    "get_rois",
    "get_wh",
    "hex_to_rgb",
    "is_location_stagnant",
    "parse_route_to_dicts",
    "parse_tuple_str",
    "round_to_nearest_5",
    "run_shell",
    "stable_angle",
]
