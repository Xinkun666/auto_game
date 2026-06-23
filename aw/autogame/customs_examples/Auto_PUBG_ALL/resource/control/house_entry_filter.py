EXCLUDED_HOUSE_GROUP_PREFIXES = ("P城", "M城", "Y城")


def is_excluded_house_group(house_id) -> bool:
    name = str(house_id or "")
    return any(name.startswith(prefix) for prefix in EXCLUDED_HOUSE_GROUP_PREFIXES)


def filter_house_entry_data(house_data):
    if not isinstance(house_data, dict):
        return {}
    return {
        house_id: entries
        for house_id, entries in house_data.items()
        if not is_excluded_house_group(house_id)
    }
