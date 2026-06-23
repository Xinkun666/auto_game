import unittest

from aw.autogame.customs_examples.Auto_PUBG_ALL.resource.control.house_entry_filter import (
    filter_house_entry_data,
    is_excluded_house_group,
)


class HouseEntryFilterTests(unittest.TestCase):
    def test_filters_p_m_and_y_city_groups(self):
        data = {
            "M城_1": [{"location": [0, 1]}],
            "P城_1": [{"location": [0, 2]}],
            "Y城_1": [{"location": [0, 3]}],
            "R城_1": [{"location": [10, 0]}],
            "学校和其他小房区_1": [{"location": [11, 0]}],
        }

        filtered = filter_house_entry_data(data)

        self.assertEqual({"R城_1", "学校和其他小房区_1"}, set(filtered))

    def test_excludes_only_city_prefixes_not_arbitrary_letters(self):
        self.assertTrue(is_excluded_house_group("M城_12"))
        self.assertTrue(is_excluded_house_group("P城_3"))
        self.assertTrue(is_excluded_house_group("Y城_8"))
        self.assertFalse(is_excluded_house_group("R城_1"))
        self.assertFalse(is_excluded_house_group("学校和其他小房区_1"))


if __name__ == "__main__":
    unittest.main()
