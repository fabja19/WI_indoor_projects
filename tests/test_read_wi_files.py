import unittest
from pathlib import Path
from modules.read_wi_files import (
    read_wi_file,
    create_object_from_file,
    create_floorplan_from_file,
    extract_materials,
    extract_structure_groups,
    extract_name,
    round_to_significant_digits,
    read_word,
    read_line,
    lines_to_dict,
    merge_dicts,
    get_key_startswith,
)
from modules.project import ObjectWI, FloorPlanWI

class TestReadWiFiles(unittest.TestCase):
    def setUp(self):
        self.wi_files_dir = Path('./wi_project_files')
        self.object_file = self.wi_files_dir / 'blackboard.object'
        self.floorplan_file = self.wi_files_dir / 'seminar_room_door_win.flp'

    def test_read_wi_file(self):
        """Test that a WI file can be read and converted into a dictionary."""
        file_content = read_wi_file(self.object_file)
        self.assertIsInstance(file_content, dict)
        self.assertIn('object:blackboard', file_content.keys())

    def test_extract_name(self):
        """Test that the name of the object or floorplan is extracted correctly."""
        file_content = read_wi_file(self.object_file)
        name = extract_name(file_content)
        self.assertEqual(name, 'object:blackboard')

    def test_extract_materials(self):
        """Test that materials are extracted correctly from the file content."""
        file_content = read_wi_file(self.object_file)
        name = extract_name(file_content)
        materials = extract_materials(file_content, name)
        self.assertIsInstance(materials, dict)
        self.assertGreater(len(materials), 0)

    def test_extract_structure_groups(self):
        """Test that structure groups are extracted correctly from the file content."""
        file_content = read_wi_file(self.object_file)
        structure_groups = extract_structure_groups(file_content)
        self.assertIsInstance(structure_groups, dict)
        self.assertGreater(len(structure_groups), 0)

    def test_create_object_from_file(self):
        """Test that an object file can be loaded into an ObjectWI instance."""
        obj = create_object_from_file(self.object_file)
        self.assertIsInstance(obj, ObjectWI)
        self.assertEqual(obj.name, 'blackboard')

    def test_create_floorplan_from_file(self):
        """Test that a floorplan file can be loaded into a FloorPlanWI instance."""
        floorplan = create_floorplan_from_file(self.floorplan_file, 4)
        self.assertIsInstance(floorplan, FloorPlanWI)
        self.assertEqual(floorplan.name, 'seminar_room_door_win')

    def test_round_to_significant_digits(self):
        """Test rounding to significant digits."""
        self.assertEqual(round_to_significant_digits(1234.5678, 3), 1230.0)
        self.assertEqual(round_to_significant_digits(0.001234, 2), 0.0012)
        self.assertEqual(round_to_significant_digits(0, 3), 0)

    def test_read_word(self):
        """Test reading a word and converting it to int, float, or string."""
        self.assertEqual(read_word("123"), 123)
        self.assertEqual(read_word("123.456"), 123.46)
        self.assertEqual(read_word("word"), "word")
        self.assertEqual(read_word("yes"), True)

    def test_read_line(self):
        """Test parsing a line into structured data."""
        self.assertEqual(read_line("key value"), {"key": "value"})
        self.assertEqual(read_line("1.23 4.56 1.0"), [1.23, 4.56, 1.0])
        self.assertEqual(read_line("123"), 123)
        self.assertIsNone(read_line(""))

    def test_lines_to_dict(self):
        """Test converting lines to a nested dictionary."""
        lines = [
            "begin_<block>",
            "key value",
            "end_<block>"
        ]
        content, remaining = lines_to_dict(lines, None)
        self.assertEqual(content, {"block": {"key": "value"}})
        self.assertEqual(remaining, [])

    def test_merge_dicts(self):
        """Test merging two dictionaries."""
        dict1 = {"key1": "value1"}
        dict2 = {"key2": "value2"}
        dict3 = {"key2": "value3"}
        merged = merge_dicts(dict1, dict2)
        merged2 = merge_dicts(dict2, dict3)
        self.assertEqual(merged, {"key1": "value1", "key2": "value2"})
        self.assertEqual(merged2, {'key2:0': 'value2', 'key2:1': 'value3'})

    def test_get_key_startswith(self):
        """Test retrieving a value by a key prefix."""
        dic = {"prefix_key": "value", "other_key": "other_value"}
        self.assertEqual(get_key_startswith(dic, "prefix"), "value")
        with self.assertRaises(KeyError):
            get_key_startswith(dic, "nonexistent")

if __name__ == "__main__":
    unittest.main()
