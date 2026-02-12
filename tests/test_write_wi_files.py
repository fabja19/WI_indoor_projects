import unittest
from pathlib import Path
from modules.read_wi_files import (
    create_object_from_file,
    create_floorplan_from_file,
    create_project_from_dir
)
from modules.write_wi_files import (
    object_to_str_list,
    floorplan_to_str_list,
    str_list_to_file,
    project_to_files,
    material_to_str_list,
    get_header,
    extract_material_index_geometry,
    polygon_to_sub_structure_str_list,
    get_file_name
)
from modules.project_creation import create_project_randomly
from modules.project import ObjectWI, FloorPlanWI, ProjectWI, MaterialWI, GeometryWI, Polygon, PolygonWI
import yaml
from shutil import rmtree

class TestWriteWiFiles(unittest.TestCase):
    def setUp(self):
        self.wi_files_dir = Path('./wi_project_files')
        self.temp_dir = Path('./temp_test_files')
        if self.temp_dir.exists():
            rmtree(self.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.object_file = self.wi_files_dir / 'blackboard.object'
        self.floorplan_file = self.wi_files_dir / 'seminar_room_door_win.flp'
        self.config_file = Path('./configs/config1.yml')
        self.xml_glob = 'x3d3_3_1*'

    def tearDown(self):
        # Clean up temporary files
        rmtree(self.temp_dir)
        pass

    def test_extract_material_index_geometry(self):
        """Test extracting material index from geometry."""
        material1 = MaterialWI(name="Material1", permittivity=1)
        material2 = MaterialWI(name="Material2", permittivity=2)
        geometry = GeometryWI(material=material1, geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), z_min=0, z_max=1, name="TestGeometry")
        material_list = [material1, material2]

        # Verify correct index
        index = extract_material_index_geometry(geometry, material_list)
        self.assertEqual(index, 0)

        # Verify ValueError for missing material
        geometry.material = None
        with self.assertRaises(ValueError):
            extract_material_index_geometry(geometry, material_list)

    def test_read_write_object(self):
        """Test reading an object file, writing it back, and re-reading it."""
        obj = create_object_from_file(self.object_file)
        self.assertIsInstance(obj, ObjectWI)

        # Write the object back to a file
        lines = object_to_str_list(obj)
        temp_file = self.temp_dir / 'blackboard.object'
        str_list_to_file(lines, temp_file)

        # Re-read the written file
        obj_reloaded = create_object_from_file(temp_file)
        self.assertIsNotNone(obj.name)
        self.assertIsNotNone(obj_reloaded.name)
        if obj_reloaded.name is not None:
            self.assertIn(obj.name, obj_reloaded.name)
        self.assertEqual(len(obj.geometry_list), len(obj_reloaded.geometry_list))

    def test_str_list_to_file(self):
        """Test writing a list of strings to a file."""
        lines = ["line1", "line2", "line3"]
        temp_file = self.temp_dir / 'test_file.txt'
        str_list_to_file(lines, temp_file)

        # Verify file content
        with open(temp_file, 'r') as f:
            content = f.readlines()
        self.assertEqual(content, ["line1\n", "line2\n", "line3\n"])

        # Verify FileExistsError
        with self.assertRaises(FileExistsError):
            str_list_to_file(lines, temp_file)

    def test_get_header(self):
        """Test generating a predefined header structure."""
        header = get_header()
        expected_header = [
            'begin_<reference>',
            'cartesian',
            'longitude 0.0',
            'latitude 0.0',
            'visible no',
            'sealevel',
            'end_<reference>'
        ]
        self.assertEqual(header, expected_header)

    def test_read_write_floorplan(self):
        """
        Test reading a floorplan file, writing it back, and re-reading it..
        """
        floorplan = create_floorplan_from_file(self.floorplan_file, 4)
        self.assertIsInstance(floorplan, FloorPlanWI)

        # Write the floorplan back to a file
        lines = floorplan_to_str_list(floorplan)
        # Verify specific lines in the serialized floorplan
        self.assertIn('begin_<floorplan>', lines[0])
        self.assertIn('end_<floorplan>', lines[-1])
        temp_file = self.temp_dir / 'seminar_room_door_win.flp'
        temp_file.unlink(missing_ok=True)
        str_list_to_file(lines, temp_file)

        # Re-read the written file
        floorplan_reloaded = create_floorplan_from_file(temp_file, 0)
        self.assertEqual(floorplan.name, floorplan_reloaded.name)
        self.assertEqual(len(floorplan.other), len(floorplan_reloaded.other))

    def test_read_write_project_from_config(self):
        """Test creating a project from a config file, writing it to files and reloading it again."""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Create a project from the config
        project, _ = create_project_randomly(config, wi_files_dir=self.wi_files_dir)
        self.assertIsInstance(project, ProjectWI)
        self.assertIsNotNone(project.floorplan)
        self.assertGreater(len(project.objects), 0)

        # Write the project to files
        project_to_files(project, self.temp_dir, xml_glob=self.xml_glob, cfg_here={})

        # Verify that files were created
        floorplan_file = self.temp_dir / 'floorplan.flp'
        self.assertTrue(floorplan_file.exists())
        for obj in project.objects:
            object_file = self.temp_dir / f"{obj.name}.object"
            self.assertTrue(object_file.exists())

        # Re-read the project
        project_reloaded = create_project_from_dir(self.temp_dir)
        self.assertEqual(project, project_reloaded)

    def test_polygon_to_sub_structure_str_list(self):
        """Test converting a PolygonWI to a sub-structure string list."""
        material = MaterialWI(name="Material1")
        polygon = PolygonWI(name="TestPolygon", geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), z_min=0, z_max=1, material=material)
        material_list = [material]

        lines = polygon_to_sub_structure_str_list(polygon, material_list)
        self.assertIn('begin_<sub_structure> TestPolygon', lines)
        self.assertIn('end_<sub_structure>', lines)

    def test_project_to_files(self):
        """Test writing a project to files."""
        material = MaterialWI(name="DefaultMaterial")
        floorplan = FloorPlanWI(
            floor=PolygonWI(
                geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                material=material,
                z_min=0,
                z_max=1,
                name="DefaultPolygon"
            ),
            name="TestFloorplan",
            other=[],
            material_walls=None
        )
        obj = ObjectWI(
            name="TestObject", 
            geometry_list=[
                PolygonWI(
                geometry=Polygon([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]),
                material=material,
                z_min=0.2,
                z_max=0.7,
                name="TestPolygon"
            ),
            ])
        project = ProjectWI(floorplan=floorplan, objects=[obj])

        # Write project to files
        project_to_files(project, self.temp_dir, xml_glob=self.xml_glob, cfg_here={})

        # Verify files
        floorplan_file = self.temp_dir / 'TestFloorplan.flp'
        object_file = self.temp_dir / 'TestObject.object'
        self.assertTrue(floorplan_file.exists())
        self.assertTrue(object_file.exists())

        # Verify ValueError for missing floorplan
        project.floorplan = None # type: ignore
        with self.assertRaises(ValueError):
            project_to_files(project, self.temp_dir, xml_glob=self.xml_glob, cfg_here={})

    def test_material_to_str_list(self):
        """Test converting a MaterialWI to a string list."""
        material = MaterialWI(name="TestMaterial", material_type="wood", thickness=0.5)
        lines = material_to_str_list(material, 0)

        self.assertIn('begin_<Material> TestMaterial', lines)
        self.assertIn('Material 0', lines)
        self.assertIn('wood', lines)
        self.assertIn('thickness 0.5', lines)
        self.assertIn('end_<Material>', lines)

    def test_get_file_name(self):
        """Test generating unique file names."""
        floorplan = FloorPlanWI(
            floor=PolygonWI(
                geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                material=MaterialWI(name="DefaultMaterial"),
                z_min=0,
                z_max=1,
                name="DefaultPolygon"
            ),
            name="TestFloorplan",
            other=[],
            material_walls=None
        )
        file_path = get_file_name(floorplan, self.temp_dir)
        self.assertEqual(file_path.name, "TestFloorplan.flp")
        (self.temp_dir / f"TestFloorplan.flp").touch()

        # Simulate existing files
        for i in range(101):
            (self.temp_dir / f"TestFloorplan{i}.flp").touch()

        with self.assertRaises(ValueError):
            get_file_name(floorplan, self.temp_dir)

if __name__ == "__main__":
    unittest.main()
