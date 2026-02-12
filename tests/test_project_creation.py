import unittest
from pathlib import Path
from modules.project_creation import (
    create_project_randomly,
    load_and_place_objects,
    randomize_materials,
    rotate_geometry,
    move_geometry_to_random_position,
    create_floorplan_randomly,
    normalize_geometry_position,
    move_geometry_by,
    move_geometry_to,
    get_values,
    extrude_wall_objects,
    extrude_wall_object,
    place_wall_objects,
    get_random_line_in_wall,
)
from modules.project import ProjectWI, ObjectWI, FloorPlanWI, MaterialWI, LineStringWI, PolygonWI, GeometryWI
from shapely.geometry import Polygon, LineString
from yaml import safe_load
from typing import cast

class TestProjectCreation(unittest.TestCase):

    def setUp(self):
        # Load a sample configuration file for testing
        config_path = Path('./configs/config1.yml')
        with open(config_path, 'r') as f:
            self.cfg = safe_load(f)

    def test_create_project_randomly(self):
        project, _ = create_project_randomly(self.cfg)
        self.assertIsInstance(project, ProjectWI)
        self.assertIsInstance(project.floorplan, FloorPlanWI)
        self.assertGreaterEqual(len(project.objects), 0)

    def test_load_and_place_objects(self):
        project = ProjectWI(floorplan=create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)[0])
        load_and_place_objects(project, 'chair', self.cfg['objects']['chair'], self.cfg['materials'], './wi_project_files', verbose=False)
        self.assertGreaterEqual(len(project.objects), 0)

    def test_randomize_materials(self):
        project = ProjectWI(floorplan=create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)[0])
        load_and_place_objects(project, 'chair', self.cfg['objects']['chair'], self.cfg['materials'], './wi_project_files', verbose=False)
        obj = project.objects[0]
        randomize_materials(obj, self.cfg['materials'])
        for geom in obj.geometry_list:
            self.assertIsInstance(geom.material, MaterialWI)

    def test_rotate_object(self):
        project = ProjectWI(floorplan=create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)[0])
        load_and_place_objects(project, 'chair', self.cfg['objects']['chair'], self.cfg['materials'], './wi_project_files', verbose=False)
        obj = project.objects[0]
        rotated_obj = cast(ObjectWI, rotate_geometry(obj, 1))
        self.assertNotEqual(obj.geometry_list[0].geometry, rotated_obj.geometry_list[0].geometry)

    def test_move_object_to_random_position(self):
        project = ProjectWI(floorplan=create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)[0])
        load_and_place_objects(project, 'chair', self.cfg['objects']['chair'], self.cfg['materials'], './wi_project_files', verbose=False)
        obj = project.objects[0]
        moved_obj = move_geometry_to_random_position(project, obj)
        self.assertIsNotNone(moved_obj)

    def test_create_floorplan_randomly(self):
        floorplan, _ = create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)
        self.assertIsInstance(floorplan, FloorPlanWI)
        self.assertIsInstance(floorplan.geometry, Polygon)

    def test_normalize_object_position(self):
        project = ProjectWI(floorplan=create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)[0])
        load_and_place_objects(project, 'chair', self.cfg['objects']['chair'], self.cfg['materials'], './wi_project_files', verbose=False)
        obj = project.objects[0]
        normalized_obj = normalize_geometry_position(obj)
        x_min, y_min = min(c[0] for c in normalized_obj.geometry.exterior.coords), min(c[1] for c in normalized_obj.geometry.exterior.coords)  # type: ignore
        self.assertAlmostEqual(x_min, 0, places=3)
        self.assertAlmostEqual(y_min, 0, places=3)

    def test_move_object_by(self):
        project = ProjectWI(floorplan=create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)[0])
        load_and_place_objects(project, 'chair', self.cfg['objects']['chair'], self.cfg['materials'], './wi_project_files', verbose=False)
        obj = project.objects[0]
        moved_obj = cast(ObjectWI, move_geometry_by(obj, [1, 1]))
        x_diff = moved_obj.geometry_list[0].geometry.bounds[0] - obj.geometry_list[0].geometry.bounds[0]
        y_diff = moved_obj.geometry_list[0].geometry.bounds[1] - obj.geometry_list[0].geometry.bounds[1]
        self.assertAlmostEqual(x_diff, 1, places=3)
        self.assertAlmostEqual(y_diff, 1, places=3)

    def test_move_object_to(self):
        project = ProjectWI(floorplan=create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)[0])
        load_and_place_objects(project, 'chair', self.cfg['objects']['chair'], self.cfg['materials'], './wi_project_files', verbose=False)
        obj = project.objects[0]
        moved_obj = move_geometry_to(obj, [0.01, 0.01])
        x_min, y_min = min(c[0] for c in moved_obj.geometry.exterior.coords), min(c[1] for c in moved_obj.geometry.exterior.coords)  # type: ignore
        self.assertAlmostEqual(x_min, 0.01, places=3)
        self.assertAlmostEqual(y_min, 0.01, places=3)

    def test_get_values(self):
        values = get_values(self.cfg['floorplan'])
        self.assertIsInstance(values, dict)
        self.assertIn('z_min', values)

    def test_generate_project_from_config(self):
        project, _ = create_project_randomly(self.cfg)
        self.assertIsInstance(project, ProjectWI)
        self.assertGreaterEqual(len(project.objects), 0)
        self.assertIsInstance(project.floorplan, FloorPlanWI)

    def test_objects_of_same_type_share_material_properties(self):
        project, _ = create_project_randomly(self.cfg)
        object_types = {}
        for obj in project.objects:
            obj_type = obj.name.rstrip('0123456789')  # Remove numeric suffix to get the base type
            if obj_type not in object_types:
                object_types[obj_type] = []
            object_types[obj_type].append(obj)
        for obj_type, objects in object_types.items():
            if len(objects) > 1:
                for idx in range(len(objects[0].geometry_list)):
                    material = objects[0].geometry_list[idx].material
                    for obj in objects[1:]:
                        geom = obj.geometry_list[idx]
                        self.assertEqual(
                            geom.material,
                            material,
                            f"Objects of type {obj_type} do not share the same material properties.",
                        )
    def test_extrude_wall_objects(self):
        project = ProjectWI(floorplan=create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)[0])
        extrude_wall_objects(project, self.cfg['floorplan'])
        self.assertGreaterEqual(len(project.objects), 0)

    def test_extrude_wall_object(self):
        line_string = LineStringWI(
            geometry=LineString([(0, 0), (1, 0)]),
            material=MaterialWI(name="TestMaterial"),
            z_min=0,
            z_max=2,
            name="TestLineString"
        )
        obj = extrude_wall_object(line_string, obj_depth=0.5)
        self.assertIsInstance(obj, ObjectWI)
        self.assertEqual(len(obj.geometry_list), 1)

    def test_place_wall_objects(self):
        floorplan, _ = create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)
        place_wall_objects(floorplan, 'window', self.cfg['floorplan']['wall_objects']['window'], self.cfg['materials'], verbose=False)
        self.assertGreaterEqual(len(floorplan.other), 0)

    def test_get_random_line_in_wall(self):
        floorplan, _ = create_floorplan_randomly(self.cfg['floorplan'], self.cfg['materials'], verbose=False)
        line = get_random_line_in_wall(floorplan, side_length=1.0)
        self.assertIsInstance(line, LineString)



if __name__ == '__main__':
    unittest.main()