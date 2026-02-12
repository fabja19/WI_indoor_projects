import unittest
from shapely.geometry import Polygon, LineString, MultiLineString, Point

from modules.project import (
    MaterialWI, GeometryWI, PolygonWI, LineStringWI, FloorPlanWI,
    ObjectWI, TxWI, ProjectWI, check_overlap_z, check_touching, check_overlap,
    check_contains, check_boundary_contains, check_is_part_of_boundary
)

class TestProject(unittest.TestCase):

    def test_materialwi_equality(self):
        material1 = MaterialWI(conductivity=1.0, permittivity=2.0, name="Material1")
        material2 = MaterialWI(conductivity=1.0, permittivity=2.0, name="Material2")
        material3 = MaterialWI(conductivity=1.5, permittivity=2.5, name="Material3")
        self.assertEqual(material1, material2)
        self.assertNotEqual(material1, material3)

    def test_materialwi_get_properties(self):
        material = MaterialWI(
            conductivity=1.0, permittivity=2.0, roughness=0.1, thickness=0.5,
            material_type="LayeredDielectric", nLayers=1, name="Material1"
        )
        properties = material.get_properties()
        self.assertEqual(properties["conductivity"], 1.0)
        self.assertEqual(properties["permittivity"], 2.0)
        self.assertEqual(properties["roughness"], 0.1)
        self.assertEqual(properties["nLayers"], 1)

    def test_geometrywi_initialization(self):
        geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        material = MaterialWI(name="TestMaterial")
        geom = GeometryWI(geometry=geometry, material=material, z_min=0, z_max=1, name="TestGeometry")
        self.assertEqual(geom.z_min, 0)
        self.assertEqual(geom.z_max, 1)
        self.assertEqual(geom.material, material)

    def test_polygonwi_split(self):
        polygon = PolygonWI(
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            material=None, z_min=0, z_max=1, name="TestPolygon"
        )
        lines = polygon.split()
        self.assertEqual(len(lines), 4)
        self.assertIsInstance(lines[0], LineStringWI)

    def test_floorplanwi_initialization(self):
        floor_geometry = PolygonWI(
            geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            material=None, z_min=0, z_max=3, name="Floor"
        )
        door = LineStringWI(
            geometry=LineString([(0, 0), (1, 0)]),
            material=MaterialWI(name="TestMaterial"), z_min=0, z_max=2, name="Door"
        )
        floorplan = FloorPlanWI(floor=floor_geometry, other=[door], name="ValidFloorPlan", material_walls=None)
        self.assertEqual(floorplan.geometry, floor_geometry.geometry)
        self.assertIn(door, floorplan.other)

    def test_floorplanwi_add_others(self):
        floor_geometry = PolygonWI(
            geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            material=None, z_min=0, z_max=3, name="Floor"
        )
        floorplan = FloorPlanWI(floor=floor_geometry, other=[], name="FloorPlan", material_walls=None)
        window = LineStringWI(
            geometry=LineString([(1, 0), (2, 0)]),
            material=MaterialWI(name="Glass"), z_min=1, z_max=2, name="Window"
        )
        floorplan.add_others(window)
        self.assertIn(window, floorplan.other)

    def test_objectwi_initialization(self):
        polygon1 = PolygonWI(
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            material=None, z_min=0, z_max=1, name="Polygon1"
        )
        polygon2 = PolygonWI(
            geometry=Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            material=None, z_min=0, z_max=1, name="Polygon2"
        )
        obj = ObjectWI(geometry_list=[polygon1, polygon2], name="TestObject")
        self.assertEqual(len(obj.geometry_list), 2)

    def test_objectwi_get_all_materials(self):
        material1 = MaterialWI(name="Material1")
        material2 = MaterialWI(name="Material2")
        polygon1 = PolygonWI(
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            material=material1, z_min=0, z_max=1, name="Polygon1"
        )
        polygon2 = PolygonWI(
            geometry=Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            material=material2, z_min=0, z_max=1, name="Polygon2"
        )
        obj = ObjectWI(geometry_list=[polygon1, polygon2], name="TestObject")
        materials = obj.get_all_materials()
        self.assertIn(material1, materials)
        self.assertIn(material2, materials)

    def test_txwi_initialization(self):
        tx = TxWI(position=[1, 1, 1], name="TestTx")
        self.assertEqual(tx.geometry, Point(1, 1))
        self.assertEqual(tx.z_min, 1)
        self.assertEqual(tx.z_max, 1)


    def test_projectwi_add_geometry(self):
        project = ProjectWI(floorplan=FloorPlanWI(floor= PolygonWI(
            geometry=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            material=None, z_min=0, z_max=3, name="Floor"
        ), other=[], name="FloorPlan", material_walls=None))        
        tx = TxWI(position=[1, 1, 1], name="TestTx")
        project.add_geometry(tx)
        self.assertIn(tx, project.tx)


    def test_check_overlap_z(self):
        geom1 = GeometryWI(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), None, 0, 1, "Geom1")
        geom2 = GeometryWI(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), None, 1, 2, "Geom2")
        self.assertTrue(check_overlap_z(geom1, geom2))

    def test_check_touching(self):
        geom1 = GeometryWI(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), None, 0, 1, "Geom1")
        geom2 = GeometryWI(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), None, 1, 2, "Geom2")
        self.assertTrue(check_touching(geom1, geom2))

    def test_check_overlap(self):
        geom1 = GeometryWI(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), None, 0, 1, "Geom1")
        geom2 = GeometryWI(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), None, 1, 2, "Geom2")
        self.assertFalse(check_overlap(geom1, geom2, touching_counts=False))
        self.assertTrue(check_overlap(geom1, geom2, touching_counts=True))

    def test_check_contains(self):
        geom1 = GeometryWI(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), None, 0, 2, "Geom1")
        geom2 = GeometryWI(Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]), None, 0, 1, "Geom2")
        self.assertTrue(check_contains(geom1, geom2))

    def test_check_boundary_contains(self):
        geom1 = GeometryWI(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), None, 0, 2, "Geom1")
        geom2 = GeometryWI(LineString([(0, 0), (2, 0)]), None, 0, 2, "Geom2")
        self.assertTrue(check_boundary_contains(geom1, geom2))

    def test_check_is_part_of_boundary(self):
        geom1 = GeometryWI(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), None, 0, 2, "Geom1")
        geom2 = GeometryWI(LineString([(0, 0), (2, 0)]), None, 0, 0, "Geom2")
        self.assertTrue(check_is_part_of_boundary(geom1, geom2))


if __name__ == "__main__":
    unittest.main()