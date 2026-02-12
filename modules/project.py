'''
Here we define an indoor project (environment) as a Python class that contains certain attributes like a floorplan, objects and Tx/Rx.
For compatibility, we stick close to how things are defined in WI.
All objects we consider are combinations of shapely geometries together with a min and max height value and a material.
'''
# shapely allows us to define polygons and e.g. check whether one polygon is contained inside another or whether two polygons have any overlap, which will be useful
from shapely import Polygon, LineString, MultiLineString, Point, MultiPolygon, union_all, plotting, equals
from shapely.geometry.base import BaseGeometry
from typing import Sequence, Any, TypeVar, Generic
import numpy as np
from numpy.typing import ArrayLike
from warnings import warn
from matplotlib import patches, pyplot as plt, lines as mlines
from matplotlib.figure import Figure

class MaterialWI:
    """
    Represents a material with specific physical properties.

    Attributes:
        conductivity (float): The electrical conductivity of the material.
        permittivity  (float): The permittivity of the material.
        roughness (float): The surface roughness of the material.
        thickness (float): The thickness of the material.
        material_type (str): The type of the material.
        nLayers (int): The number of layers in the material.
        DielectricLayer (dict): A dictionary representing the dielectric layer properties like conductivity etc above.
            WI requires this for some material types.
        name (str): The name of the material.
    """
    def __init__(self, 
            name : str,
            conductivity : float | None = None,
            permittivity : float | None = None,
            roughness : float | None = None,
            thickness : float | None = None,
            material_type : str | None = None,
            nLayers : int | None = None,
            **DielectricLayers : dict[str,dict[str, float|int]]
        ) -> None:
        """
        Initializes the material properties.

        Parameters:
            conductivity (float): The electrical conductivity of the material.
            permittivity (float): The relative permittivity (dielectric constant) of the material.
            roughness (float): The surface roughness of the material.
            thickness (float): The thickness of the material.
            material_type (str): The material type.
            nLayers (int): The number of layers.
            **DielectricLayers (dict[str,dict[str, float|int]]): one or several dictionary representing the dielectric layer properties,
                    number should correspond to nLayers.
            name (str): The name of the material.
        """
        self.conductivity = conductivity
        self.permittivity = permittivity
        self.roughness = roughness
        self.thickness = thickness
        self.material_type = material_type
        self.nLayers = nLayers
        self.name = name
        self.DielectricLayer = []
        for k, v in DielectricLayers.items():
            assert 'DielectricLayer' in k, f'{k=} not DielectricLayer?'
            if v is None:
                continue
            self.DielectricLayer.append(v)
    
    def get_properties(self) -> dict[str, str|int|float|list|None]:
        '''
        Creates a dict with the material attributes that are called 'properties' in WI.
        '''
        return {
                'conductivity' : self.conductivity,
                'permittivity' : self.permittivity,
                'roughness' : self.roughness,
                'nLayers' : self.nLayers,
                'DielectricLayer' : self.DielectricLayer
            }
    
    def get_conductivity(self) -> float | None:
        """
        Returns the conductivity value for the object.

        If the `conductivity` attribute is set, it returns its value.
        Otherwise, it returns the 'conductivity' value from the first element of the `DielectricLayer` list.

        Returns:
            float | None: The conductivity value, or None if not available.
        """
        if self.material_type=='PEC':
            return np.inf
        elif self.conductivity is not None:
            return self.conductivity  
        elif len(self.DielectricLayer) > 0:
            return self.DielectricLayer[0]['conductivity']
        else: 
            return None
        
    def get_thickness(self) -> float | None:
        """
        Returns the thickness value for the object.

        If the `thickness` attribute is set, it returns its value.
        Otherwise, it returns the 'thickness' value from the first element of the `DielectricLayer` list.

        Returns:
            float | None: The thickness value, or None if not available.
        """
        if self.material_type=='PEC':
            return 0
        elif self.thickness is not None:
            return self.thickness  
        elif len(self.DielectricLayer) > 0:
            return self.DielectricLayer[0]['thickness']
        else: 
            return None
    
    def get_permittivity(self) -> float | None:
        """
        Returns the permittivity value for the object.

        If the object's `permittivity` attribute is set (not None), it returns that value.
        Otherwise, it returns the permittivity from the first element of the `DielectricLayer` list.

        Returns:
            float | None: The permittivity value, or None if not available.
        """
        if self.material_type=='PEC':
            return 1
        elif self.permittivity is not None:
            return self.permittivity   
        elif len(self.DielectricLayer) > 0:
            return self.DielectricLayer[0]['permittivity']
        else: 
            return None
        
    def __eq__(self, 
            other : Any
        ) -> bool:
        """
        Compare this object with another for equality, excluding the material name.

        Args:
            other (Any): The object to compare with this instance.

        Returns:
            bool: True if the other object is an instance of MaterialWI and has 
                  the same properties as this instance (excluding the name), False otherwise.
        """
        if isinstance(other, MaterialWI):
            return {k : v for k, v in self.__dict__.items() if not k=='name'} == {k : v for k, v in other.__dict__.items() if not k=='name'}
        else:
            return False

G = TypeVar("G", bound=BaseGeometry)

class GeometryWI(Generic[G]):
    """
    GeometryWI is a class that represents a 2.5D geometric object with an associated material.

    Attributes:
        geometry (shapely.BaseGeometry): The 2D geometric representation of the object.
        material (MaterialWI | None): The material associated with the geometric object, or None for Tx/Rx, 
            floorplans and objects that are composed of several objects.
        z_min (float): The minimum height of the object.
        z_max (float): The maximum height of the object.
        name (str): Name to be written into WI file.
    """
    def __init__(self,
            geometry : G,
            material : MaterialWI | None,
            z_min : float,
            z_max : float, 
            name : str | None
        ) -> None:
        """
        Initializes a new instance of GeometryWI.
        Args:
            geometry (shapely.BaseGeometry): The 2D geometric representation of the object.
            material (Material): The material properties of the object.
            z_min (float): The minimum height of the object.
            z_max (float): The maximum height of the object.
            
        """
        if z_max < z_min:
            raise ValueError(f'{z_max=} must be greater than or equal to {z_min=}')
        self.geometry : G = geometry
        self.material = material
        self.z_min = z_min
        self.z_max = z_max
        self.name = name if name is not None else 'None'
    
    def get_all_materials(self) -> list[MaterialWI]:
        """
        Returns a list containing the material associated with the instance, if it exists.
        For a simple geometry, this is not very useful.
        But for Objects defined below, this will include all materials of the contained geometries.

        Returns:
            list[MaterialWI]: A list with the material if it is not None, otherwise an empty list.
        """
        return [self.material] if self.material is not None else []

    def get_hull_coords(self) -> list[tuple[float,float]]:
        """
        Returns the coordinates of the convex hull of the geometry associated with this object.

        The method computes the coordinates of the convex hull for various geometry types. 
        For empty geometries, it returns an empty list. For valid geometries,
        it returns a list of (x, y) tuples representing the exterior coordinates of the convex hull polygon,
        or a single (x, y) tuple if the hull is a point.

        Returns:
            list[tuple[float, float]]: List of (x, y) coordinates representing the convex hull.

        Raises:
            ValueError: If the geometry type is unsupported or the convex hull has an unexpected type.
        """
        geom = self.geometry
        if geom.is_empty:
            return []

        # Convert to a unified geometry for hull computation
        if isinstance(geom, (LineString, Polygon, Point)):
            hull = geom.convex_hull
        elif isinstance(geom, (MultiLineString, MultiPolygon)):
            hull = union_all(geom).convex_hull
        else:
            raise ValueError(f"Unsupported geometry type: {type(geom)}")

        # The convex hull is always a Polygon or Point
        if hull.geom_type == 'Polygon':
            return list(hull.exterior.coords) # type: ignore
        elif hull.geom_type == 'Point':
            return [(hull.x, hull.y)] # type: ignore
        else:
            raise ValueError(f"Unexpected hull geometry type: {hull.geom_type}")

    def __eq__(self, other : Any) -> bool:
        if isinstance(other, GeometryWI):
            return check_equals_geometrically(self, other) and self.material == other.material
        else:
            return False

    def get_contained_geometries(self) -> Sequence["GeometryWI"]:
        """
        Returns a list containing all the "sub"geometries that will correspond to WI (sub)structures.
        This is useful for floorplans and ObjectWI to recursively obtain them all.

        Returns:
            list[GeometryWI]: A list with all contained geometries.
        """
        return [self]

class LineStringWI(GeometryWI[LineString]):
    """
    A class representing a LineString with associated material properties. These will correspond to substructures in WI. 
    Only used for wall parts like door, windows, since all other objects should have a certain depth.
    Attributes:
        geometry (LineString): The 2D geometric representation of the LineString.
        material (MaterialWI): The material associated with the LineString.
        z_min (float): The minimum height of the object.
        z_max (float): The maximum height of the object.
        name (str): Name to be written into WI file.
    """
    def __init__(self,
            geometry : LineString,
            material : MaterialWI | None,
            z_min : float, 
            z_max : float,
            name : str | None
        ) -> None:
        """
        Initializes a new instance of the class LineStringWI.
        Args:
            geometry (LineString): The 2D geometric representation of the object.
            material (Material): The material properties associated with the object.
            z_min (float): The minimum height of the object.
            z_max (float): The maximum height of the object.
            name (str): Name to be written into WI file.
        """
        super().__init__(
            geometry=geometry, 
            material=material, 
            z_min=z_min, 
            z_max=z_max, 
            name=name
        )

class PolygonWI(GeometryWI[Polygon]):
    """
    A class representing a polygon with associated material properties. These will correspond to substructures 
    in WI. This should represent all geometries we use, except wall parts (windows, doors) and Tx/Rx.
    Attributes:
        geometry (Polygon): The 2D geometric representation of the polygon.
        material (MaterialWI | None): The material associated with the polygon.
    """
    def __init__(self,
            geometry : Polygon,
            material : MaterialWI | None,
            z_min : float,
            z_max : float, 
            name : str | None
        ) -> None:
        """
        Initializes a new instance of the class PolygonWI.
        Args:
            geometry (Polygon): The geometric representation of the object.
            material (Material): The material properties associated with the object.
            z_min (float): The minimum height of the object.
            z_max (float): The maximum height of the object.
            name (str): Name to be written into WI file.
        """
        super().__init__(
            geometry=geometry, 
            material=material, 
            z_min=z_min, 
            z_max=z_max, 
            name=name
        )
    
    def split(self) -> list[LineStringWI]:
        lines = []
        coords = self.geometry.boundary.coords
        for i in range(len(coords)):
            if coords[(i - 1)%len(coords)]==coords[i]:
                continue
            lines.append(
                LineStringWI(
                    geometry=LineString([coords[(i - 1)%len(coords)], coords[i]]), 
                    material=self.material,
                    z_min=self.z_min,
                    z_max=self.z_max,
                    name=f'{self.name}{i}'))
        return lines

class FloorPlanWI(PolygonWI): 
    """
    We are currently handling this as a subclass of GeometryWI, so that we can easily check whether objects are 
    contained. Represents a floor plan with walls and additional flat objects inside the walls like doors, windows.

    Attributes:
        floor (PolygonWI): The polygon representing the walls of the floor plan.
        other (LineStringWI): The line string representing the door, windows etc within the walls.
        name (str | None): Name to be written into the WI file.
        material_walls (MaterialWI | None): The material associated with the walls.

    Notes:
        - The `geometry` parameter must fully contain all geometries in other.
    """
    def __init__(self,
            floor : PolygonWI, 
            other : list[LineStringWI],
            name : str | None,
            material_walls : MaterialWI | None,
        ) -> None:
        """
        Initializes an instance of FloorPlanWI.

        Args:
            floor (PolygonWI): The polygon representing the walls of the structure.
            other (list[LineStringWI]): A list of line strings or multi-line strings representing doors, windows, etc.
            name (str | None): Name to be written into the WI file.
            material_walls (MaterialWI | None): The material associated with the walls.

        Raises:
            ValueError: 
                - If the objects in `other` are not contained within the walls.
                - If any two objects in `other` overlap with each other.
        """
        for o in other:
            if not check_boundary_contains(floor, o):
                raise ValueError(f'Object {o.name} is not part of the walls')
        for o1 in other:
            for o2 in other:
                if o1 != o2 and check_overlap(o1, o2, touching_counts=False):
                    raise ValueError(f'Objects {o1.name} and {o2.name} in other cannot overlap')
        self.other = other
        self.material_walls = material_walls
        super().__init__(
            geometry=floor.geometry,  # type: ignore
            material=floor.material, 
            z_min=floor.z_min, 
            z_max=floor.z_max, 
            name=name
        )
    
    def add_others(
            self, 
            other : list[LineStringWI]|LineStringWI
        ) -> None:
        """
        Adds one or more LineStringWI objects to the 'other' attribute of the instance.

        Parameters:
            other (list[LineStringWI] | LineStringWI ): 
                A single LineStringWI object, or a list of such objects, to be added.

        Raises:
            ValueError: 
                - If any object in 'other' is not contained within the boundaries of the instance.
                - If any two objects in 'other' overlap (excluding touching edges).
                - If any new object overlaps with existing objects in `self.other`.

        Notes:
            - If a single object is passed instead of a list, it will be converted into a list.
            - The 'other' attribute of the instance is extended with the provided objects.
                    """
        if not isinstance(other, list):
            other = [other]
        for o in other:
            if not check_boundary_contains(self, o):
                raise ValueError(f'Object {o.name} is not part of the walls')
        for o1 in other:
            for o2 in other:
                if o1 != o2 and check_overlap(o1, o2, touching_counts=False):
                    raise ValueError(f'Objects {o1.name} and {o2.name} cannot overlap')
        for o1 in other:
            for o2 in self.other:
                if check_overlap(o1, o2, touching_counts=False):
                    raise ValueError(f'New object {o1.name} cannot overlap with already defined object {o2.name}')
        self.other.extend(other)

    def get_all_materials(self) -> list[MaterialWI]:
        material_list = [self.material_walls] if self.material_walls is not None else []
        if self.material is not None and (len(material_list)==0 or material_list[0] != self.material):
            material_list.append(self.material) 
        for m in self.other:
            if m.material is not None and not any(m.material == mat for mat in material_list):
                material_list.append(m.material)
        return material_list

    def split(self) -> list[tuple[LineStringWI,list[LineStringWI]]]:
        wall_list = super().split()
        # adjust materials, the ones assigned here were for the ceiling/floor?
        for w in wall_list:
            w.material = self.material_walls
        # now for each wall segment find the corresponding objects in other
        wall_obj_list = [(w, [o for o in self.other if check_contains(w, o)]) for w in wall_list]
        if not all(any(o in os for _, os in wall_obj_list) for o in self.other):
            raise RuntimeError(f'Not all objects of the floorplan contained in a wall?')
        return wall_obj_list

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FloorPlanWI):
            return (super().__eq__(other) and equal_ignore_order(self.other, other.other) and self.material_walls==other.material_walls)
        else:
            return False

    def get_contained_geometries(self) -> Sequence[GeometryWI]:
        return self.other + [LineStringWI(LineString(self.geometry.exterior), self.material_walls, self.z_min, self.z_max, "walls")]

class ObjectWI(PolygonWI):
    """
    A class representing an object composed of multiple PolygonWI geometries.

    Attributes:
        geometry_list (Sequence[PolygonWI]): The list of PolygonWI objects provided.
        geometry (Polygon): The union of all geometries in `geometry_list`. We can use this to check whether the 
            object is contained in or overlaps with others.
        z_min (float): The minimum z-coordinate among all objects in `geometry_list`.
        z_max (float): The maximum z-coordinate among all objects in `geometry_list`.
        name (str | None): The name of the instance.
    """
    def __init__(self,
            geometry_list : Sequence[PolygonWI],
            name : str | None
        ) -> None:
        """
        Initializes an instance of ObjectWI.

        Args:
            geometry_list (Sequence[PolygonWI]): A sequence of PolygonWI objects that define the geometry.
                - All objects in the list must not overlap with each other.
            name (str | None): An optional name for the instance.

        Raises:
            Warning: If any two objects in `geometry_list` overlap.
            ValueError: If any object in `geometry_list` is not an instance of PolygonWI.       
        """
        # if not all([not check_overlap(o1, o2, touching_counts=False) for o1 in geometry_list for o2 in geometry_list if not o1==o2]):
        #     warn(f'objects in geometry_list are overlapping for object {name=}')
        if not all([isinstance(o, PolygonWI) for o in geometry_list]):
            raise ValueError(f'we require objects to be composed of PolygonWI but {[type(o) for o in geometry_list]=}')
        self.geometry_list = geometry_list
        boundary = union_all([g.geometry for g in geometry_list])
        z_min, z_max = min([g.z_min for g in geometry_list]), max([g.z_max for g in geometry_list])
        super().__init__(
            geometry=boundary,  # type: ignore
            material=None, 
            z_min=z_min, 
            z_max=z_max, 
            name=name
        )
        
    def get_all_materials(self) -> list[MaterialWI]:
        """
        Retrieves a list of all unique materials used in the geometry list.
        This method iterates through the `geometry_list` attribute, collects the 
        `material` property of each geometry object (if it is not `None`), and 
        returns a list of unique materials.
        Returns:
            list[MaterialWI]: A list of unique `MaterialWI` objects used in the 
            geometry list.
        """
        material_list = []
        for m in self.geometry_list:
            if m.material is not None and not any(m.material == mat for mat in material_list):
                material_list.append(m.material)
        return material_list
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ObjectWI):
            return equal_ignore_order(self.geometry_list, other.geometry_list)
        else: 
            return False

    def get_contained_geometries(self) -> Sequence[GeometryWI]:
        return [g for sg in self.geometry_list for g in sg.get_contained_geometries()]

class TxWI(GeometryWI[Point]):
    """
    TxWI is a subclass of GeometryWI that represents a transmitter with a specific position.

    Attributes:
        position (Sequence[float]): A sequence of three floats representing the (x, y, z) coordinates of the transmitter.
        name (str | None): An optional name for the transmitter.
    """
    def __init__(self,
            position : Sequence[float],
            name : str | None,
        ) -> None:
        if len(position) != 3:
            raise ValueError(f"Position must have exactly 3 elements, but got {len(position)}")
        super().__init__(
            geometry=Point(position[:2]), 
            material=None, 
            z_min=position[2], 
            z_max=position[2], 
            name=name
        )

class RxWI():
    '''
        Stores the parameters for an Rx grid. 
        Not the cleanest solution, but enough for our purposes. 
        Side lengths will be derived from the floorplan.

        Attributes:
            height (Sequence[float]): A sequence of three floats representing the (x, y, z) coordinates of the transmitter.
            name (str | None): An optional name for the transmitter.
    '''
    def __init__(self,
            height : float,
            spacing : float,
            name : str,
        ) -> None:
        self.height = height
        self.spacing = spacing
        self.name = name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RxWI):
            return {k : v for k, v in self.__dict__.items() if not k=='name'} == {k : v for k, v in other.__dict__.items() if not k=='name'}
        else: 
            return False
        
class ProjectWI:
    """
    A class representing a WI indoor project, which includes a floorplan, 
    geometrical objects, transmitters (Tx), and receiver (Rx) grids. This class provides 
    methods to manage and validate the addition of these components while ensuring 
    spatial constraints such as containment and non-overlapping geometries.
    Attributes:
        floorplan (FloorPlanWI): The floorplan of the project.
        objects (list[GeometryWI]): A list of general geometry objects in the project.
        tx (list[TxWI]): A list of transmitter objects in the project.
        rx (list[RxWI]): A list of receiver objects in the project.
        properties (dict[str, float]): simulation properties such as humidity, pressure, temperature.
    Methods:
        objects_to_check() -> list[GeometryWI]:
            Generates a combined list of geometries from `objects`, `tx`, and `rx` 
            to be used for intersection checks when adding more objects.
        add_geometry(geometry: GeometryWI) -> None:
            after performing necessary validations for overlap and containment.
        add_tx(*args):
            Placeholder method for adding transmitter objects. Not yet implemented.
        add_rx(*args):
            Placeholder method for adding receiver objects. Not yet implemented.
    """
    def __init__(self, 
            floorplan : FloorPlanWI,
            objects : list[ObjectWI] | None = None,
            tx : list[TxWI] | None = None,
            rx : list[RxWI] | None = None,
            properties : dict[str, float] = {
                'pressure' : 1013.25,
                'temperature' : 22.2,
                'humidity' : 50
            }
        ) -> None:
        """
        Args:
            floorplan (FloorPlanWI | None): The floor plan associated with the project. 
            objects (list[GeometryWI], optional): A list of geometric objects within the project. 
                Defaults to an empty list.
            tx (list[TxWI], optional): A list of transmitter objects. Defaults to an empty list.
            rx (list[RxWI], optional): A list of receiver objects. Defaults to an empty list.
        """
        self.floorplan = floorplan
        self.objects = objects if objects is not None else []
        self.tx = tx if tx is not None else []
        self.rx = rx if rx is not None else []
        self.properties = properties
    
    def objects_to_check(self) -> list[GeometryWI]:
        """
        Generates a list of geometries to check for intersections when adding a new geometries.
        Returns:
            list: A combined list of geometries from `objects` and `tx`.
        """
        geometries_list : list[GeometryWI] = [o for o in self.objects]
        geometries_list.extend(self.tx)
        return geometries_list
    
    def add_geometry(
            self, 
            geometry : GeometryWI,
        ) -> None:
        """
        Adds a geometry object to the appropriate list (tx, or objects) 
        after performing necessary validations.
        Parameters:
            geometry (GeometryWI): The geometry object to be added. It can be 
            of type TxWI, or other GeometryWI types.
        Raises:
            ValueError: If the geometry overlaps with any existing objects 
            (as determined by `check_no_overlap`) or if the geometry is not 
            contained within the floorplan (if a floorplan exists, as determined 
            by `check_contained`).
        Behavior:
            - If the geometry is of type TxWI, it is added to the `tx` list.
            - Otherwise, it is added to the `objects` list.
        """
        if not all([not check_overlap(o, geometry, touching_counts=False) for o in self.objects_to_check()]):
            raise ValueError(f'Cannot add geometry because it overlaps with other objects: {[o.name for o in self.objects_to_check() if check_overlap(o, geometry, touching_counts=False)]}')
        if self.floorplan is not None and not check_contains(geom1=self.floorplan, geom2=geometry):
            raise ValueError('Cannot add geometry because it is not contained inside the floorplan.')
        if isinstance(geometry, TxWI):
            self.tx.append(geometry)
        elif isinstance(geometry, ObjectWI):
            self.objects.append(geometry)
        else:
            raise ValueError(f'{type(geometry)=}')

    def add_rx(
            self,
            rx : RxWI
        ) -> None:
        """
        Add a receiver (RxWI instance) to the list of receivers.

        Args:
            rx (RxWI): The receiver instance to be added.

        Returns:
            None
        """
        self.rx.append(rx)

    def __eq__(self, other : Any, verbose : bool = False) -> bool:
        if isinstance(other, ProjectWI):
            if not verbose:
                return (self.floorplan == other.floorplan and equal_ignore_order(self.objects, other.objects) 
                    and equal_ignore_order(self.tx, other.tx) and equal_ignore_order(self.rx, self.rx) and self.properties == other.properties)
            else:
                if (self.floorplan == other.floorplan and equal_ignore_order(self.objects, other.objects) 
                    and equal_ignore_order(self.tx, other.tx) and equal_ignore_order(self.rx, self.rx) and self.properties == other.properties):
                    return True
                else:
                    print(f'projects are different because:\n\t {self.floorplan == other.floorplan = } \n\t {equal_ignore_order(self.objects, other.objects) =} \
                    \n\t {equal_ignore_order(self.tx, other.tx) = } \n\t {equal_ignore_order(self.rx, self.rx) = } \n\t {self.properties == other.properties =}')
                    return False
        else:
            return False

    def set_properties(self, **kwargs) -> None:
        self.properties = kwargs


def check_overlap_z(
        geom1 : GeometryWI, 
        geom2 : GeometryWI
    ) -> bool:
    """
    Checks whether two geometries overlap (at least in one height) in the Z-axis.
    Args:
        geom1 (GeometryWI): The first geometry with Z-axis range attributes `z_min` and `z_max`.
        geom2 (GeometryWI): The second geometry with Z-axis range attributes `z_min` and `z_max`.

    Returns:
        bool: `True` if the Z-axis ranges of the two geometries overlap, `False` otherwise.
    """
    return not (geom1.z_min > geom2.z_max or geom2.z_min > geom1.z_max)

def check_touching(
        geom1 : GeometryWI, 
        geom2 : GeometryWI
    ) -> bool:
    """
    Determines whether two geometries are touching based on their spatial and 
    vertical relationships.

    A touch is defined as either:
    1. The geometries touch in the XY plane and overlap in the Z dimension.
    2. The geometries are not disjoint in the XY plane and their Z dimensions 
        are adjacent (i.e., the minimum Z of one geometry equals the maximum Z 
        of the other).

    Args:
        geom1 (GeometryWI): The first geometry with spatial and Z-dimension properties.
        geom2 (GeometryWI): The second geometry with spatial and Z-dimension properties.

    Returns:
        bool: True if the geometries are touching based on the defined criteria, 
                False otherwise.
    """
    if geom1.geometry.touches(geom2.geometry) and check_overlap_z(geom1, geom2):
        return True
    elif not geom1.geometry.disjoint(geom2.geometry) and (geom1.z_min==geom2.z_max or geom1.z_max==geom2.z_min):
        return True
    else:
        return False


def check_overlap(
        geom1 : GeometryWI,
        geom2 : GeometryWI,
        touching_counts : bool
    ) -> bool:
    """
    Determines whether two geometries overlap in 3D space.

    Args:
        geom1 (GeometryWI): The first geometry with associated 3D spatial information.
        geom2 (GeometryWI): The second geometry with associated 3D spatial information.
        touching_counts (bool): If True, touching geometries are considered overlapping.

    Returns:
        bool: True if the geometries overlap (or at least touch, depending on `touching_counts`), 
              False otherwise.

    Notes:
        - Two geometries are considered non-overlapping if they are spatially disjoint 
          or if their z-coordinate ranges do not intersect.
        - If `touching_counts` is False, the function further checks if the geometries 
          are merely touching and excludes such cases from being considered overlapping.
    """
    if geom1.geometry.disjoint(geom2.geometry) or geom1.z_max < geom2.z_min or geom2.z_max < geom1.z_min:
        return False
    else:
        if touching_counts:
            return True
        else:
           return not check_touching(geom1, geom2)
    
def check_contains(
        geom1 : GeometryWI,
        geom2 : GeometryWI
    ) -> bool: 
    """
    Determines whether one geometry (geom1) fully contains another geometry (geom2),
    including both their 2D and vertical (z-axis) extents.

    Args:
        geom1 (GeometryWI): The first geometry object, which is checked for containing the second geometry.
        geom2 (GeometryWI): The second geometry object, which is checked for being contained within the first geometry.

    Returns:
        bool: True if geom1 fully contains geom2 in both 2D and vertical dimensions, 
              otherwise False.
    """
    if geom1.geometry.contains(geom2.geometry) and geom1.z_min <= geom2.z_min and geom1.z_max >= geom2.z_max:
        return True
    else: 
       return False

def check_equals_geometrically(
        geom1 : GeometryWI,
        geom2 : GeometryWI
    ) -> bool: 
    """
    Determines whether one geometry (geom1) equals another geometry (geom2) geometrically (material not checked),
    including both their 2D and vertical (z-axis) extents.

    Args:
        geom1 (GeometryWI): The first geometry object.
        geom2 (GeometryWI): The second geometry object.

    Returns:
        bool: True if geom1and geom2 are equal.
    """
    if (geom1.geometry.buffer(0.001).contains(geom2.geometry) and geom2.geometry.buffer(0.001).contains(geom1.geometry) 
        and np.isclose(geom1.z_min, geom2.z_min, 0, 0.001) and np.isclose(geom1.z_max, geom2.z_max, 0, 0.001)):
        return True
    else: 
        # print(f'{geom1.geometry.buffer(0.001).contains(geom2.geometry)=} \n \
        #       {geom2.geometry.buffer(0.001).contains(geom1.geometry)=} \n \
        #       {geom1.z_min == geom2.z_min=} \n \
        #       {geom1.z_max == geom2.z_max=}')
        return False

def check_boundary_contains(
        geom1 : GeometryWI,
        geom2 : GeometryWI
    ) -> bool: 
    """
    Checks if the boundary of the first geometry (geom1) contains the second geometry (geom2)
    and verifies that the z-coordinate range of geom1 encompasses that of geom2.

    Args:
        geom1 (GeometryWI): The first geometry.
        geom2 (GeometryWI): The second geometry object to check containment against.

    Returns:
        bool: True if geom1's boundary contains geom2 and geom1's z-coordinate range 
              fully encompasses geom2's z-coordinate range, otherwise False.
    """
    return (geom1.geometry.boundary.contains(geom2.geometry) 
        and geom1.z_min <= geom2.z_min 
        and geom1.z_max >= geom2.z_max
        )
        
def equal_ignore_order(a: Sequence, b: Sequence) -> bool:
    """
    Checks if two lists contain the same elements, ignoring order.
    Use only when elements are neither hashable nor sortable, as this method is very inefficient.

    Args:
        a (list): First list.
        b (list): Second list.

    Returns:
        bool: True if both lists contain the same elements, False otherwise.
    """
    return all([any([a1==b1 for b1 in b]) for a1 in a]) and all([any([a1==b1 for a1 in a]) for b1 in b])  


def check_is_part_of_boundary(
        geom1 : GeometryWI,
        geom2 : GeometryWI
    ) -> bool:
    """
    Checks if `geom2` is part of the boundary of `geom1`.

    This function verifies the following conditions:
    1. The boundary of `geom1` contains the geometry of `geom2`.
    2. The minimum or maximum z-coordinate of `geom1` matches that of `geom2`.

    Args:
        geom1 (GeometryWI): The first geometry object, representing the boundary.
        geom2 (GeometryWI): The second geometry object to check against the boundary.

    Returns:
        bool: True if `geom2` is part of the boundary of `geom1`, otherwise False.
    """
    return (geom1.geometry.boundary.contains(geom2.geometry)  
        and (geom1.z_min == geom2.z_min 
            or geom1.z_max == geom2.z_max)
        )


#Define colors for different object types, including metal plates
colors = {
    'chair': 'orange',
    'table': 'orange',
    # 'table_chair_group': 'orange',
    'cabinet': 'gray',
    'blackboard': 'black',
    'radiator': 'red',
    'lamp': 'tab:olive',
    'whiteboard': 'white',
    'frame' : 'tab:brown',
    'win': 'lightblue',
    'door': 'green',
    'cable' : 'tab:pink',
#     'metal_plate': 'blue',  # Added color for metal plates
    'metal': 'black',          # Added color for steel plates
    # 'aluminium': 'purple',     # Added color for aluminum plates
    # 'copper': 'yellow'        # Added color for copper plates
}

handles = [patches.Patch(color=c, label=k) for k, c in colors.items()]
handles.append(mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Tx'))

def get_color(geom : GeometryWI) -> str:
    """
    Returns the color associated with the geometry's name.

    Args:
        geom (GeometryWI): The geometry object.

    Returns:
        str: The color string.
    """
    for k, v in colors.items():
        if k in geom.name:
            return v
    print(f'no color defined for {geom.name}')
    return 'blue'


def plot_room(
        floorplan : FloorPlanWI, 
        object_list : list[ObjectWI], 
        tx_list : list[TxWI]
    )  -> Figure:
    """
    Plots the floorplan, objects, and transmitters in a room.

    Args:
        floorplan (FloorPlanWI): The floorplan object.
        object_list (list[ObjectWI]): List of object geometries.
        tx_list (list[TxWI]): List of transmitter geometries.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    fig = plt.figure(figsize=(20, 10))
    plotting.plot_polygon(floorplan.geometry)
    for g in floorplan.other:
        plotting.plot_line(g.geometry, color=get_color(g))
    for o in object_list:
        plotting.plot_polygon(o.geometry, color=get_color(o)) 
        for g in o.geometry_list:
            plotting.plot_polygon(g.geometry, color=get_color(o), linewidth=.1, add_points=False) 
    for t in tx_list:
        plotting.plot_points(t.geometry, color='red', marker='x')
    plt.legend(handles=handles)
    return fig

def plot_project(project : ProjectWI) -> Figure:
    """
    Plots the entire project including floorplan, objects, and transmitters.

    Args:
        project (ProjectWI): The project to plot.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    return plot_room(project.floorplan, project.objects, project.tx)

