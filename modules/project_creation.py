'''
In this file we define the logic for the creation of new projects.
The first step will be to generate a floorplan with random shape/sizes of the walls and add a door and windows.
After that, iteratively we can load objects and move them to some arbitrary positions, potentially rotate them. 
Doing this, we always have to check that they are contained inside the room and that they do not overlap with other objects, and some other conditions (radiatiors on/close to the wall, lamps under the ceiling...).
I suggest that we create some config file with all parameters in one place, e.g. min and max side lengths of the room, min/max number of chairs, ranges the parameter values for materials can take etc.
Then we can test different config files with varying parameters, which is less confusing and ensures reproducibility.
'''
from numpy.random import default_rng
from numbers import Number
from typing import Any, cast, TypeVar
from shapely import transform, Polygon, affinity, Point, LineString, LinearRing
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
from warnings import warn
import copy 
from shutil import rmtree
import yaml

from .utils import get_startswith_key, strip_suffix, round_to_significant_digits
from .read_wi_files import create_object_from_file, create_project_from_dir
from .project import ProjectWI, FloorPlanWI, ObjectWI, PolygonWI, MaterialWI, LineStringWI, GeometryWI, TxWI, RxWI, check_boundary_contains, check_contains
from .write_wi_files import project_to_files

# Type variable for generic geometry types
T = TypeVar('T', bound=GeometryWI)

# rng instance here or when calling create_project_randomly?
rng = default_rng()

def create_project_randomly(
        cfg : dict, 
        wi_files_dir : str | Path = './wi_project_files',
        verbose : bool = False
    ) -> tuple[ProjectWI,dict]:
    """
    Creates a project with a randomly generated floorplan and places objects within it.
    Args:
        cfg (dict): Configuration dictionary containing parameters for floorplan generation 
            and object placement. 
        wi_files_dir (str | Path, optional): Directory path where WI object files are stored. 
            Defaults to './wi_project_files'.
    Returns:
        ProjectWI: An instance of the ProjectWI class containing the generated floorplan 
            and placed objects.
    """
    cfg_mat = cfg.get('materials', {})
    cfg_here = copy.deepcopy(cfg)
    flp, cfg_flp = create_floorplan_randomly(cfg_flp=cfg['floorplan'], cfg_mat=cfg_mat, verbose=verbose)
    cfg_here['floorplan'] = cfg_flp

    cfg_pp = get_values(cfg['project_properties'])
    project = ProjectWI(floorplan=flp, properties=cfg_pp)
    cfg_here['project_properties'] = cfg_pp

    extrude_wall_objects(project=project, cfg_flp=cfg_flp)
    
    for k, cfg_obj in cfg['objects'].items():
        cfg_obj = load_and_place_objects(project=project, name=k, cfg_obj=cfg_obj, cfg_mat=cfg_mat, wi_files_dir=wi_files_dir, verbose=verbose)
        cfg_here['objects'][k] = cfg_obj

    if verbose:
        print(f'created project with {len(flp.other)} things inside the walls and {len(project.objects)} objects')

    cfg_tx = get_values(cfg['tx'])
    place_geometry(geom_orig=TxWI(position=[0,0,cfg_tx['z_min']], name='tx'), number=cfg_tx['number'], project=project, verbose=verbose)
    cfg_here['tx'] = cfg_tx

    cfg_rx = get_values(cfg['rx'])
    project.add_rx(rx=RxWI(name='rx_grid', **cfg_rx))
    cfg_here['rx'] = cfg_rx

    ### materials are defined for each object individually
    del cfg_here['materials']

    return project, cfg_here

def create_random_copies(
        project_dir : Path, 
        cfg : dict, 
        materials : bool, 
        project_properties : bool, 
        object_offset : float, 
        n_copies : int, 
        xml_glob : str, 
        out_dir : Path
    ) -> None:
    """
    Creates multiple randomized copies of a project directory.

    This function generates `n_copies` of a given project directory, applying randomizations to materials and/or project properties as specified. 
    Each copy is saved in a new directory with a unique name. Optionally, existing directories can be overwritten.

    Args:
        project_dir (Path): Path to the original project directory.
        cfg (dict): Configuration dictionary.
        materials (bool): Whether to randomize materials.
        project_properties (bool): Whether to randomize project properties like temperature.
        object_offset (float): Maximum spatial offset to apply to objects.
        object_rot (float): Maximum rotation to apply to objects.
        n_copies (int): Number of randomized copies to create.
        xml_glob (str): Glob pattern for XML files to process.
        out_dir (Path): Where to save the copies.

    Returns:
        None
    """
    if not project_properties and not materials and object_offset==0:
        raise ValueError(f'With the given parameters, we do not randomize anything')

    try:
        project = create_project_from_dir(project_dir=project_dir)
    except Exception as e:
        raise Exception(e, f"{project_dir=}")
    with open(project_dir / 'config_here.yml', 'r') as f:
        cfg_orig = yaml.safe_load(f)
    
    for n in range(n_copies):
        project_dir_copy = out_dir /f'{project_dir.stem}_copy{n}'
        if project_dir_copy.exists():
            continue
        for tries in range(10):
            try:
                project_copy, cfg_updated = randomize_existing_project(
                    project=project, 
                    materials=materials, 
                    project_properties=project_properties, 
                    cfg=cfg, 
                    object_offset=object_offset, 
                    cfg_orig=cfg_orig
                )
            except Exception as e:
                print(f'{str(project_dir)}:\t trying to create random copy {n}/{n_copies}, got Exception {e}')
                continue
            
            project_to_files(project=project_copy, project_dir=project_dir_copy, xml_glob=xml_glob, cfg_here=cfg_updated)
            #### safety check! we have some rare cases where generated environments cannot be loaded later
            try:
                p_new = create_project_from_dir(project_dir=project_dir_copy)
                break
            except Exception as e:
                rmtree(project_dir_copy)
                if tries == 9:
                    raise RuntimeError(f'Could not create a valid copy in 10 tries for {project_dir=}')
                continue

def randomize_existing_project(
        project : ProjectWI, 
        materials : bool, 
        project_properties : bool, 
        cfg : dict, 
        object_offset : float, 
        cfg_orig : dict
    ) -> tuple[ProjectWI,dict]:
    """
    Randomizes aspects of an existing ProjectWI instance based on provided options.

    Args:
        project (ProjectWI): The project instance to randomize.
        materials (bool): If True, randomizes materials for the floorplan and objects.
        project_properties (bool): If True, randomizes project-level properties.
        cfg (dict): Configuration dictionary containing randomization settings for materials and project properties.
        object_offset (float): Offset value for randomizing object positions (not implemented).
        object_rot (float): Rotation value for randomizing object orientations (not implemented).
        cfg_orig (dict): Config generated when saving the original project. It will be copied and updated with e.g. new material properties.

    Returns:
        ProjectWI: The randomized project instance.
        dict : Config dict of changed parameters

    Raises:
        NotImplementedError: If object_offset or object_rot is non-zero, as position and rotation randomization is not implemented.
    """
    cfg_new = copy.deepcopy(cfg_orig)
    if materials:
        project, cfg_new = randomize_project_materials(cfg=cfg, cfg_new=cfg_new, project=project)
    if project_properties:
        project, cfg_new = randomize_project_properties(cfg=cfg, cfg_new=cfg_new, project=project)
    if object_offset != 0:
        project = randomize_project_object_positions(cfg=cfg, project=project, object_offset=object_offset)
    return project, cfg_new

def randomize_project_materials(
        cfg: dict,
        cfg_new: dict,
        project: ProjectWI,
    ) -> tuple[ProjectWI, dict[Any, Any]]:
    """
    Randomizes and assigns materials to the components of a project based on configuration dictionaries.

    This function updates the materials for the floorplan, wall objects, and standard objects in the given
    `project` according to the specifications in `cfg` and `cfg_new`. It also updates the configuration
    dictionaries to reflect the assigned materials.

    Args:
        cfg (dict): The original configuration dictionary containing material and object specifications.
        cfg_new (dict): The configuration dictionary to be updated with new material assignments.
        project (ProjectWI): The project instance whose components' materials will be randomized and assigned.

    Returns:
        tuple[ProjectWI, dict[Any, Any]]: 
            - The updated project instance with assigned materials.
            - The updated configuration dictionary reflecting the new material assignments.

    Raises:
        ValueError: If a matching object for a wall object cannot be found.
        RuntimeError: If a material listed for an object is not defined in the configuration, or if a geometry
                        within an object does not have a material assigned.
    """
    ### flp materials
    cfg_flp = get_values(cfg['floorplan'])
    mat_flp_name, mat_flp_walls_name = cfg_flp['material'], cfg_flp['material_walls']
    cfg_mat_flp = get_values(cfg['materials'])
    project.floorplan.material = MaterialWI(name=mat_flp_name, **cfg_mat_flp[mat_flp_name])
    project.floorplan.material_walls = MaterialWI(name=mat_flp_walls_name, **cfg_mat_flp[mat_flp_walls_name])
    # cfg_flp = insert_material_into_cfg(cfg_flp, cfg_mat_flp, [mat_flp_name, mat_flp_walls_name])
    cfg_flp = insert_material_into_cfg(cfg_flp, cfg_mat_flp, [])
    for k in ['material', 'material_walls', mat_flp_name, mat_flp_walls_name]:
        cfg_new['floorplan'][k] = cfg_flp[k]
    ### wall objects
    for wo, cfg_wobj in cfg['floorplan']['wall_objects'].items():
        individually = cfg_wobj.get('randomize_individually', False)
        if individually:
            obj_names = [k for k in cfg_new['floorplan']['wall_objects'].keys() if strip_suffix(k)==(wo)]
            flp_objects = project.floorplan.other
            for obj_name in obj_names:
                cfg_wobj_it = get_values(cfg_wobj)
                mat_wobj_name = cfg_wobj_it['material']
                cfg_new['floorplan']['wall_objects'][obj_name]['material'] = mat_wobj_name
                cfg_mat_wobj_it = get_values(cfg['materials'])
                wobj_mat = MaterialWI(name=mat_wobj_name, **cfg_mat_wobj_it[mat_wobj_name])
                # find the corresponding object
                flp_objects_matching = [o for o in flp_objects if o.name==obj_name]
                if len(flp_objects_matching) != 1:
                    raise ValueError(f'looking for {obj_name=} found {len(flp_objects_matching)=}, because this list only contains:\n{[o.name for o in flp_objects]}')
                flp_objects_matching[0].material = wobj_mat
                try:
                    # cfg_new['floorplan']['wall_objects'][obj_name] = insert_material_into_cfg(cfg_new['floorplan']['wall_objects'][obj_name], cfg_mat_wobj_it, materials_used=[])#[mat_wobj_name])
                    cfg_new['floorplan']['wall_objects'][obj_name][mat_wobj_name] = cfg_mat_wobj_it[mat_wobj_name]

                except KeyError as e:
                    raise KeyError(f'randomize_project_materials: KeyError {e}')
        else:
            mat_wobj_name = get_values(cfg_wobj)['material']
            cfg_mat_wobj_it = get_values(cfg['materials'])
            wobj_mat = MaterialWI(name=mat_wobj_name, **cfg_mat_wobj_it[mat_wobj_name])
            for wobj in project.floorplan.other + project.objects:
                if wo == strip_suffix(wobj.name):
                    wobj.material = wobj_mat
            try:
                cfg_new['floorplan']['wall_objects'][wo] = insert_material_into_cfg(cfg_new['floorplan']['wall_objects'][wo], cfg_mat_wobj_it, materials_used=[])#[mat_wobj_name])
            except KeyError as e:
                    raise KeyError(f'randomize_project_materials: KeyError {e}')
    ### now the same for standard objects, recall that the materials for these are not stored in the original config for creating the projects but only in the one for saved parameters
    for wo, cfg_obj in cfg_new['objects'].items():
        mats_obj_name = [v for k, v in cfg_obj.items() if k.startswith('material')]
        for m in mats_obj_name:
            if not m in cfg_obj.keys():
                raise RuntimeError(f'{m=} was listed as material but not defined in {cfg_obj=} for object {wo=}')
        cfg_mat_obj = get_values(cfg['materials'])
        obj_mats = {mat_obj_name : MaterialWI(name=mat_obj_name, **cfg_mat_obj[mat_obj_name]) for mat_obj_name in mats_obj_name}
        ### assign materials 
        for obj in project.objects:
            if obj.name.startswith(wo):
                for geom in obj.geometry_list:
                    if geom.material is None:
                        raise RuntimeError(f'{obj.name=} contains {geom.name=} with {geom.material=}')
                    try:
                        geom.material = obj_mats[geom.material.name]
                    except KeyError as e:
                        raise KeyError(e)
        ### update config
        try:
            cfg_new['objects'][wo] = insert_material_into_cfg(cfg_new['objects'][wo], cfg_mat_obj, [])
        except KeyError as e:
            raise KeyError(f'randomize_project_materials: KeyError {e}')
    return project, cfg_new

def randomize_project_properties(
        cfg : dict, 
        cfg_new : dict, 
        project : ProjectWI
    ) -> tuple[ProjectWI, dict[Any, Any]]:
    """
    Randomizes and sets properties for a given project based on configuration dictionaries.

    Args:
        cfg (dict): The original configuration dictionary containing project properties.
        cfg_new (dict): A configuration dictionary to be updated with new project properties.
        project (ProjectWI): The project instance to update with randomized properties.

    Returns:
        tuple[ProjectWI, dict[Any, Any]]: A tuple containing the updated project instance and the updated configuration dictionary.
    """
    proj_prop = get_values(cfg['project_properties'])
    project.set_properties(**proj_prop)
    cfg_new['project_properties'] = proj_prop
    return project, cfg_new

def randomize_project_object_positions(
        cfg : dict, 
        project : ProjectWI, 
        object_offset : float
    ) -> ProjectWI:
    ### loop over all objects and Tx in the project, try for 10 times to draw a random number from [0, object_offset] and a random angle in [0, 2*np.pi] and move the object
    cwoet, cwoem, com, cot, cwoft, cwofm, ctt, ctm = 0, 0, 0, 0, 0, 0, 0, 0

    for obj in list(project.objects):
        ### check whether the object is an extruded wall_object or a "free standing" one
        is_wall_object = any(obj.name.startswith(n) for n in cfg["floorplan"]["wall_objects"])
        # Create a deep copy to preserve original state for fallback
        if is_wall_object:
            cwoet += 1
            containing_wall = find_touching_wall(floorplan=project.floorplan, wall_object=obj)
            direction_vector_normalized = calculate_direction_vector_wall(wall=containing_wall)
            for _ in range(100):
                direction_vector = direction_vector_normalized * object_offset * (rng.random() - 0.5) * 2
                try:
                    obj_moved = move_geometry_by(obj, direction_vector)
                    try:
                        # check that the wall object is still adjacent to a wall!
                        find_touching_wall(floorplan=project.floorplan, wall_object=obj_moved)
                    except:
                        continue
                    # Temporarily remove the original object to avoid self-overlap
                    project.objects.remove(obj)
                    project.add_geometry(obj_moved)
                    cwoem += 1
                    break
                except:
                    ### if the moved object cannot be added, we readd the original one
                    try:
                        project.add_geometry(obj)
                    except Exception as e:
                        raise RuntimeError(f'extruded wall object fail, because {e}')
                    continue
        else:
            if not any(obj.name.startswith(n) for n in cfg["objects"]):
                raise RuntimeError(f"could not find {obj.name=} in {cfg=}")
            cot += 1
            for _ in range(100):
                offset = rng.random() * object_offset
                angle = rng.random() * 2 * np.pi
                dx = offset * np.cos(angle)
                dy = offset * np.sin(angle)
                obj_moved = move_geometry_by(obj, (dx, dy))
                try:
                    project.objects.remove(obj)
                    project.add_geometry(obj_moved)
                    com += 1
                    break
                except:
                    try:
                        project.add_geometry(obj)
                    except Exception as e:
                        raise RuntimeError(f'object fail because {e}')
                    continue

    for line_obj in list(project.floorplan.other):
        if not any(line_obj.name.startswith(n) for n in cfg["floorplan"]["wall_objects"]) or "frame" in line_obj.name:
            continue
        cwoft += 1
        containing_wall = find_containing_wall(floorplan=project.floorplan, wall_object=line_obj)
        direction_vector_normalized = calculate_direction_vector_wall(wall=containing_wall)
        corresponding_frames = find_corresponding_frames(obj_list=project.floorplan.other, wall_object=line_obj)
        for _ in range(100):
            direction_vector = direction_vector_normalized * object_offset * (rng.random() - 0.5) * 2
            try:
                obj_moved = move_geometry_by(line_obj, direction_vector)
                frames_moved = [move_geometry_by(frame, direction_vector) for frame in corresponding_frames]
                # Temporarily remove the original object to avoid self-overlap
                for obj_f in [line_obj] + corresponding_frames:
                    project.floorplan.other.remove(obj_f)
                project.floorplan.add_others([obj_moved] + frames_moved)
                cwofm += 1
                break
            except:
                ### if the moved object cannot be added, we readd the original one
                try:
                    project.floorplan.add_others([line_obj] + corresponding_frames)
                except Exception as e:
                    raise Exception(f'linestring wall objects fail because {e}')
                continue

    for tx in list(project.tx):
        ctt += 1
        for _ in range(100):
            offset = rng.random() * object_offset
            angle = rng.random() * 2 * np.pi
            dx = offset * np.cos(angle)
            dy = offset * np.sin(angle)
            tx_moved = move_geometry_by(tx, (dx, dy))
            try:
                # Temporarily remove the original Tx to avoid self-overlap
                project.tx.remove(tx)
                project.add_geometry(tx_moved)
                ctm += 1
                break
            except Exception:
                # Re-add the original Tx if placement failed
                try:
                    project.add_geometry(tx)
                except:
                    raise Exception('tx fail')
                continue
    
    # print(f'successfully moved {ctm}/{ctt} Tx, {com}/{cot} objects, {cwoem}/{cwoet} extruded wall objects and {cwofm}/{cwoft} flat wall objects')
    return project

def find_containing_wall(
        floorplan : FloorPlanWI, 
        wall_object : LineStringWI
    ) -> LineStringWI:
    """
    Finds and returns the wall segment within a floorplan that contains the specified wall object.

    Args:
        floorplan (FloorPlanWI): The floorplan object containing wall segments.
        wall_object (ObjectWI): The wall object to locate within the floorplan.

    Returns:
        LineStringWI: The wall segment that contains the wall object.

    Raises:
        ValueError: If the geometry of the wall object is not a LineString.
        RuntimeError: If no containing wall segment is found for the wall object.
    """
    if not isinstance(wall_object.geometry, LineString):
        raise ValueError(f'Only flat wall objects can be contained in walls but {type(wall_object.geometry)=}')
    wall_segments = PolygonWI.split(floorplan)
    containing_wall = None
    for ws in wall_segments:
        if check_contains(ws, wall_object):
            containing_wall = ws
            break
    if containing_wall is None:
        raise RuntimeError(f'Could not find a wall containing {wall_object.name}. Wall coordinates: {list(floorplan.geometry.exterior.coords)}, wall_object coordinates: {list(wall_object.geometry.coords)}')
    return containing_wall

def find_touching_wall(
        floorplan : FloorPlanWI, 
        wall_object : ObjectWI
    ) -> LineStringWI:
    """
    Finds and returns the segment of the floorplan wall that is touching the given wall object.

    Args:
        floorplan (FloorPlanWI): The floorplan object containing the walls.
        wall_object (PolygonWI): The wall object to check for touching segments.

    Returns:
        LineStringWI: The segment of the floorplan wall that touches the wall object.

    Raises:
        RuntimeError: If no touching wall segment is found.

    """
    wall_segments = PolygonWI.split(floorplan)
    obj_side_segments = wall_object.split()
    touching_wall = None
    for ws in wall_segments:
        if any(check_contains(ws, oss) for oss in obj_side_segments):
            touching_wall = ws
            break
    if touching_wall is None:
        raise RuntimeError(f'Could not find a wall touching {wall_object.name}. Wall coordinates: {list(floorplan.geometry.exterior.coords)}, wall_object coordinates: {list(wall_object.geometry.exterior.coords)}')
    return touching_wall

def calculate_direction_vector_wall(wall : LineStringWI) -> np.ndarray:
    """
    Calculates the normalized direction vector of a wall represented by a LineStringWI object.

    Parameters:
    wall (LineStringWI): The wall object containing geometry information.

    Returns:
    np.ndarray: A unit vector representing the direction of the wall.
    """
    wall_coords = np.array(wall.geometry.coords)
    wall_direction = wall_coords[1] - wall_coords[0]
    wall_direction = wall_direction / np.linalg.norm(wall_direction)
    return wall_direction

def find_corresponding_frames(
        obj_list : list[LineStringWI], 
        wall_object : LineStringWI
    ) -> list[LineStringWI]:
    """
    Finds and returns a list of frame objects from obj_list that correspond to the given wall_object.

    Args:
        obj_list (list[LineStringWI]): A list of LineStringWI objects to search through.
        wall_object (LineStringWI): The wall object to match frames against.

    Returns:
        list[LineStringWI]: A list of LineStringWI objects whose names start with the wall_object's name and contain "frame".
    """
    return [o for o in obj_list if o.name.startswith(wall_object.name) and "frame" in o.name]

def extrude_wall_objects(
        project : ProjectWI,    
        cfg_flp : dict
    ) -> None:
    """
    Extrudes wall objects (e.g., windows, doors) into 3D polygons and adds them to the project.

    Args:
        project (ProjectWI): The project containing the floorplan.
        cfg_flp (dict): Configuration dictionary for the floorplan, including wall object parameters.

    Returns:
        None: Modifies the project in place by adding extruded wall objects.

    Notes:
        - Wall objects are identified by their names in the floorplan's `other` attribute.
        - Objects are extruded based on their depth specified in the configuration.
    """
    ids_to_be_removed = []
    for wall_obj_name, wall_obj_cfg in cfg_flp.get('wall_objects', {}).items():
        wall_obj_cfg = get_values(wall_obj_cfg)
        obj_depth = wall_obj_cfg.get('depth', 0)
        if obj_depth == 0:
            continue
        for idl, line_string in enumerate(project.floorplan.other):
            if line_string.name.startswith(wall_obj_name):
                ids_to_be_removed.append(idl)
                obj_polygon = extrude_wall_object(line_string=line_string, obj_depth=obj_depth)
                try:
                    project.add_geometry(obj_polygon)
                except Exception as e:
                    print(f'failed to extrude {line_string.name} to {np.asarray(obj_polygon.geometry.boundary.coords)} because {e}')
                    continue
    project.floorplan.other = [l for idl, l in enumerate(project.floorplan.other) if not idl in ids_to_be_removed]

def extrude_wall_object(
        line_string : LineStringWI,
        obj_depth : float
    ) -> ObjectWI:
    """
    Extrudes a wall object (represented as a LineStringWI) into a 3D polygon with the specified depth.

    Args:
        line_string (LineStringWI): The wall object as a line string to extrude.
        obj_depth (float): The depth to extrude the object.

    Returns:
        ObjectWI: The extruded wall object as a polygon.
    """
    c = np.asarray(line_string.geometry.coords)
    if c[0,0]==c[1,0]==0:
        off = np.array([obj_depth, 0])
    elif c[0,0]==c[1,0]:
        off = np.array([-obj_depth, 0])
    elif c[0,1]==c[1,1]==0:
        off = np.array([0, obj_depth])
    elif c[0,1]==c[1,1]:
        off = np.array([0, -obj_depth])
    else:
        raise ValueError(f'wall not parallel to any coordinate axis? coordinates: {c=}')
    c_new = (c[0,:], c[0,:] + off, c[1,:] + off, c[1, :])
    poly = Polygon(c_new)
    poly.normalize()

    return ObjectWI([PolygonWI(
                poly, 
                material=line_string.material, 
                z_min=line_string.z_min, 
                z_max=line_string.z_max, 
                name=f'{line_string.name}_poly')], 
            name=line_string.name)

def place_wall_objects(
        floorplan : FloorPlanWI,
        name : str, 
        cfg_obj : dict,
        cfg_mat : dict,
        verbose : bool
    ) -> dict:
    """
    Generates a random line segment of a specified length within the walls of a floorplan.

    Args:
        floorplan (FloorPlanWI): The floorplan containing the walls.
        side_length (float): The desired length of the line segment.

    Returns:
        LineString | None: A random line segment if a suitable wall is found, or None if no wall meets the length requirement.

    Notes:
        - The function selects a random wall and generates a line segment within it.
        - If no wall is long enough, a warning is issued, and None is returned.
    """
    if isinstance(cfg_obj['side_length'], list):
        side_length_min = cfg_obj['side_length'][0]
    else:
        side_length_min = cfg_obj['side_length']
    

    if not cfg_obj.get('randomize_individually', False):
        cfg_mat = get_values(cfg_mat)
        cfg_obj = get_values(cfg_obj)
        material = MaterialWI(name=cfg_obj['material'], **cfg_mat[cfg_obj['material']])
    
    n_tries = 100
    n_obj_desired = get_values(cfg_obj)['number']
    successes = 0
    frames = 0
    cfg_obj_return = {}
    
    for n_obj in range(n_obj_desired):
        placed = False
        for n in range(n_tries):
            cfg_obj_it = get_values(cfg_obj)
            cfg_mat_it = get_values(cfg_mat)
            if cfg_obj.get('randomize_individually', False):
                material = MaterialWI(name=cfg_obj_it['material'], **cfg_mat_it[cfg_obj_it['material']])
            z_min, z_max, side_length_max = cfg_obj_it['z_min'], cfg_obj_it['z_max'], cfg_obj_it['side_length']
            if not z_min >= 0 and z_max <= floorplan.z_max:
                raise ValueError(f'{z_min=}, {z_max=} for {name}, but {floorplan.z_min=}, {floorplan.z_max=}')
            if z_max <= z_min:
                continue
            for k in range(5):
                side_length = side_length_max - k * (side_length_max - side_length_min) / 5
                line = get_random_line_in_wall(floorplan=floorplan, side_length=side_length)
                if line is None:
                    continue
                try:
                    wall_obj = LineStringWI(geometry=line, material=material, z_min=z_min, z_max=z_max, name=f'{name}{n_obj}')
                    floorplan.add_others(wall_obj)
                    successes += 1
                    placed = True
                except ValueError:
                    continue
                if 'frame' in cfg_obj_it.keys():
                    frames += add_potential_frame(floorplan=floorplan, wall_obj=wall_obj, cfg_frame=cfg_obj_it['frame'], cfg_mat=cfg_mat)
                break
            if placed:
                if cfg_obj.get('randomize_individually', False):
                    cfg_obj_return[f'{name}{n_obj}'] = {k : v for k, v in insert_material_into_cfg(cfg_obj_it, cfg_mat=cfg_mat_it, materials_used=[]).items() if not k in ['randomize_individually', 'number']}
                break
        if successes > n_obj_desired:
            raise RuntimeError(f'{successes=} > {n_obj_desired=}')
    if not cfg_obj.get('randomize_individually', False):
        cfg_obj_return[name] = insert_material_into_cfg(cfg_obj, cfg_mat=cfg_mat, materials_used=[])
    if verbose:
        print(f'managed to place {successes} / {n_obj_desired} instances of {name} in walls with {frames} frames')
    return cfg_obj_return

def add_potential_frame(
        floorplan : FloorPlanWI, 
        wall_obj : LineStringWI, 
        cfg_frame : dict, 
        cfg_mat : dict
    ) -> bool:
    """
    Attempts to add a potential frame around an object inside of a wall to a floorplan, based on the given wall object and configuration.

    Parameters:
        floorplan (FloorPlanWI): The floorplan object to which the frame will be added.
        wall_obj (LineStringWI): The object inside the wall represented as a LineStringWI instance.
        cfg_frame (dict): Configuration dictionary for the frame, containing:
            - 'chance' (float): Probability of adding the frame (0 to 1).
            - 'side_length' (float): Length of the frame's side/width.
            - 'material' (str): Name of the material to use for the frame.
        cfg_mat (dict): Configuration dictionary for materials, where keys are material names and values are their properties.

    Returns:
        bool: True if the frame was tried to add, False otherwise.

    Notes:
        - The function generates four LineStringWI objects (two sides, top, bottom) to represent the frame.
        - Each LineStringWI object is created with the specified material and dimensions.
        - The function attempts to add these LineStringWI objects to the floorplan using `floorplan.add_others`.
        - If an exception occurs during the addition of any LineStringWI object, it is caught and logged, and the function continues with the next object.
    """
    random_number = rng.random()
    if random_number > cfg_frame['chance']:
        return False
    side_length = cfg_frame['side_length']
    line_start, line_end = np.asarray(wall_obj.geometry.coords[0]), np.asarray(wall_obj.geometry.coords[1])
    line_direction = line_end - line_start
    line_direction = line_direction / np.linalg.norm(line_direction)
    side0 = LineStringWI(
        geometry=LineString([line_start - side_length * line_direction, line_start]), 
        z_min=wall_obj.z_min,
        z_max=wall_obj.z_max,
        name=f'{wall_obj.name}_frame0_{cfg_frame["material"]}',
        material = MaterialWI(name=cfg_frame['material'], **get_values(cfg_mat[cfg_frame['material']]))
    )
    side1 = LineStringWI(
        geometry=LineString([line_end, line_end + side_length * line_direction]), 
        z_min=wall_obj.z_min,
        z_max=wall_obj.z_max,
        name=f'{wall_obj.name}_frame1_{cfg_frame["material"]}',
        material = MaterialWI(name=cfg_frame['material'], **get_values(cfg_mat[cfg_frame['material']]))
    )
    top = LineStringWI(
        geometry=LineString([line_start - side_length * line_direction, line_end + side_length * line_direction]), 
        z_min=wall_obj.z_max,
        z_max=wall_obj.z_max + side_length,
        name=f'{wall_obj.name}_frame_top_{cfg_frame["material"]}',
        material = MaterialWI(name=cfg_frame['material'], **get_values(cfg_mat[cfg_frame['material']]))
    )
    bottom = LineStringWI(
        geometry=LineString([line_start - side_length * line_direction, line_end + side_length * line_direction]), 
        z_min=wall_obj.z_min - side_length,
        z_max=wall_obj.z_min,
        name=f'{wall_obj.name}_frame_bottom_{cfg_frame["material"]}',
        material = MaterialWI(name=cfg_frame['material'], **get_values(cfg_mat[cfg_frame['material']]))
    )
    return_value = False
    for l in [side0, side1, top, bottom]:
        try:
            floorplan.add_others(l)
            return_value = True
        except Exception as e:
            continue
    return return_value

def get_random_line_in_wall(
        floorplan : FloorPlanWI, 
        side_length : float
    ) -> LineString | None:
    """
    Selects a random line segment of a specified length along a wall in the given floorplan.

    Args:
        floorplan (FloorPlanWI): The floorplan object containing the geometry of the space.
        side_length (float): The required length of the line segment to be selected.

    Returns:
        LineString | None: A LineString representing a segment of the wall with the specified length,
        or None if no wall is long enough.

    Notes:
        - The function randomly selects a wall segment of at least the required length and returns a random
            sub-segment of the specified length along that wall.
        - If no wall meets the length requirement, a warning is issued and None is returned.
    """
    walls = ring_to_segments(floorplan.geometry.boundary) # type: ignore
    walls = [w for w in walls if w.length >= side_length]
    if len(walls) == 0:
        warn(f'no wall of the floorplan has required length {side_length}')
        return None
    w = walls[rng.integers(0, len(walls))]
    start_rel = rng.random() * (w.length - side_length)
    wall_start, wall_end = np.array(w.coords[0]), np.array(w.coords[1])
    dir = wall_end - wall_start
    dir = dir / np.linalg.norm(dir)
    return LineString((wall_start + start_rel * dir, wall_start + (start_rel + side_length) * dir))
    
def load_and_place_objects(
        project : ProjectWI, 
        name : str, 
        cfg_obj : dict, 
        cfg_mat : dict, 
        wi_files_dir : str | Path,
        verbose : bool
    ) -> dict:
    """
    Loads an object from a file, places it in a random allowed position within the project, 
    and applies random rotations and material randomization.
    Args:
        project (ProjectWI): The project instance where the object will be placed.
        name (str): The name of the object to be loaded and placed.
        cfg (dict): Configuration dictionary containing object placement rules and parameters.
        wi_files_dir (str | Path): Directory path where the object files are located.
    Returns:
        None - inplace modification of project
    Behavior:
        - Attempts to load the object file corresponding to the given name.
        - If loading fails, logs the exception and exits the function.
        - Retrieves configuration values for the object from the `cfg` dictionary.
        - For the specified number of instances (`cfg['objects'][name]['number']`):
            - Assigns a unique name to each instance.
            - Applies random rotations (0, 90, 180, 270 degrees).
            - Moves the object to a random allowed position based on the `where` parameter.
            - If the object is successfully moved, randomizes its materials using the configuration.
    """
    obj_orig = create_object_from_file(Path(wi_files_dir) / f'{name}.object')
    # we want to randomize materials, but in a way that is consistent across all objects of this type
    cfg_mat = get_values(cfg_mat)
    materials_used = randomize_materials(obj_orig, cfg_mat) 
    cfg_obj = get_values(cfg_obj)
    place_geometry(
        geom_orig=obj_orig, 
        number=cfg_obj['number'], 
        project=project,
        verbose=verbose
    )
    return insert_material_into_cfg(cfg_obj, cfg_mat=cfg_mat, materials_used=materials_used)
    
def place_geometry(
        geom_orig : GeometryWI,
        number : int,
        project : ProjectWI,
        verbose : bool
    ) -> None:
    """
    Places multiple instances of a given geometry object at random allowed positions within a project.

    Each instance is optionally rotated by a random multiple of 90 degrees before placement.
    The function attempts to place the specified number of instances and reports how many were successfully placed.

    Args:
        geom_orig (GeometryWI): The original geometry object to be placed.
        number (int): The number of instances to attempt to place.
        project (ProjectWI): The project within which to place the geometry instances.

    Returns:
        None
    """
    # move the object to a random position that is allowed, as often as determined by the parameter number
    successes = 0
    for k in range(number):
        geom = rotate_geometry(geom=geom_orig, k=rng.integers(0, 4))
        geom.name = f'{geom_orig.name}{k+1}'
        moved_obj = move_geometry_to_random_position(project=project, geom=geom)
        if moved_obj is not None:
            successes += 1
    if verbose:
        print(f'managed to place {successes} / {number} instances of {geom_orig.name}')

def randomize_materials(
        thing : GeometryWI, 
        cfg : dict
    ) -> list[str]:
    """
    Randomizes the materials of the geometries in the given object based on the provided configuration.
    Args:
        thing (GeometryWI): The geometry whos materials will be randomized. May also be a FloorplanWI or ObjectWI, in that case also materials of contained geometries are randomized.
        cfg (dict): A dictionary containing material configuration. It should include a 'materials' key
                    with randomized or fixed parameter values for each material or it should already be
                    the dict taken from the 'materials' entry of the main config.
    Returns:
        None: This function modifies the materials of the things's geometries in place.
    """
    # use the get_values function to draw random parameters
    cfg_mat = get_values(cfg['materials']) if 'materials' in cfg.keys() else get_values(cfg)

    materials_used = []
    for m in thing.get_all_materials():
        if m is not None:
            m_new = MaterialWI(name=m.name, **cast(dict, get_startswith_key(cfg_mat, m.name)))
            materials_used.append(m.name)
            if thing.material == m:
                    thing.material = m_new
            if isinstance(thing, ObjectWI):
                for g in thing.geometry_list:
                    # recreate the material with the now randomized values
                    if g.material == m:
                        g.material = m_new
            elif isinstance(thing, FloorPlanWI):
                for o in thing.other:
                    if o.material == m:
                        o.material = m
                if thing.material_walls == m:
                    thing.material_walls = m_new
    return materials_used

def rotate_geometry(
        geom : T, 
        k : int
    ) -> T:
    """
    Rotates the geometry of an ObjectWI instance by 90 degrees multiplied by k.

    Args:
        object (ObjectWI): The object containing geometries to be rotated.
        k (int): The number of 90-degree rotations to apply. Positive values 
                    rotate counterclockwise, and negative values rotate clockwise.

    Returns:
        ObjectWI: A new ObjectWI instance with the rotated geometries.

    Notes:
        - The rotation is performed around the centroid of the object's geometry.
        - Each geometry in the object's geometry list is individually rotated.
        - The material, z_min, z_max, and name properties of each geometry are preserved.
    """
    if k % 4==0 or isinstance(geom, TxWI):
        return copy.deepcopy(geom)
    centroid = geom.geometry.centroid
    if isinstance(geom, ObjectWI):
        geometries_rotated = [
            PolygonWI(
                affinity.rotate(copy.deepcopy(g.geometry), k * 90, centroid), 
                material=g.material, 
                z_min=g.z_min, 
                z_max=g.z_max, 
                name=g.name
                ) 
            for g in geom.geometry_list
        ]
        return cast(T, ObjectWI(geometry_list=geometries_rotated, name=geom.name))
    else:
        geom_copy = copy.deepcopy(geom)
        geom_copy.geometry = affinity.rotate(geom_copy.geometry, k * 90, centroid)
        return geom_copy

def move_geometry_to_random_position(
        project : ProjectWI, 
        geom : T, 
    ) -> T | None:
    """
    Attempts to move an object to a random position within a project based on the specified location type.
    Parameters:
        project (ProjectWI): The project containing the floorplan and geometry where the object will be moved.
        object (ObjectWI): The object to be moved to a new position.
    Returns:
        ObjectWI | None: The moved object if successfully placed, or None if placement fails after multiple attempts.
    Notes:
        - The function tries up to `n_tries` random positions within the bounding box.
        - The function assumes a rectangular bounding box for floorplans, which may not match the actual geometry.
        - Objects that fail to be placed after `n_tries` attempts will not be added to the project.
    """
    n_tries = 100
    x_max, y_max = max([c[0] for c in project.floorplan.geometry.exterior.coords]), max([c[1] for c in project.floorplan.geometry.exterior.coords]) # type: ignore
    for k in range(n_tries):
        x, y = round(rng.random() * x_max, 3), round(rng.random() * y_max, 3)
        geom_moved = move_geometry_to(geom, (x,y))
        try:
            project.add_geometry(geom_moved)
        except Exception as e:
            continue
        return geom_moved
    return None

def create_floorplan_randomly(
        cfg_flp : dict,
        cfg_mat : dict,
        verbose : bool
    ) -> tuple[FloorPlanWI,dict]:
    """
    Creates a random floorplan based on the provided configuration.
    Args:
        cfg (dict): A dictionary containing configuration parameters for the floorplan and materials.
    Returns:
        FloorPlanWI: An instance of the FloorPlanWI class representing the generated floorplan.
    Notes:
        - The function generates random parameters for the floorplan dimensions and materials.
        - The floorplan geometry is defined as a polygon with a cutoff in the x and y dimensions.
        - The creation of windows and doors is currently marked as a TODO.
    """
    params = get_values(cfg_flp)
    params_mat = get_values(cfg_mat)
    # draw random paramters again, so we get different values of side length, cutoff for y
    params_y_flp = get_values(cfg_flp) 
    material_walls = MaterialWI(name=params['material_walls'], **params_mat[params['material_walls']])
    mat_flp_name = params['material']
    if mat_flp_name == material_walls.name:
        material_flp = material_walls
    else:
        material_flp = MaterialWI(name=mat_flp_name, **params_mat[params['material']])
    x_max, y_max = round(params['side_length'], 3), round(params_y_flp['side_length'], 3)
    if params.get('cutoff_chance', 1) < rng.random():
        if params['cutoff_length'] > 0:
            x_cut, y_cut = round(x_max * params['cutoff_length'], 3), round(y_max * params_y_flp['cutoff_length'], 3) 
        else:
            params['cutoff_length'] = params_y_flp['cutoff_length'] = 0
            x_cut, y_cut = None, None
    else:
        params['cutoff_length'] = params_y_flp['cutoff_length'] = 0
        x_cut, y_cut = None, None
    floorplan = FloorPlanWI(
        PolygonWI(
            # geometry=Polygon([[x_cut, y_cut], [x_cut, 0], [x_max, 0], [x_max, y_max], [0, y_max], [0, y_cut]]),
            # let's try cutting upper right instead of lower left corner, this makes it easier to determine where the inside is...
            geometry=Polygon([[0, 0], [x_max, 0], [x_max, y_max - y_cut], [x_max - x_cut, y_max - y_cut], [x_max - x_cut, y_max], [0, y_max]] if (x_cut is not None and y_cut is not None) else 
                             [[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]]),
            material=material_flp, 
            z_min=params['z_min'],
            z_max=params['z_max'],
            name='walls_floor_ceiling'),
        other=[],
        name='floorplan', 
        material_walls=material_walls)
    
    params = insert_material_into_cfg(params, params_mat, [])#, [params['material_walls']])
    params['side_length_y'] = params_y_flp['side_length']
    params['cutoff_length_y'] = params_y_flp['cutoff_length']
    
    # now add objects inside/close to walls
    for k, cfg_obj in cfg_flp.get('wall_objects', {}).items():
        cfg_obj = place_wall_objects(floorplan=floorplan, name=k, cfg_obj=cfg_obj, cfg_mat=cfg_mat, verbose=verbose)
        if k in params['wall_objects'].keys():
            del params['wall_objects'][k]
        params['wall_objects'].update(cfg_obj)
            

    return floorplan, params

def normalize_geometry_position(
        geom : T
    ) -> T:
    """
    Normalizes the position of the given object by translating its geometry 
    so that the minimum x and y coordinates of its geometry are aligned to the origin (0, 0).
    Args:
        object (ObjectWI): The object whose position is to be normalized.
    Returns:
        ObjectWI: A copy of the object with its geometry translated to the normalized position.
    """
    x_min, y_min = min(c[0] for c in geom.geometry.exterior.coords), min(c[1] for c in geom.geometry.exterior.coords) # type: ignore
    return move_geometry_by(geom=geom, direction=(-x_min, -y_min))

def move_geometry_by(
        geom: T, 
        direction: ArrayLike
    ) -> T:
    """
    Moves an object by a specified direction vector.
    This function takes a GeometryWI and a direction vector,
    and returns a new `GeometryWI` instance with all its geometries moved by
    the given direction. The movement is applied to each geometry in the
    object's geometry list by transforming their coordinates.
    Args:
        geom (GeometryWI): The object to be moved.
        direction (ArrayLike): A vector specifying the direction and magnitude
            of the movement. It is added to the coordinates of each geometry.
    Returns:
        GeometryWI: A copy of the object with its geometries moved by the specified direction.
    """
    if isinstance(geom, ObjectWI):
        geometries_moved = [
            PolygonWI(
                transform(g.geometry, lambda x: x + np.array(direction)), 
                material=g.material, 
                z_min=g.z_min, 
                z_max=g.z_max, 
                name=g.name
                ) 
            for g in geom.geometry_list
        ]
        geom_new = cast(T, ObjectWI(geometry_list=geometries_moved, name=geom.name))
    elif isinstance(geom, LineStringWI):
        moved_geometry = transform(geom.geometry, lambda x: x + np.array(direction))
        return cast(T, LineStringWI(
            geometry=moved_geometry,
            material=geom.material,
            z_min=geom.z_min,
            z_max=geom.z_max,
            name=geom.name
        ))
    elif isinstance(geom, PolygonWI):
        moved_geometry = transform(geom.geometry, lambda x: x + np.array(direction))
        geom_new = cast(T, PolygonWI(
            geometry=moved_geometry,
            material=geom.material,
            z_min=geom.z_min,
            z_max=geom.z_max,
            name=geom.name
        ))
    elif isinstance(geom, TxWI):
        # For TxWI, the position is stored in geometry (Point) and z_min/z_max
        direction_array = np.array(direction)
        moved_geometry = transform(geom.geometry, lambda x: x + direction_array[:2])  # Only x, y
        current_z = geom.z_min  # z_min and z_max are the same for TxWI
        moved_position = [moved_geometry.x, moved_geometry.y, current_z]
        geom_new = cast(T, TxWI(position=moved_position, name=geom.name))
    else:
        raise TypeError(f'{type(geom)=}')
    return geom_new

def move_geometry_to(
        geom: T, 
        target_location: ArrayLike
    ) -> T:
    """
    Moves the given object to a specified target location.
    This function calculates the minimum x and y coordinates of the object's geometry
    and determines the translation vector required to move the object so that its 
    minimum coordinates align with the target location. The object is then moved 
    using the `move_object_by` function.
    Args:
        object (ObjectWI): The object to be moved. 
        target_location (ArrayLike): The target location to which the object should 
            be moved. It should be an array-like structure containing x and y coordinates.
    Returns:
        ObjectWI: Copy of the object moved to the target location.
    """
    coords = geom.get_hull_coords()
    if not all(len(c) >= 2 for c in coords):
        raise RuntimeError(f'{coords=}')
    x_min, y_min = min(c[0] for c in coords), min(c[1] for c in coords) 
    return move_geometry_by(geom=geom, direction=np.asarray(target_location) - np.array([x_min, y_min]))

def get_values(
        cfg: dict
    ) -> dict[Any, Any]:
    """
    Generate a dictionary of values from a given input configuration. The configuration may
    define possible ranges or lists of values, from which we draw randomly, or fixed values.

    This function processes a potentially nested dictionary and generates a new dictionary
    where the values are determined based on the following rules:
    
    1. If a value in `cfg` is a list of two numbers, it is interpreted as a range:
        - If both numbers are integers, a random integer within the range is selected.
        - Otherwise, a random float within the range is selected.
    2. If a value in `cfg` is a list of more than two elements or objects that are not numbers, 
        one element is randomly selected from the list.
    3. If a value in `cfg` is a dictionary, the function is called recursively on that
        dictionary.
    4. For all other types of values, they are assumed fixed and directly copied to the output 
        dictionary.

    Args:
        cfg (dict): A nested dictionary containing configuration values.

    Returns:
        dict[Any, Any]: A dictionary with the same structure as `cfg`, but with values
                        generated or selected based on the rules described above.

    Note:
        - The function assumes that the `rng` object (a random number generator) and
            the `Number` type are defined in the global scope.
        - Edge cases, such as lists containing exactly two specific numbers, are not
            explicitly handled.
    """
    val_dict = {}
    for k, v in cfg.items():
        if isinstance(v, list):
            if len(v)==2 and all(isinstance(w, Number) for w in v):
                # we interpret this as lower and upper bound for allowed values
                val_min, val_max = v
                if all(isinstance(w, int) for w in [val_min, val_max]):
                    val_dict[k] = int(rng.integers(val_min, val_max))
                else:
                    val_dict[k] = round_to_significant_digits(rng.random() * (val_max - val_min) + val_min, 3)
            else:
                # we interpret this as a list to choose one element from randomly
                # note we dont cover the edge case of drawing one out of two specfic numbers
                val_dict[k] = v[rng.integers(0, len(v))]
        elif isinstance(v, dict):
            val_dict[k] = get_values(v)
        else:
            val_dict[k] = v
    return val_dict

def insert_material_into_cfg(
        cfg : dict[str,Any], 
        cfg_mat : dict[str,Any], 
        materials_used : list[str]
    ) -> dict[str, Any]:
    """
    Inserts material information into the configuration dictionary for the used materials.

    Args:
        cfg (dict): The configuration dictionary to update.
        cfg_mat (dict): The dictionary of available materials and their properties.
        materials_used (list[str]): List of material names used. Only list materials here, which are not already given in the config. This happens for objects loaded from files.

    Returns:
        dict: The updated configuration dictionary with material details included.
    """
    if len(materials_used) >= 1:
        for idm, mat_name in enumerate(materials_used):
            cfg[f'material{idm}'] = mat_name

    for k in list(cfg.keys()):
        v = cfg[k]
        if k.startswith('material') and not k=='material_type':
            if isinstance(v, dict):
                # print(f'{k=}\n{v=}\n{cfg=}')
                continue
            if isinstance(v, str) and strip_suffix(v) in cfg_mat.keys():
                try:
                    # cfg[k] = {v : cfg_mat[v]}
                    cfg[v] = cfg_mat[strip_suffix(v)]
                except Exception as e:
                    raise ValueError(f'{k=}\n{v=}\n{e}')
            else:
                raise KeyError(f'material {k} not found in {cfg_mat=}')
        elif isinstance(v, dict):
            cfg[k] = insert_material_into_cfg(v, cfg_mat=cfg_mat, materials_used=[])
    return cfg

def ring_to_segments(ring : LinearRing) -> list[LineString]:
    """
    Splits a LinearRing geometry into a list of LineString segments.

    Each segment represents one edge of the ring.

    Args:
        ring (LinearRing): The input LinearRing geometry.

    Returns:
        list[LineString]: A list of LineString objects, each representing a segment of the ring.
    """
    coords = list(ring.coords)
    return [LineString([coords[i], coords[i+1]]) for i in range(len(coords) - 1)]
