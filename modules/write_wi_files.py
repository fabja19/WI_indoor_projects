'''
Functions to write our python representation of projects defined in project.py to files which can be read by WI and add them all to an xml file.
Our implementation of floorplans and objects is probably not the most efficient one.
We could write a little bit less class specific logic by following WI's logic more closely, but this works.
'''
from pathlib import Path
from typing import Any, cast
import numpy as np
from shapely import Polygon, Point, LineString, ops
from warnings import warn
from matplotlib.colors import to_rgba
from xml.etree import ElementTree  as ET
import copy
import yaml
from matplotlib import pyplot as plt

from .project import ObjectWI, ProjectWI, FloorPlanWI, MaterialWI, PolygonWI, LineStringWI, GeometryWI, check_contains, TxWI, RxWI, plot_project
from .utils import round_to_significant_digits

def str_list_to_file(
        lines : list[str],
        file_name : str | Path,
    ) -> None:
    """
    Writes a list of strings to a file, ensuring each string is written on a new line.
    If the specified file already exists, a FileExistsError is raised.

    Args:
        lines (list[str]): A list of strings to be written to the file.
        file_name (str | Path): The name or path of the file to write to.

    Raises:
        FileExistsError: If the file specified by `file_name` already exists.

    Returns:
        None
    """
    file_name = Path(file_name)
    if file_name.exists():
        raise FileExistsError(f'{file_name=}')
    with open(file_name, 'w', newline='\r\n') as f:
        f.writelines([l + '\n' for l in lines])
    
def get_header() -> list[str]:
    """
    Generates a predefined header structure for a file.

    Returns:
        list[str]: A list of strings representing the header.
    """
    return [
        'begin_<reference>',
        'cartesian',
        'longitude 0.0',
        'latitude 0.0',
        'visible no',
        'sealevel',
        'end_<reference>'
    ]   

def extract_material_index_geometry(
        geometry : GeometryWI, 
        material_list : list[MaterialWI]
    ) -> int:
    """
    Extracts the index of the material associated with a given geometry 
    from a list of materials.

    Args:
        geometry (GeometryWI): The geometry object containing a material 
            reference to be resolved.
        material_list (list[MaterialWI]): A list of MaterialWI objects 
            to search for the material index.

    Returns:
        int: The index of the material in the material_list.

    Raises:
        ValueError: If the geometry does not have an associated material.
    """
    if geometry.material is None:
        raise ValueError(f'{geometry.material=} for {geometry=}, {geometry.name=}')
    return extract_material_index(geometry.material, material_list=material_list)

def extract_material_index(
        material : MaterialWI, 
        material_list : list[MaterialWI]
    ) -> int:
    """
    Extracts the index of a specific material from a list of materials.

    Args:
        material (MaterialWI): The material to find in the list. Must not be None.
        material_list (list[MaterialWI]): The list of materials to search through.

    Returns:
        int: The index of the matching material in the material list.

    Raises:
        ValueError: If the provided material is None.
        ValueError: If no matching material is found or if multiple matches are found.
    """
    if material is None:
        raise ValueError("Material cannot be None")
    matching_material_indices = [i for i in range(len(material_list)) if material_list[i]==material]
    if len(matching_material_indices) != 1:
        raise ValueError(f'found matching materials {matching_material_indices} for {material.name=},\n{[m.name for m in material_list]=}')
    material_index = matching_material_indices[0]
    return material_index

def polygon_to_sub_structure_str_list(
        polygon : PolygonWI,
        material_list : list[MaterialWI],
        material_walls : MaterialWI | None = None,
        top_invisible : bool = False
    ) -> list[str]:
    """
    Converts a PolygonWI object into a list of strings representing its sub-structure.

    Args:
        polygon (PolygonWI): The polygon to be serialized. May be a floorplan.
        material_list (list[MaterialWI]): A list of materials used in the project.
        material_walls (MaterialWI | None, optional): The material to use for walls. If None, the polygon's material is used. Defaults to None.

    Returns:
        list[str]: A list of strings representing the polygon's sub-structure in WI format.

    Notes:
        - The function generates top, bottom, and side faces for the polygon.
        - It makes sure that the objects_inside are cut out properly from each wall.
        - Extra faces of objects_inside are appended to the structure after the main faces.
    """
    if polygon is None:
        return []
    material_index = extract_material_index_geometry(geometry=polygon, material_list=material_list)
    material_index_walls = material_index if material_walls is None else extract_material_index(material=material_walls, material_list=material_list)

    coords = polygon.geometry.normalize().boundary.coords
    coords = [coords[i] for i in range(len(coords)) if not coords[i]==coords[(i+1)%len(coords)]][::-1]
    
    # top
    lines = [
        f'begin_<sub_structure> {polygon.name}',
        f'begin_<face> Top', 
    ] 
    if top_invisible:
        lines += [f'invisible']
    lines.extend([
        f'Material {material_index}', 
        f'nVertices {len(coords)}'
    ])
    lines.extend([
        f'{potential_numbers_to_str([c[0], c[1], polygon.z_max])}' 
        for c in coords#[::-1]
    ])
    lines.append('end_<face>')

    # sides
    walls_with_objects = polygon.split()
    for i, wo in enumerate(walls_with_objects):
        wall, objects = wo if isinstance(wo, (list,tuple)) else (wo, [])
        lines.extend(wall_with_holes_to_str_list(
            wall=wall,
            face_index=i,
            material_index=material_index_walls,
            holes=objects,
            material_list=material_list
        ))

    # bottom
    lines.extend([
        f'begin_<face> Bottom', 
        'double_sided',
        f'Material {material_index}', 
        f'nVertices {len(coords)}'
    ] )
    lines.extend([
        f'{potential_numbers_to_str([c[0], c[1], polygon.z_min])}' 
        for c in coords[::-1]
    ])
    lines.append('end_<face>')
    lines.append('end_<sub_structure>')
    return lines

def wall_with_holes_to_str_list(
        wall : LineStringWI,
        face_index : int,
        material_index : int,
        holes : list[LineStringWI],
        material_list : list[MaterialWI]
    ) -> list[str]:
    """
    Converts a wall with holes (e.g., windows, doors) into a list of strings representing its structure.

    Args:
        wall (LineStringWI): The wall geometry to be serialized.
        face_index (int): The index of the wall face.
        material_index (int): The material index for the wall.
        holes (list[LineStringWI]): A list of LineStringWI objects representing holes in the wall.
        material_list (list[MaterialWI]): A list of materials used in the project.

    Returns:
        list[str]: A list of strings representing the wall and its holes in WI format.

    Raises:
        ValueError: If any hole is not contained within the wall.
    """
    lines = []
    if not all(check_contains(wall, h) for h in holes):
        raise ValueError(f'Some holes are not in this wall.')
    # wall and holes are lying in a vertical plane, represent them there as 2D polygons wrt a 2D CS with origin in the first wall coordinate
    origin = wall.geometry.boundary.geoms[0]
    direction = np.squeeze(np.array(wall.geometry.boundary.geoms[1].coords) - np.array(origin.coords))
    direction = direction / np.linalg.norm(direction)
    holes_polys = [create_polygon_local(h, origin) for h in holes] # type: ignore
    wall_poly = Polygon(create_polygon_local(wall, origin).exterior.coords, holes=[h.exterior.coords[::-1] for h in holes_polys]) # type: ignore
    # split it up 
    coords_for_verts = sorted(list(set(x for hole in holes_polys for x, _ in hole.exterior.coords)))
    verticals = [LineString([(c, wall.z_min), (c, wall.z_max)]) for c in coords_for_verts]
    splits = split_polygon_recursively(poly=wall_poly, verticals=verticals)
    splits = [s.simplify(0) for s in splits]
    # calculate back to normal coordinates
    # currently, the x axis of the 2D polygons corresponds to the wall vector, y axis to z
    for i, split_poly in enumerate(splits):
        if not isinstance(split_poly, Polygon):
            raise RuntimeError(f'after splitting up wall, one segment is not Polygon but {type(split_poly)}')
        lines.extend(
            polygon2d_to_face(
                poly=split_poly, 
                origin=origin, 
                direction=direction, 
                material_index=material_index, 
                name=f'Wall{face_index}_{i}'
            )
        )
        

    # write all hole polys into faces, taking into account their materials
    for i, (hole, hole_poly) in enumerate(zip(holes, holes_polys)):
        hole_material_index = extract_material_index_geometry(geometry=hole, material_list=material_list)
        lines.extend(polygon2d_to_face(
            poly=hole_poly,
            origin=origin,
            direction=direction,
            material_index=hole_material_index,
            name=f'{hole.name}'
        ))

    return lines

def split_polygon_recursively(
        poly : Polygon,
        verticals : list[LineString]
    ) -> list[Polygon]:
    """
    Recursively splits a Polygon using a list of vertical LineStrings.

    Given a polygon and a list of vertical lines, this function splits the polygon
    by the first line, then recursively splits each resulting sub-polygon by the
    remaining lines. The process continues until all lines have been used. This is necessary
    to "cut out" e.g. windows from the walls, since in WI polygons cannot contain holes.

    Args:
        poly (Polygon): The input polygon to be split.
        verticals (list[LineString]): A list of vertical LineStrings to split the polygon.

    Returns:
        list[Polygon]: A list of resulting polygons after all splits.

    Raises:
        RuntimeError: If a geometry resulting from a split is not a Polygon.
    """
    if len(verticals) == 0:
        return [poly]
    vert, verticals_remaining = verticals[0], verticals[1:]
    polys_split = list(ops.split(poly, vert).geoms)
    polys_split_further = []
    for p in polys_split:
        if not isinstance(p, Polygon):
            raise RuntimeError(f'{type(p)=}')
        polys_split_further.extend(split_polygon_recursively(p, verticals_remaining))
    return polys_split_further

def polygon2d_to_face(
        poly : Polygon, 
        origin : Point,
        direction : np.ndarray,
        material_index : int,
        name : str
    ) -> list[str]:
    """
    Converts a 2D polygon into a list of strings representing a 3D face definition.

    Args:
        poly (Polygon): The 2D polygon to convert, with coordinates in local space.
        origin (Point): The origin point to offset the polygon's coordinates.
        direction (np.ndarray): A 2-element array representing the direction vectors for x and y axes.
        material_index (int): The index of the material to assign to the face.
        name (str): The name to assign to the face.

    Returns:
        list[str]: A list of strings representing the face definition in the target format.
    """
    lines = []
    coords = poly.exterior.coords
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    lines.extend([
        f'begin_<face> {name}',
        'double_sided',
        f'Material {material_index}',
        f'nVertices {len(coords)}'
    ])
    for coord in coords:
        x, y = coord
        global_x = origin.x + x * direction[0]
        global_y = origin.y + x * direction[1]
        global_z = y
        lines.append(f'{potential_numbers_to_str([global_x, global_y, global_z])}')
    lines.append('end_<face>')
    return lines

def create_polygon_local(
        line : LineStringWI, 
        origin : Point
    ) -> Polygon:
    """
    Creates a rectangular Polygon in a local coordinate system based on a given line and origin point.

    Args:
        line (LineStringWI): The line object containing geometry and z-range information.
        origin (Point): The origin point from which distances to the line are measured.

    Returns:
        Polygon: A rectangular polygon defined by the distances from the origin to the line and the z-range of the line.

    Raises:
        AssertionError: If the line geometry does not consist of exactly two coordinates.
    """
    assert len(line.geometry.coords) == 2, f'{len(line.geometry.coords)=} != 2'
    x1, x2 = origin.distance(line.geometry), origin.hausdorff_distance(line.geometry)
    y1, y2 = line.z_min, line.z_max
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

def floorplan_to_str_list(
        floorplan : FloorPlanWI
    ) -> list[str]:
    """
    Converts a FloorPlanWI object into a list of strings to be written into WI file.

    Args:
        floorplan (FloorPlanWI): The floorplan.

    Returns:
        list[str]: A list of strings representing the floorplan in WI format.

    Notes:
        - Materials are serialized first, followed by the structure group.
        - The structure group includes the main polygon and additional line strings (e.g., windows, doors).
    """
    material_list = floorplan.get_all_materials()
    lines = [f'begin_<floorplan> {floorplan.name.replace('floorplan:', '')}']
    lines.extend(get_header())
    for k, m in enumerate(material_list):
        lines.extend(material_to_str_list(m, k))
    lines.extend([
        'begin_<structure_group>',
        'begin_<structure>'
    ])

    lines.extend(polygon_to_sub_structure_str_list(
        polygon=floorplan, 
        material_list=material_list,
        material_walls=floorplan.material_walls,
        top_invisible=True
    ))

    lines.extend([
        'end_<structure>',
        'end_<structure_group>',
        'end_<floorplan>'
    ])
    return lines

def ls_mls_to_str_list(
        mls : LineStringWI, 
        material_list : list[MaterialWI]
    ) -> list[str]:
    """
    Converts a LineStringWI or MultiLineStringWI object into a list of strings representing its structure.

    Args:
        mls (LineStringWI | MultiLineStringWI): The line string or multi-line string to be serialized.
        material_list (list[MaterialWI]): A list of materials used in the project.

    Returns:
        list[str]: A list of strings representing the line string or multi-line string in WI format.

    Notes:
        - MultiLineStringWI objects are processed recursively, with each LineStringWI serialized individually.
        - Each face is represented as a vertical rectangle with four vertices.
    """
    lines = []
    material_index = extract_material_index_geometry(geometry=mls, material_list=material_list)
    name = mls.name if not mls.name is None else "LS"
    coords = mls.geometry.coords
    # iterate
    for i in range(0, len(coords)):
        k = (i - 1) % len(coords)
        l = (i ) % len(coords)
        lines.extend([
            f'begin_<face> {name}{i}', 
            'double_sided',
            f'Material {material_index}', 
            f'nVertices 4'
        ])
        ### add walls
        lines.append(f'{coords[l][0]} {coords[l][1]} {mls.z_max}')
        lines.append(f'{coords[k][0]} {coords[k][1]} {mls.z_max}')
        lines.append(f'{coords[k][0]} {coords[k][1]} {mls.z_min}')
        lines.append(f'{coords[l][0]} {coords[l][1]} {mls.z_min}')
        lines.append('end_<face>')
    return lines

def object_to_str_list(
        object: ObjectWI
    ) -> list[str]:
    """
    Converts an ObjectWI instance into a list of strings representing its structure 
    and materials in a specific format.

    Args:
    object (ObjectWI): The object to be converted. It contains materials and 
                geometry information.

    Returns:
    list[str]: A list of strings representing the object, including its 
            materials and geometry, formatted for further processing.

    The function performs the following steps:
    1. Retrieves all materials associated with the object.
    2. Adds a header and object-specific information to the output.
    3. Iterates through the materials and converts each to a string representation.
    4. Processes the geometry of the object, converting each polygon into a 
        structured string representation.
    5. Appends start and end markers for the object and its structures.

    Note:
    - The function assumes the existence of helper functions like 
        `get_header`, `material_to_str_list`, and `polygon_to_sub_structure_str_list`.
    - The format of the output strings is specific to the application's requirements.
    """
    material_list = object.get_all_materials()
    lines = [f'begin_<object> {object.name}']
    lines.extend(get_header())
    for k, m in enumerate(material_list):
        lines.extend(material_to_str_list(m, k))
    for polygon in object.geometry_list:
        lines.extend([
            'begin_<structure_group>',
            'begin_<structure>'
        ])
        lines.extend(polygon_to_sub_structure_str_list(
            polygon=polygon, 
            material_list=material_list
        ))
        lines.extend([
            'end_<structure>',
            'end_<structure_group>',
        ])
    lines.append('end_<object>')
    return lines

def material_to_str_list(
        material: MaterialWI, 
        material_id: int
    ) -> list[str]:
    """
    Converts a MaterialWI object into a list of strings representing the lines we have to write into the WI files.

    Args:
        material (MaterialWI): The material object to be converted. If None, an empty list is returned.
        material_id (int): The unique identifier for the material.

    Returns:
        list[str]: A list of strings representing the material's properties in WI format.

    Notes:
        - The format includes the material's type, thickness, and other properties.
        - Properties are serialized as key-value pairs or nested structures.
    """
    if material is None:
        return []
    lines = [f'begin_<Material> {material.name}'] 
    lines.append(f'Material {material_id}')
    lines.append(f'{material.material_type}')
    lines.extend(get_material_color(material.name))
    lines.extend(dict_to_lines({'thickness': potential_numbers_to_str(material.thickness)}))
    mat_properties = material.get_properties()
    lines.extend(dict_to_lines({k : v for k, v in mat_properties.items() if not 'DielectricLayer' in k}))
    for k, v in mat_properties.items():
        if not 'DielectricLayer' in k:
            continue
        elif isinstance(v, list) and len(v) >= 0:
            for dl in v:
                if dl is None:
                    continue
                lines.append('begin_<DielectricLayer>')
                lines.extend(dict_to_lines(dl))
                lines.append('end_<DielectricLayer>')
        else:
            raise ValueError(f'{k=}\t{v=}')
    lines.append('end_<Material>')
    return lines

def dict_to_lines(
        dic: dict
    ) -> list[str]:
    """
    Converts a nested dictionary into a list of strings, where each string represents
    a key-value pair or a nested structure in a specific format.

    Args:
        dic (dict): The input dictionary to be converted. The dictionary can contain
            values of types `int`, `float`, `str`, `list`, or `dict`. Nested dictionaries
            are supported.

    Returns:
        list[str]: A list of strings representing the dictionary's structure and content.

    Raises:
        NotImplementedError: If a value in the dictionary is of an unsupported type.

    Notes:
        - Keys and values of type `int`, `float`, or `str` are written as "key: value".
        - Values of type `list` are written as "key: item1 item2 ...".
        - Nested dictionaries are enclosed within "begin_<key>" and "end_<key>" markers.
        - Values that are `None` are skipped and not included in the output.
    """
    lines = []
    for k, v in dic.items():
        if v is None:
            continue
        elif isinstance(v, (int, float, str, list)):
            #########
            # maybe numbers close to 0 have to be written in scientific notation?
            lines.append(f'{k} {potential_numbers_to_str(v)}')
        elif isinstance(v, dict):
            lines.append(f'begin_<{k}>')
            lines.extend(dict_to_lines(v))
            lines.append(f'end_<{k}>')
        else:
            raise NotImplementedError(f'in dict_to_lines found value {v=}')
    return lines

def project_to_files(
        project : ProjectWI, 
        project_dir : str | Path,
        xml_glob : str,
        cfg_here : dict | None
    ) -> None:
    """
    Writes a ProjectWI object to files in a specified directory.

    Args:
        project (ProjectWI): The project to be serialized.
        project_dir (str | Path): The directory where the files will be saved.

    Raises:
        ValueError: If the project does not have a floorplan.

    Notes:
        - Will also save a plot to the project dir
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    if project.floorplan is None:
        raise ValueError('no floorplan')
    file_name_flp = get_file_name(project.floorplan, project_dir)
    str_list_to_file(
        lines=floorplan_to_str_list(floorplan=project.floorplan),
        file_name=file_name_flp
    )
    file_names_objects = []
    for object in project.objects:
        file_name_object = get_file_name(object, project_dir)
        str_list_to_file(
            lines=object_to_str_list(object=object),
            file_name=file_name_object
        )   
        file_names_objects.append(file_name_object)
    tx_list, rx_list = project.tx, project.rx
    n_txrx = len(tx_list) + len(rx_list)
    # the project file is not necessary when we run from xml files, but we should start with these to check in the GUI that everything works
    object_file_paths = project_to_setup(floorplan_file=file_name_flp, object_files=file_names_objects, first_txrx_number=n_txrx+1, project_dir=project_dir, properties=project.properties)
    tx_dicts, rx_dicts = generate_txrx_file(project_dir=project_dir, project=project, tx_list=tx_list, rx_list=rx_list)

    fig = plot_project(project=project)
    fig.savefig(project_dir / 'plot.png')
    plt.close()
    
    project_to_xmls(project_dir=project_dir, object_file_paths=object_file_paths, tx_dicts=tx_dicts, rx_dicts=rx_dicts, xml_glob=xml_glob, properties=project.properties)

    if cfg_here is not None:
        with open(project_dir / 'config_here.yml', 'w') as f:
            yaml.safe_dump(cfg_here, f)
    

def generate_txrx_file(
        project_dir : Path,
        project: ProjectWI,
        tx_list : list[TxWI], 
        rx_list : list[RxWI]
    ) -> tuple[dict[int, dict[str,str]], dict[int, dict[str,str]]]:
    """
    Generates a combined transmitter (TX) and receiver (RX) configuration file for a wireless indoor project using template files.
    Args:
        project_dir (Path): The directory where the output 'txrx_file.txrx' will be saved.
        project (ProjectWI): The project object containing floorplan and geometry information.
        tx_list (list[TxWI]): List of transmitter objects to be included in the configuration.
        rx_list (list[RxWI]): List of receiver objects to be included in the configuration.
    Raises:
        FileNotFoundError: If the required TX or RX template files are not found.
    Description:
        - Reads TX and RX template files from 'wi_templates/tx_point.tx' and 'wi_templates/rx_grid.rx'.
        - For each transmitter in tx_list, fills in the template with the transmitter's name, project ID, and location.
        - For each receiver in rx_list, fills in the template with the receiver's name, project ID, height, spacing, and grid dimensions based on the project's floorplan.
        - Concatenates all filled templates and writes the result to 'txrx_file.txrx' in the specified project directory.
    """
    template_file_tx, template_file_rx = Path('wi_templates/tx_point.tx'), Path('wi_templates/rx_grid.rx')
    if not all(t.exists() for t in [template_file_tx, template_file_rx]):
        raise FileNotFoundError(f'we need template files {template_file_tx} and {template_file_rx}')
    with open(template_file_tx, 'r') as f:
        template_tx = f.read()
    with open(template_file_rx, 'r') as f:
        template_rx = f.read()
    
    tx_dicts = {}
    rx_dicts = {}
    project_id = 1
    txrx_str = ''
    for tx in tx_list:
        coords = tx.get_hull_coords()
        params_dict = {
            'TXNAME' : tx.name,
            'PROJECTID' : str(project_id),
            'LOCATION' : f'{coords[0][0]} {coords[0][1]} {tx.z_min}'
        }
        s =  template_tx
        for k, v in params_dict.items():
            s = s.replace(k, v)
        tx_dicts[project_id] = params_dict
        txrx_str += s
        project_id += 1
    coords = project.floorplan.get_hull_coords()
    x_max, y_max = max([c[0] for c in coords]), max([c[1] for c in coords]) 
    for rx in rx_list:
        params_dict = {
                'RXNAME' : rx.name,
                'PROJECTID' : str(project_id),
                'HEIGHT' : str(rx.height),
                'SPACING' : str(rx.spacing),
                'XVERT' : str(rx.spacing / 2),
                'YVERT' : str(rx.spacing / 2),
                'LENGTHX' : str(x_max - rx.spacing),
                'LENGTHY' : str(y_max - rx.spacing)
        }
        s =  template_rx
        for k, v in params_dict.items():
            s = s.replace(k, v)
        rx_dicts[project_id] = params_dict
        txrx_str += s
        project_id += 1
    with open(project_dir / 'txrx_file.txrx', 'w', newline='\r\n') as f:
        f.write(txrx_str)
    
    return tx_dicts, rx_dicts

def project_to_setup(
        floorplan_file : str | Path, 
        object_files : list[str | Path], 
        first_txrx_number : int | str, 
        project_dir : str | Path,
        properties : dict[str,float],
    ) -> list[Path]:
    """
    Generates a 'project.setup' file for a given project directory using a template and provided floorplan and object files.

    Args:
        floorplan_file (str | Path): Path to the floorplan file to include as the main feature.
        object_files (list[str | Path]): List of paths to additional object files to include as features.
        first_txrx_number (int | str): The first available transmitter/receiver number to use in the setup file.
        project_dir (str | Path): Directory where all files will be saved.

    Returns:
        A list of the object files paths, to insert into the xml file.

    Raises:
        FileNotFoundError: If the required template setup file does not exist.

    Notes:
        - The function expects a template file at 'wi_templates/seminar_room_project.setup'.
        - All file paths in the generated setup file are made relative to the project directory.
    """
    template_file = Path('wi_templates/seminar_room_project.setup')
    object_file_paths = []
    if not template_file.exists():
        raise FileNotFoundError(f'we need a template setup file {template_file}')
    with open(template_file, 'r') as f:
        template = f.read()
    feature_str = ''
    for idf, file_path in enumerate([floorplan_file] + object_files):
        feature_str += 'begin_<feature>\n'
        feature_str += f'feature {idf}\n'
        feature_str += 'floorplan\n' if file_path==floorplan_file else 'object\n'
        feature_str += 'active\n'
        feature_str += f'filename {str(Path(file_path).relative_to(project_dir))}\n'
        feature_str += 'end_<feature>\n'
        if file_path!=floorplan_file:
            object_file_paths.append(file_path)
    template = template.replace('FEATURES_HERE\n', feature_str)
    template = template.replace('FIRSTAVAILABLEFEATURENUMBER', f'{idf + 1}')
    template = template.replace('FIRSTAVAILABLETXRXNUMBER', str(first_txrx_number))
    template = template.replace('HUMIDITY', str(properties['humidity']))
    template = template.replace('TEMPERATURE', str(properties['temperature']))
    template = template.replace('PRESSURE', str(properties['pressure']))
    with open(Path(project_dir) / 'project.setup', 'w', newline='\r\n') as f:
        f.write(template)
    return object_file_paths
    
def project_to_xmls(
        project_dir : Path, 
        object_file_paths : list[Path], 
        tx_dicts : dict[int, dict[str,str]], 
        rx_dicts : dict[int, dict[str,str]],
        xml_glob : str,
        properties : dict[str,float]
    ) -> None:
    """
    Generates XML files for each transmitter-receiver (TX-RX) pair in a wireless indoor project.

    This function reads a template XML file, modifies it by adding objects and TX/RX configurations,
    and writes out a separate XML file for every combination of TX and RX provided.

    Args:
        project_dir (Path): The directory where the generated XML files will be saved.
        object_file_paths (list[Path]): List of file paths to object files to be added to the XML.
        tx_dicts (dict[int, dict[str, str]]): Dictionary mapping TX IDs to their configuration dictionaries.
        rx_dicts (dict[int, dict[str, str]]): Dictionary mapping RX IDs to their configuration dictionaries.
        xml_glob (str): Used to find template xml files.

    Returns:
        None: This function writes files to disk and does not return a value.

    Side Effects:
        - Reads the template XML files from "wi_templates".
        - Writes one XML file per TX-RX pair and per xml template to `project_dir`, named as "{template.stem}_tx{tx_id}_rx{rx_id}.xml".
    """
    for xml_file in Path('wi_templates').glob(xml_glob):
        with open(xml_file, "r") as f:
            xml_str = f.read()

        # Replace illegal tags
        xml_str = xml_str.replace("remcom::rxapi::", "remcom_rxapi_")

        # Parse from string → root element -> ElementTree
        root = ET.fromstring(xml_str.encode())

        add_objects_to_xml(root=root, object_file_paths=object_file_paths)
        output_dir = root.findall('.//PathResultsDatabase/remcom_rxapi_PathResultsDatabase/Filename/remcom_rxapi_FileDescription/Filename/remcom_rxapi_String')
        assert len(output_dir) == 1, f'{output_dir=}'
        output_dir = output_dir[0].attrib['Value']
        (project_dir / output_dir).parent.mkdir(exist_ok=True)
        # in the xml files, keys are with capital letters
        # temperature appears in various places, also for antennas, so we have to be specific
        properties_adjusted = {f'.//remcom_rxapi_X3D/{k.title()}/remcom_rxapi_Double' : v for k, v in properties.items()}

        change_multiple_values_in_nested_etree(root, properties_adjusted)

        for tx_id, tx_d in tx_dicts.items():
            for rx_id, rx_d in rx_dicts.items():
                root_here = add_txrx_to_xml(root=root, tx_dict=tx_d, rx_dict=rx_d)

                # Serialize back to string and restore original prefixes
                modified_str = ET.tostring(root_here, encoding="unicode")
                modified_str = '<!DOCTYPE InSite>\n' + modified_str.replace("remcom_rxapi_", "remcom::rxapi::").replace(" />", "/>") + "\n"

                # Write final result. For some reason, xml files must have Unix line endings apparently, whereas the files for the GUI use windows line endings
                with open(project_dir / f"{xml_file.stem}_tx{tx_id}_rx{rx_id}.xml", "w", newline='\n') as f:
                    f.write(modified_str)

def add_txrx_to_xml(
        root : ET.Element, 
        tx_dict : dict[str,str], 
        rx_dict : dict[str,str]
    ) -> ET.Element:
    """
    Adds or updates transmitter (Tx) and receiver (Rx) information in an XML tree.
    This function takes an XML root element and two dictionaries containing Tx and Rx parameters,
    then updates the corresponding Tx and Rx blocks in the XML with the provided values.
    Args:
        root (ET.Element): The root element of the XML tree to be modified.
        tx_dict (dict[str, str]): Dictionary containing transmitter parameters. Must include keys:
            - 'TXNAME': Name of the transmitter.
            - 'PROJECTID': Project ID for the transmitter.
            - 'LOCATION': Space-separated string with X, Y, Z coordinates.
        rx_dict (dict[str, str]): Dictionary containing receiver parameters. Must include keys:
            - 'RXNAME': Name of the receiver.
            - 'PROJECTID': Project ID for the receiver.
            - 'HEIGHT': Height coordinate (Z).
            - 'XVERT': X coordinate.
            - 'YVERT': Y coordinate.
            - 'SPACING': Spacing value.
            - 'LENGTHX': Length in X direction.
            - 'LENGTHY': Length in Y direction.
    Returns:
        ET.Element: A deep copy of the original XML root element with updated Tx and Rx information.
    Raises:
        ValueError: If the XML structure does not contain exactly two Tx/Rx blocks,
                    or if the expected Tx or Rx elements are missing.
    """
    root_new = copy.deepcopy(root)
    txrx_blocks = root_new.findall(".//TxRxSet")   
    if not len(txrx_blocks) == 2:
        raise ValueError(f'expected 2 Tx/Rx blocks but found {len(txrx_blocks)=}')
    if len(txrx_blocks[0].findall('.//Transmitter')) == 1:
        if not len(txrx_blocks[1].findall('.//Receiver')) == 1:
            raise ValueError(f'expected 1 Tx, 1 Rx in xml file')
        tx_block, rx_block = txrx_blocks
    elif len(txrx_blocks[1].findall('.//Transmitter')) == 1:
        if not len(txrx_blocks[0].findall('.//Receiver')) == 1:
           raise ValueError(f'expected 1 Tx, 1 Rx in xml file')
        rx_block, tx_block = txrx_blocks
    else:
        raise ValueError(f'expected 1 Tx, 1 Rx in xml file')
    
    # find Rx ID in root, replace the necessary parameter
    change_multiple_values_in_nested_etree(rx_block, {
        './/ShortDescription/remcom_rxapi_String' : rx_dict['RXNAME'],
        './/OutputID/remcom_rxapi_Integer' : rx_dict['PROJECTID'],
        './/remcom_rxapi_CartesianPoint/Z/remcom_rxapi_Double' : rx_dict['HEIGHT'],
        './/remcom_rxapi_CartesianPoint/X/remcom_rxapi_Double' : rx_dict['XVERT'],
        './/remcom_rxapi_CartesianPoint/Y/remcom_rxapi_Double' : rx_dict['YVERT'],
        './/Spacing/remcom_rxapi_Double' : rx_dict['SPACING'],
        './/LengthX/remcom_rxapi_Double' : rx_dict['LENGTHX'],
        './/LengthY/remcom_rxapi_Double' : rx_dict['LENGTHY'],

    })
    # same for Tx
    x, y, z = tx_dict['LOCATION'].split()
    change_multiple_values_in_nested_etree(tx_block,{
        './/ShortDescription/remcom_rxapi_String' : tx_dict['TXNAME'],
        './/OutputID/remcom_rxapi_Integer' : tx_dict['PROJECTID'],
        './/remcom_rxapi_CartesianPoint/X/remcom_rxapi_Double' : x,
        './/remcom_rxapi_CartesianPoint/Y/remcom_rxapi_Double' : y,
        './/remcom_rxapi_CartesianPoint/Z/remcom_rxapi_Double' : z,
    })

    return root_new

def add_objects_to_xml(
        root : ET.Element, 
        object_file_paths : list[Path]
    ) -> None:
    """
    Adds object geometry nodes to an XML tree at the location of a target geometry block.

    This function locates a specific geometry block within the provided XML root, removes it,
    and inserts new geometry nodes for each object file path provided. Each new node is a deep
    copy of the original target geometry block, with its file path updated accordingly.

    Args:
        root (ET.Element): The root element of the XML tree to modify.
        object_file_paths (list[Path]): A list of file paths representing the objects to add.

    Raises:
        RuntimeError: If the target geometry block or required sub-node cannot be found.
        ValueError: If the target geometry block's parent cannot be determined or is ambiguous.
    """
    # find the object geometry block and its index, remove it
    geometry_blocks = root.findall(".//Geometry")   
    target_geom = None
    for geom in geometry_blocks:
        if geom.find("{*}remcom_rxapi_Object") is not None:
            target_geom = geom
    if target_geom is None:
        raise RuntimeError()
    parents = root.findall(".//{*}remcom_rxapi_GeometryList")
    found = False
    for p in parents:
        if target_geom in list(p):
            # print(f'found target in {p}')
            if found:
                raise ValueError()
            parent = p
            found = True
    if not found:
        raise ValueError()
    insert_index = list(parent).index(target_geom)
    parent.remove(target_geom)

    for i, ofp in enumerate(object_file_paths):
        new_node = copy.deepcopy(target_geom)
        change_value_in_nested_etree(element=new_node, tag='.//Filename/remcom_rxapi_String', value="./" + ofp.name)
        parent.insert(insert_index + i, new_node)
    
def change_value_in_nested_etree(
        element : ET.Element, 
        tag : str, 
        value : Any
    ) -> None:
    """
    Updates the 'Value' attribute of a specific nested XML element found by tag.

    Args:
        element (ET.Element): The root XML element to search within.
        tag (str): The tag name to search for among the children of the element.
        value (Any): The new value to set for the 'Value' attribute.

    Raises:
        RuntimeError: If the number of found elements with the given tag is not exactly one.
        KeyError: If the found element does not have a 'Value' attribute.
    """
    potential_nodes = element.findall(tag)
    if not len(potential_nodes) == 1:
        raise RuntimeError(f'{len(potential_nodes)=} when searching in {element=} for {tag=}')
    node = potential_nodes[0]
    if 'Value' not in node.attrib:
        raise KeyError(f"No 'Value' attribute found in node {ET.tostring(node)}")
    node.attrib['Value'] = str(value)

def change_multiple_values_in_nested_etree(
        element : ET.Element, 
        tags_values : dict[str, Any]
    ) -> None:
    """
    Updates multiple values in a nested ElementTree element.

    Iterates over the provided dictionary of tag-value pairs and updates each corresponding tag
    within the given XML element using the `change_value_in_nested_etree` function.

    Args:
        element (ET.Element): The root XML element to search and update.
        tags_values (dict[str, Any]): A dictionary where keys are tag names and values are the new values to set.

    Returns:
        None
    """
    for k, v in tags_values.items():
        change_value_in_nested_etree(element=element, tag=k, value=v)


def generate_material_block(color: str | tuple[float, float, float] | tuple[float, float, float, float]) -> list[str]:
    """
    Converts a matplotlib-style color into a flat-shaded material block.

    Parameters:
        color: A matplotlib color — can be a color name, hex string, or RGB(A) tuple.

    Returns:
        A list of strings representing the material definition block.
    """
    rgba = to_rgba(color)  # Converts to (r, g, b, a) tuple with floats in [0, 1]
    r, g, b, a = rgba

    return [
        'begin_<Color> ',
        f'ambient {r:.6f} {g:.6f} {b:.6f} {a:.6f}',
        f'diffuse {r:.6f} {g:.6f} {b:.6f} {a:.6f}',
        'specular 0.000000 0.000000 0.000000 1.000000',
        'emission 0.000000 0.000000 0.000000 0.000000',
        'shininess 0.000000',
        'end_<Color>'
    ]

colors = {
    'wood': 'brown',
    'metal' : 'black',
    'glass' : 'aliceblue',
    'concrete' : 'gray',
    'steel': 'silver',        
    'aluminium': 'purple',    
    'copper': 'orange'
}

def get_material_color(material_name : str | None) -> list[str]:
    """
    Returns a list of strings representing the color block for a given material name.

    If the material name is None or does not match any key in the `colors` dictionary,
    the color 'gray' is used by default. Otherwise, the function searches for a key in
    the `colors` dictionary that is a substring of the material name and returns the
    corresponding color block.

    Args:
        material_name (str | None): The name of the material to look up.

    Returns:
        list[str]: A list of strings representing the color block for the material.
    """
    if material_name is None:
        return generate_material_block('gray')
    for k, v in colors.items():
        if k in material_name:
            return generate_material_block(v)
    return generate_material_block('gray')

def get_file_name(
        geometry: GeometryWI, 
        save_dir: str | Path
    ) -> Path:
    """
    Generate a unique file path for a given geometry object in the specified directory.

    The function determines the file name based on the type of the geometry object.
    If the file already exists, it appends a numeric suffix to the file name to ensure
    uniqueness. If more than 100 attempts are made to generate a unique file name,
    a ValueError is raised.

    Args:
        geometry (GeometryWI): The geometry object for which the file name is generated.
                                Must be an instance of `FloorPlanWI` or `ObjectWI`.
        save_dir (str | Path): The directory where the file will be saved.

    Returns:
        Path: A unique file path for the geometry object.

    Raises:
        ValueError: If the geometry type is unsupported or if a unique file name
                    cannot be generated after 100 attempts.
    """
    stem = geometry.name.replace('object:', '').replace('floorplan:', '')
    if isinstance(geometry, FloorPlanWI):
        stem = "floorplan" if stem is None else stem
        suffix = '.flp'
    elif isinstance(geometry, ObjectWI):
        stem = "object" if stem is None else stem
        suffix = '.object'
    else:
        raise ValueError(f'{type(geometry)=}')
    save_dir = Path(save_dir)
    n_tries = 0
    file_path = save_dir / f'{stem}{suffix}'
    while file_path.exists():
        n_tries += 1
        file_path = save_dir / f'{stem}{n_tries}{suffix}'
        if n_tries > 100:
            raise ValueError(f'{file_path=} exists already, cannot save file')
    return file_path

def potential_numbers_to_str(x: str | int | float | list | None) -> str:
    """
    Converts various types of input into a string representation.

    Args:
        x (str | int | float | list | None): The input value to be converted. 
            - If `x` is a string, it is returned as is.
            - If `x` is an integer, it is converted to a string.
            - If `x` is a float, it is rounded to 5 significant digits and then converted to a string.
            - If `x` is a list, each element is recursively converted to a string and joined with spaces.
            - If `x` is None, the string "None" is returned.

    Returns:
        str: The string representation of the input.

    Raises:
        ValueError: If the input type is not one of the supported types (str, int, float, list, or None).
    """
    if isinstance(x, str):
        return x
    elif isinstance(x, int):
        return str(x)
    elif isinstance(x, float):
        return str(round_to_significant_digits(x, 5))
    elif isinstance(x, list):
        s = ''
        for e in x:
            if isinstance(e, dict):
                dict_str_list = dict_to_lines(e)
                for idl, l in enumerate(dict_str_list):
                    if l == '' or l is None:
                        continue
                    s += f'{l}\n' if idl < len(dict_str_list) - 1 else f'{l}' 
            else:
                s += f'{potential_numbers_to_str(e)} '
        return s
    elif x is None:
        return "None"
    else:
        raise ValueError(f'{x=}')