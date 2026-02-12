'''
Functions to read in files as they are created by WI and use them to create a ProjectWI object as defined in project.py.
This allows us to properly test everything in project.py, but will also be needed for the creation with new projects, together with some functions to translate and rotate objects.
'''
import re
from pathlib import Path
from typing import Any, cast
from shapely import Polygon, LineString,  equals
import math
from warnings import warn, catch_warnings

from .project import ObjectWI, FloorPlanWI, LineStringWI, MaterialWI, PolygonWI, check_boundary_contains,  ProjectWI, TxWI, RxWI
from .utils import round_to_significant_digits, merge_dicts, get_key_startswith, get_key_startswith_all

def read_word(
        word : str
    ) -> int | float | str | bool | None:
    """
    This function attempts to interpret the input string as an integer. If that
    fails, it tries to interpret it as a float. If both conversions fail, it 
    returns the input as a string.

    Args:
        word (str): The input string to be converted.

    Returns:
        int | float | str: The converted value as an integer, float, or the 
        original string if conversion is not possible.
    """
    if word=='yes':
        return True
    elif word=='no':
        return False
    elif word=='None':
        return None
    try:
        nr = int(word)
    except ValueError:
        try:
            # coordinates in the files may be slightly off, so we round 
            # for material properties this could be the wrong scale we round? But later we generate them anyways randomly from configs
            nr = round_to_significant_digits(float(word), 5) 
        except ValueError:
            nr = word
    return nr

def read_line(
        line : str
    ) -> dict[str, int | float | str | None] | list[int | float | str | None] | int | float | str | None:
    """
    Parses a line of text and returns its content in a structured format.

    Args:
        line (str): A single line of text to be parsed.

    Returns:
        dict[str, int | float | str]: If the line contains exactly two words, 
            returns a dictionary with the first word as the key and the parsed 
            second word as the value.
        list[int | float | str]: If the line contains more than two words, 
            returns a list of parsed words.
        int | float | str: If the line contains exactly one word, returns the 
            parsed word.
        None: If the line is empty or contains only whitespace.

    Notes:
        - Words are parsed as integers, floats, or strings based on their format.
        - Empty lines or lines with only whitespace are ignored and return `None`.
    """
    line_split = line.split()
    if len(line_split) == 2:
        return {line_split[0] : read_word(line_split[1])}
    elif len(line_split) > 2:
        return [read_word(w) for w in line_split]
    elif len(line_split) == 1:
        return read_word(line)
    else: 
        return None
    
def lines_to_dict(
        lines : list[str], 
        keyword_end : str | None
    ) -> tuple[dict[str|int|float, Any], list[str]]:
    """
    Parses a list of lines into a nested dictionary structure based on specific keywords.

    Args:
        lines (list[str]): A list of strings representing the lines to be parsed.
        keyword_end (str | None): A keyword indicating the end of a nested block. If None, 
                                    the function assumes it is parsing the top-level structure.

    Returns:
        tuple[dict[str | int | float, Any], list[str]]:
            - A dictionary containing the parsed content. Keys can be strings, integers, or floats, 
                and values can be nested dictionaries, lists, or other parsed content.
            - A list of remaining lines that were not processed.

    Raises:
        ValueError: If there are overlapping keys in nested dictionaries or if `keyword_end` 
                        is not None when the parsing is complete.

    Notes:
        - Lines starting with `begin_<` indicate the start of a nested block. The keyword following 
            `begin_<` is used as the key for the nested content.
        - Lines starting with `end_<` indicate the end of a nested block.
        - If a line's content is a dictionary, it is merged into the current dictionary. If it is 
            not a dictionary, it is added to a `properties` list within the dictionary.
        - The function ensures that keys in nested dictionaries do not overlap.
    """
    content = {}
    while len(lines) > 0:
        line = lines.pop(0)
        if line.startswith(f'end_<{keyword_end}'):
            return content, lines
        if line.startswith(f'begin_<'):
            key0 = line[7:line.find('>')]
            key = key0
            if len(line.split()) > 1:
                for k in line.split()[1:]:
                    key += f':{k}'
            line_content, lines = lines_to_dict(lines, keyword_end=key0)
            content = merge_dicts(content, {key : line_content})
            continue
        line_content = read_line(line)
        if isinstance(line_content, dict):
            if not content.keys().isdisjoint(line_content.keys()):
                raise ValueError(f' keys detected: {content.keys()=}\n{line_content.keys()=}')
            content.update(line_content)
        elif line_content is None:
            continue
        else:
            if 'properties'in content.keys():
                content['properties'].append(line_content)
            else:
                content['properties'] = [line_content]
        
    if keyword_end is not None:
        raise ValueError(f'{keyword_end=} is not None')
    return content, []
    
def read_wi_file(
        file : Path
    ) -> dict[str|int|float, Any]:
    """
    Reads a WI file and converts its content into a dictionary.
    Args:
        file (str | Path): The path to the WI file to be read.
    Returns:
        dict[str | int | float, Any]: A dictionary representation of the WI file content.
    Raises:
        Exception: If there are remaining lines that could not be processed 
                        after reading the entire file.
    """
    try:
        lines = file.read_text().splitlines()
    except Exception as e:
        raise Exception(f'{file=}\n{e}')
    output = lines_to_dict(lines, None)
    if len(output[1]) != 0:
        raise Exception(f'after reading the whole file, remaining {output[1]}')
    return output[0]

def extract_materials(
        file_content : dict, 
        name : str
    ) -> dict[int, dict[str, float|str]]:
    """
    Extracts material information from a nested dictionary structure.

    Args:
        file_content (dict): A dictionary containing the data to extract materials from.
        name (str): The key in `file_content` that contains the material data.

    Returns:
        dict[int, dict[str, float | str]]: A dictionary where the keys are material indices (int),
        and the values are dictionaries containing material properties such as:
            - 'material_type' (str): The type of the material.
            - 'name' (str): The name of the material.
            - 'conductivity' (float | None): The conductivity of the material, if available.
            - 'permittivity' (float | None): The permittivity of the material, if available.
            - 'roughness' (float | None): The roughness of the material, if available.
            - 'thickness' (float | None): The thickness of the material, if available.
            - 'nLayers' (float | None): The number of layers in the material, if available.
            - 'DielectricLayer' (float | None): The dielectric layer properties, if available.

    Raises:
        KeyError: If a required key is missing in the material data.

    Notes:
        - The function identifies materials by keys starting with 'Material:'.
        - If a property is not present in the material data, it is set to `None`.
    """
    materials = {}
    for k, v in file_content[name].items():
        if k.startswith('Material:'):
            try:
                ind = v['Material']
                materials[ind] = {
                    'material_type' : v['properties'][0],
                    'name'  :   k.replace('Material:', '')
                }
                for l in ['conductivity', 'permittivity', 'roughness', 'thickness', 'nLayers']:#, 'DielectricLayer']:
                    if l in v.keys():
                        materials[ind][l] = v[l]
                    else:
                        materials[ind][l] = None
                for l, w in v.items():
                    if l.startswith('DielectricLayer'):
                        materials[ind][l] = w
            except KeyError as e:
                print(f'{v.keys()=}')
                raise(e)
    
    return materials    

def extract_structure_groups(
        file_content: dict
    ) -> dict[str, int | str]:
    """
    Extracts all key-value pairs from a nested dictionary where the keys are strings
    that start with 'structure_group'. The function recursively traverses the input
    dictionary to find and merge matching key-value pairs from nested dictionaries.

    Args:
        file_content (dict): A dictionary potentially containing nested dictionaries
                                and keys starting with 'structure_group'.

    Returns:
        dict[str, int | str]: A dictionary containing all key-value pairs where the
                                keys start with 'structure_group', merged from all levels
                                of the input dictionary.
    """
    sg = {k : v for k, v in file_content.items() if isinstance(k, str) and k.startswith('structure_group')}
    sub_dicts = [extract_structure_groups(v) for v in file_content.values() if isinstance(v, dict)]
    for v in sub_dicts:
        sg = merge_dicts(sg, v)
    return sg

def extract_name(
        file_content: dict
    ) -> str:
    """
    Extracts the name key from a dictionary representing file content.

    This function identifies the key in the dictionary that is not 
    named 'properties'. It checks that there is exactly one such key 
    and returns it as the name.

    Args:
        file_content (dict): A dictionary containing file content, 
                                expected to have a single key other than 'properties'.

    Returns:
        str: The extracted name key from the dictionary.

    Raises:
        ValueError: If there is not exactly one key other than 'properties'.
    """
    keys = [k for k in file_content.keys() if not k=='properties']
    if len(keys) != 1:
        raise ValueError(f'Expected exactly one name key, but found {keys=}')
    return keys[0]

def create_geometry_from_substructure(
        material_dict_wi: dict[int, MaterialWI], 
        sub_structure_dict: dict[str, dict[str, int | list[list[float]]]],
        wall_material_index : int | None
    ) -> tuple[PolygonWI, list[LineStringWI]]:
    """
    Creates a geometry representation from a substructure definition.

    Args:
        material_dict_wi (dict[int, MaterialWI]): A dictionary mapping material
            indices to MaterialWI objects.
        sub_structure_dict (dict[str, dict[str, int | list[list[float]]]]): A
            dictionary defining the substructure. Each key represents a face,
            and its value contains properties such as material index, number of
            vertices, and coordinates.
        wall_material_index (int | None): The material index for walls. If provided, 
            faces with this material are ignored.

    Returns:
        tuple[PolygonWI, list[LineStringWI]]: A tuple containing:
            - A PolygonWI object representing the final polygon.
            - A list of LineStringWI objects representing filtered lines.

    Raises:
        ValueError: If the input data is invalid or inconsistent, such as:
            - A key in `sub_structure_dict` does not start with 'face'.
            - The number of vertices does not match the provided coordinates.
            - The top and bottom polygons have different geometries or materials.
            - A line is not part of the boundary or not contained in the polygon.

    Notes:
        - Vertical rectangles are identified by having two horizontal edges and two vertical edges.
        - Lines with materials different from the polygon are validated as part of the boundary.
    """
    polygons : list[PolygonWI] = []
    lines : list[LineStringWI]  = []
    for k, v in sub_structure_dict.items():
        if not k.startswith('face'):
            raise ValueError(f'{k=} is not a face')
        if k.startswith('face:'):
            k = k.replace('face:', '')
        material_index = cast(int, v['Material'])
        
        nvert = v['nVertices']
        coords = [c for c in v['properties'] if isinstance(c, list) or isinstance(c, tuple)] # type: ignore
        if not (isinstance(coords, list) and len(coords) == nvert):
            raise ValueError(f'Invalid coordinates: {coords=} but expected {nvert=}')

        if all([c[2]==coords[0][2] for c in coords]):
            polygons.append(PolygonWI(
                cast(Polygon, Polygon([c[:2] for c in coords]).normalize()), 
                material=material_dict_wi[material_index], 
                z_min=coords[0][2], 
                z_max=coords[0][2], 
                name=k))
        else:
            if material_index==wall_material_index:
                continue
            if not (len(coords) == 4 or (len(coords) == 5 and coords[0] == coords[-1])):
                raise ValueError(f'{coords=}, expected vertical rectangle')
            for i in range(4):
                p1, p2 = coords[i], coords[(i+1) % 4]
                if p1[2]==p2[2]:
                    if not (coords[(i+2)%4][:2] == p2[:2] and coords[(i+3)%4][:2] == p1[:2]):
                        raise ValueError(f'{coords=} not giving one vertical or horizontal line')
                    zs = [coords[(i+2)%4][2], p2[2]]
                    lines.append(LineStringWI(LineString([p1[:2], p2[:2]]), material=material_dict_wi[material_index], z_min=min(zs), z_max=max(zs), name=k))
                    break

    if len(polygons) != 2:
        warn(f'Expected exactly two polygons (top, bottom), but found {len(polygons)} with {[(p.name, p.z_min, p.z_max) for p in polygons]=}')
        if len(polygons) < 2:
            raise ValueError(f'Expected exactly two polygons (top, bottom), but found {len(polygons)} with {[(p.name, p.z_min, p.z_max) for p in polygons]}')
        # raise ValueError(f'Expected exactly two polygons (top, bottom), but found {len(polygons)} with {[p.z_min for p in polygons]}')
    # both polygons should be the same except for height and orientation

    if not equals(polygons[0].geometry, polygons[1].geometry):
        raise ValueError(f'top and bottom polygon have different geometries: {polygons[0].geometry=} != {polygons[1].geometry=}')
    if not polygons[0].material == polygons[1].material:
        raise ValueError(f'top and bottom polygon have different material: {polygons[0].material=} != {polygons[1].material=}')
    # we keep the first one, just need to adjust heights
    polygon_final = polygons[0]

    if polygon_final.z_min < polygons[1].z_min:
        polygon_final.z_max = polygons[1].z_min
    else:
        polygon_final.z_min = polygons[1].z_min
    # all lines should be either part of the boundary of the polygon, or some part of a wall (e.g. window/door for floorplans) with a different material, the latter we keep
    lines_to_keep = []
    for line in lines:
        if line.material != polygon_final.material:
            if not check_boundary_contains(polygon_final, line):
                # raise ValueError(f'{line=} not contained in {polygon_final=}')
                warn(f'line with {line.name=}, {line.material.name=} not contained in {polygon_final.name=}, {polygon_final.material.name=}')
            lines_to_keep.append(line)
        else:
            if not check_boundary_contains(polygon_final, line):
                coords = [c for c in polygon_final.geometry.boundary.coords]
                if len(coords)==0:
                    coords = []
                raise ValueError(f'line with {[c for c in line.geometry.coords]} and {line.z_min=}, {line.z_max=}, {line.name=} not part of the boundary of polygon_final with: \n\
                    {coords}, {polygon_final.z_min=}, {polygon_final.z_max=}, {polygon_final.name=}\nsee ./error_polygons_lines.png')

    return polygon_final, lines_to_keep

def create_object_from_structure_groups(
        material_dict_wi: dict[int, MaterialWI], 
        structure_group_dict: dict[str, dict[str, dict[str, dict[str, dict[str, int | list[list[float]]]]]]], 
        name: str
    ) -> ObjectWI:
    """
    Creates an ObjectWI instance from the provided structure groups and material dictionary.

    Args:
        material_dict_wi (dict[int, MaterialWI]): A dictionary mapping material IDs to MaterialWI objects.
        structure_group_dict (dict[str, dict[str, dict[str, dict[str, dict[str, int | list[list[float]]]]]]]): 
            A nested dictionary representing structure groups, where each group contains sub-structures 
            with their respective properties and geometry data.
        name (str): The name to assign to the created ObjectWI instance.

    Returns:
        ObjectWI: An instance of ObjectWI containing the geometry created from the structure groups.

    Raises:
        ValueError: If the keys in `structure_group_dict` do not start with 'structure_group' 
                        or if the `lines` list is not empty after geometry creation. Because this 
                        may only happen for floorplans. 
        Exception: If an error occurs during the creation of geometry from a sub-structure, 
                    providing details about the specific structure group causing the issue.
    """
    polygon_list = []
    for k, v in structure_group_dict.items():
        if not k.startswith(f'structure_group'):
            raise ValueError(f'expected only structure_groups, but {structure_group_dict.keys()=}')
        try:
            sub_structure_dict = get_key_startswith(v['structure'], 'sub_structure')
            if not isinstance(sub_structure_dict, dict):
                raise ValueError(f'{type(sub_structure_dict)=}')
            with catch_warnings(record=True) as caught_warnings:
                polygon, lines = create_geometry_from_substructure(material_dict_wi=material_dict_wi, sub_structure_dict=sub_structure_dict, wall_material_index=None)
                if caught_warnings:
                    warning = caught_warnings[-1]
                    warn(f"Creating ObjectWI with {name=}: {warning.message}")
        except Exception as e:
            raise Exception(f'for structure_group {k} in object {name} error: {e}')
        if not len(lines) == 0:
            w_msg = ''
            for l in lines:
                w_msg += f'\n\t{l.name}: {list(l.geometry.coords)},'
            warn(f'for object with {name=}, lines should be empty, but remaining: {w_msg}')
            # raise ValueError(f'for object, {lines=} should be empty')
        polygon_list.append(polygon)
    return ObjectWI(geometry_list=polygon_list, name=name)

def create_floorplan_from_structure_groups(
        material_dict_wi: dict[int, MaterialWI], 
        structure_group_dict: dict[str, dict[str, dict[str, dict[str, dict[str, int | list[list[float]]]]]]], 
        name: str,
        wall_material_index : int | None
    ) -> FloorPlanWI:
    """
    Creates a FloorPlanWI object from the provided material and structure group dictionaries.

    Args:
        material_dict_wi (dict[int, MaterialWI]): A dictionary mapping material IDs to MaterialWI objects.
        structure_group_dict (dict[str, dict[str, dict[str, dict[str, dict[str, int | list[list[float]]]]]]]): 
            A nested dictionary containing structure group data, including sub-structure geometry information.
        name (str): The name of the floor plan.

    Returns:
        FloorPlanWI: An instance of FloorPlanWI representing the generated floor plan.

    Raises:
        ValueError: If the length of `structure_group_dict` is not equal to 1.

    Warnings:
        - The function is not fully functional for floorplans, as wall materials are missing.
        - Additional wall parts may be present in the output, to be fixed.
    """
    if not len(structure_group_dict) == 1:
        raise ValueError(f'{len(structure_group_dict)=} != 1')
    if wall_material_index  is None:
        # warn(f'{wall_material_index=}, will use 0 with {material_dict_wi[0].name=}')
        wall_material_index = 0
    sub_structure_dict = get_key_startswith(structure_group_dict['structure_group']['structure'], 'sub_structure')
    if not isinstance(sub_structure_dict, dict):
        raise ValueError(f'{type(sub_structure_dict)=}')
    polygon, lines = create_geometry_from_substructure(material_dict_wi=material_dict_wi, sub_structure_dict=sub_structure_dict, wall_material_index=wall_material_index)
    return FloorPlanWI(floor=polygon, other=lines, name=name, material_walls=material_dict_wi[wall_material_index]) 

def create_materials_from_dict(
        material_dict: dict[int, dict[str, float | str]]
    ) -> dict[int, MaterialWI]:
    """
    Converts a dictionary of material properties into a dictionary of MaterialWI objects.

    Args:
        material_dict (dict[int, dict[str, float | str]]): 
            A dictionary where the keys are integers representing material IDs, 
            and the values are dictionaries containing material properties. 
            Each inner dictionary should have keys corresponding to the attributes 
            of the MaterialWI class.

    Returns:
        dict[int, MaterialWI]: 
            A dictionary where the keys are the same material IDs, and the values 
            are MaterialWI objects created using the properties from the input dictionary.

    Raises:
        TypeError: If the input dictionary contains invalid data types that do not 
                    match the expected attributes of the MaterialWI class.
    """
    material_dict_wi = {}
    for k, v in material_dict.items():
        material_dict_wi[k] = MaterialWI(**v) # type: ignore
    return material_dict_wi

def create_floorplan_from_file(
        file: Path,
        wall_material_index : int | None,
    ) -> FloorPlanWI:
    """
    Creates a FloorPlanWI object from a given file.

    This function reads the content of the specified file, extracts relevant
    information such as the name, materials, and structure groups, and uses
    this data to generate a FloorPlanWI object. If we know the index walls are
    made of, this helps to correctly create objects. Otherwise, all wall parts 
    are saved as additional objects inside the wall. NOTE: actually, we coud just
    use one random face from the wall, say that its material defines the wall and 
    everything else is cut into it?


    Args:
        file (Path): The path to the file containing the floorplan data.
        wall_material_index (int, optional): Index of wall material.

    Returns:
        FloorPlanWI: The generated floorplan object based on the file content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file content is invalid or missing required data.
    """
    file_content = read_wi_file(file)
    name = extract_name(file_content)
    materials = create_materials_from_dict(extract_materials(file_content, name))
    structure_groups = extract_structure_groups(file_content)
    return create_floorplan_from_structure_groups(material_dict_wi=materials, structure_group_dict=structure_groups, name=name.replace('floorplan:', ''), wall_material_index=wall_material_index) # type: ignore

def create_object_from_file(
        file: Path
    ) -> ObjectWI:
    """
    Creates an ObjectWI instance from the contents of a WI file.

    Args:
        file (Path): The path to the WI file to be processed.

    Returns:
        ObjectWI: An instance of ObjectWI created based on the file's contents.

    This function performs the following steps:
        1. Reads the contents of the WI file.
        2. Extracts the name of the object from the file content.
        3. Extracts material information and creates materials from the extracted data.
        4. Extracts structure groups from the file content.
        5. Creates and returns an ObjectWI instance using the extracted data.
    """
    file_content = read_wi_file(file)
    name = extract_name(file_content)
    materials = create_materials_from_dict(extract_materials(file_content, name))
    structure_groups = extract_structure_groups(file_content)
    return create_object_from_structure_groups(material_dict_wi=materials, structure_group_dict=structure_groups, name=name.replace('object:', '')) # type: ignore

def create_project_from_dir(project_dir : str | Path, wall_material_index : int | None = None) -> ProjectWI:
    """
    Creates a ProjectWI instance by reading and parsing files from the specified project directory.

    The function expects the directory to contain exactly one `.setup` file, which defines the project structure,
    including the floorplan, object files, transmitter/receiver (tx/rx) file, and additional properties.
    It constructs the project by:
        - Reading the setup file to determine required files.
        - Creating the floorplan from the specified file.
        - Adding geometry objects from the listed object files.
        - Adding transmitters and receivers if a tx/rx file is specified.

    Args:
        project_dir (str | Path): Path to the directory containing the project files.

    Returns:
        ProjectWI: An instance of ProjectWI initialized with the floorplan, objects, transmitters, receivers, and properties.

    Raises:
        FileNotFoundError: If there is not exactly one `.setup` file in the directory.
    """
    project_dir = Path(project_dir)
    setup_files = list(project_dir.glob('*.setup'))
    if not len(setup_files) == 1:
        raise FileNotFoundError(f'no or several setup files in {project_dir=}: {setup_files=}')
    floorplan_file, object_files, txrx_file, properties = read_setup_file(setup_file=setup_files[0])
    project = ProjectWI(create_floorplan_from_file(file=project_dir / floorplan_file, wall_material_index=wall_material_index), properties=properties)
    for object_file in object_files:
        project.add_geometry(create_object_from_file(project_dir / object_file))
    if txrx_file is not None:
        tx_list, rx_list = create_txrx_lists_from_file(project_dir / txrx_file)
        for rx in rx_list:
            project.add_rx(rx)
        for tx in tx_list:
            project.add_geometry(tx)
    return project

def create_txrx_lists_from_file(txrx_file : Path) -> tuple[list[TxWI], list[RxWI]]:
    """
    Parses a WI file to extract transmitter (Tx) and receiver (Rx) information and returns them as separate lists.

    Args:
        txrx_file (Path): Path to the WI file containing transmitter and receiver definitions.

    Returns:
        tuple[list[TxWI], list[RxWI]]: 
            - A list of TxWI objects representing transmitters.
            - A list of RxWI objects representing receivers.

    Raises:
        ValueError: If a key in the WI file cannot be clearly identified as a transmitter or receiver, 
                    or if the key is not a string.

    Notes:
        - Only transmitter points (keys containing 'points' and 'tx') and receiver grids (keys containing 'grid' and 'rx') are processed.
        - Emits a warning for any other types of entries found in the WI file.
    """
    txrx = read_wi_file(txrx_file)
    tx_list, rx_list = [], []
    for k, v in txrx.items():
        if not isinstance(k, str):
            raise ValueError(f'{k=}')
        if 'points' in k and 'tx' in k: 
            if not (v['is_transmitter'] and not v['is_receiver']):
                raise ValueError(f'{k=}, {v=} not clearly a Tx or Rx?')
            coords = v['location']['properties'][0]
            tx_list.append(TxWI(coords, k.replace('points:', '')))
        elif 'grid' in k and 'rx' in k:
            if not (not v['is_transmitter'] or v['is_receiver']):
                raise ValueError(f'{k=}, {v=} not clearly a Tx or Rx?')
            height, spacing = v['location']['properties'][0][2], v['location']['spacing']
            rx_list.append(RxWI(height=height, spacing=spacing, name=k.replace('grid:', '')))
        else:
            warn(f'currently we only read Tx points and Rx grids, found {k}?')
    return tx_list, rx_list

def read_setup_file(setup_file : Path) -> tuple[str, list[str], str|None, dict[str, float]]:
    """
    Reads a setup file and extracts the floorplan file, object files, and study area properties.

    Args:
        setup_file (Path): Path to the setup file to be read.

    Returns:
        tuple[str, list[str], dict[str, float]]: 
            - The filename of the floorplan file (str).
            - A list of filenames for object files (list of str).
            - Filename of the txrx file, potentially None.
            - A dictionary containing study area properties such as 'temperature', 'humidity', and 'pressure' (dict of str to float).

    Raises:
        ValueError: If the setup file content is not as expected, such as missing or multiple floorplan files, 
                    or if required keys are not present in the file.

    Notes:
        - The function expects the setup file to contain specific keys and structure.
        - Issues a warning if a feature with unrecognized properties is encountered.
    """
    setup = get_key_startswith(read_wi_file(setup_file), 'project')
    if not isinstance(setup, dict):
        raise ValueError(f'content {setup=} from {setup_file} is not a dict')
    study_area = get_key_startswith(setup, 'studyarea')
    if not isinstance(study_area, dict):
        raise ValueError(f'{study_area=} from {setup_file} is not a dict')
    properties = {k : study_area['model'][k] for k in ['temperature', 'humidity', 'pressure']}
    # find the correct files
    feature_list = get_key_startswith_all(setup, 'feature')
    floorplan_file, object_files = None, []
    for feature_dict in feature_list:
        if 'object' in feature_dict['properties'] and feature_dict['filename'].endswith('.object'):
            object_files.append(feature_dict['filename'])
        elif 'floorplan' in feature_dict['properties'] and feature_dict['filename'].endswith('.flp'):
            if floorplan_file is not None:
                raise ValueError(f'{floorplan_file=} but also found {feature_dict=} in {setup_file}')
            floorplan_file = feature_dict['filename']
        else:
            warn(f'found feature with {feature_dict['properties']=} we could not interpret')
    if not isinstance(floorplan_file, str):
        raise ValueError(f'{floorplan_file=}')
    txrx_file = setup.get('txrx_sets', {}).get('filename', None)
    return floorplan_file, object_files, txrx_file, properties
    