import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from typing import Sequence
import numpy as np
from pathlib import Path
from shapely import LineString, Polygon
from geo_rasterize import rasterize as rasterize_geo
from PIL import Image
import h5py
import json

from .project_creation import create_project_from_dir
from .project import ProjectWI, GeometryWI, LineStringWI, PolygonWI
from .utils import dBm_W, W_dBm

material_properties_considered = ['conductivity', 'permittivity', 'thickness']

#############
### we use geo_rasterize instead of rasterio, because it doesnt rely on GDAL, so the installation is  a lot more compact
### in my tests, the results looked the same and the runtime was also comparable
# from rasterio.features import rasterize as rasterize_io
# from rasterio.transform import AffineTransformer, Affine
#############

def rasterize_project_from_dir(
        project_dir: Path,
        out_dir: Path,
        x_max: float, 
        y_max: float, 
        x_steps: int, 
        y_steps: int,
        heights: Sequence[float],
        names_to_hide: list[str],
        materials: Sequence[str | Sequence[str]],
        properties_min_max: dict | None
    ) -> None:
    """
    Rasterizes a project from a given directory and saves the resulting images and metadata to the specified output directory.
    This function generates raster images representing class labels and material properties at specified heights, 
    and saves them as PNG files. It also exports transformation metadata as a JSON file. If the output files already exist, 
    the function will not overwrite them.
    Args:
        project_dir (Path): Path to the directory containing the project data.
        out_dir (Path): Path to the directory where output files will be saved.
        x_max (float): Maximum x-coordinate for rasterization.
        y_max (float): Maximum y-coordinate for rasterization.
        x_steps (int): Number of steps (pixels) along the x-axis.
        y_steps (int): Number of steps (pixels) along the y-axis.
        heights (Sequence[float]): List of heights at which to rasterize the project.
        names_to_hide (list[str]): List of object names to exclude from rasterization.
        materials (Sequence[str | Sequence[str]]): List of material names or sequences of material names to consider.
        properties_min_max (dict | None): Dictionary specifying the min and max values for material properties, 
            used for normalization. If None, material properties are not rasterized.
    Returns:
        None
    """
    ### define output files and check whether they exist already
    file_names_props = [out_dir / f'{project_dir.stem}_height{height}_{prop}.png' for height in heights for prop in material_properties_considered]
    raster_imgs = [out_dir / f'{project_dir.stem}_height{height}_classes.png' for height in heights]
    tx_file = out_dir / f'{project_dir.stem}_tx.json'

    if all(f.exists() for f in raster_imgs + file_names_props + [tx_file]):
        return

    project = create_project_from_dir(project_dir)
    array_dict_classes = rasterize_project_classes(
        project=project, 
        x_max=x_max, 
        y_max=y_max, 
        x_steps=x_steps, 
        y_steps=y_steps, 
        heights=heights, 
        names_to_hide=names_to_hide, 
        materials=materials
    )

    for height, arr in array_dict_classes.items():
        file_name = out_dir / f'{project_dir.stem}_height{height}_classes.png'
        if file_name.exists():
            # raise FileExistsError(f'{file_name=} already is a file!')
            continue
        image_cl = Image.fromarray(arr)
        image_cl.save(file_name)

    
    if properties_min_max is not None and not all(f.exists() for f in file_names_props):
        array_dict_mat_props = rasterize_project_material_properties(
            project=project, 
            x_max=x_max, 
            y_max=y_max, 
            x_steps=x_steps, 
            y_steps=y_steps, 
            heights=heights, 
            names_to_hide=names_to_hide, 
        )

        array_dict_mat_props_gray_scale = convert_material_properties_to_gray_scale(
            array_dict_mat_props=array_dict_mat_props,
            properties_min_max=properties_min_max
        )

        for (prop, height), array in array_dict_mat_props_gray_scale.items():
            file_name_prop = out_dir / f'{project_dir.stem}_height{height}_{prop}.png'
            if file_name_prop.exists():
                continue
            image_prop = Image.fromarray((array * 255).astype(np.uint8))
            image_prop.save(file_name_prop)

    
    if not tx_file.exists():
        with open(tx_file, 'w') as f:
            json.dump({
                i : [tx.geometry.coords[0][1], tx.geometry.coords[0][0], tx.z_min] for i, tx in enumerate(project.tx)
            }, f)

def get_min_max_properties_from_config(config: dict) -> dict[str, list[int | float]]:
    """
    Computes the minimum and maximum values for each material property specified in `material_properties_considered`
    from a given configuration dictionary.

    Args:
        config (dict): A dictionary where keys are material names and values are dictionaries containing
            material properties (fixed values or ranges).

    Returns:
        dict: A dictionary mapping each property name to a list [max_value, min_value], representing the
            maximum and minimum values found for that property across all materials.

    Raises:
        ValueError: If a property value is found that is neither a numeric type nor a sequence of numerics.
    """
    properties_min_max = {prop : [1e6, -1] for prop in material_properties_considered}
    for mat_name, mat_props in config.items():
        for prop, (mini, maxi) in properties_min_max.items():
            if prop in mat_props.keys():
                mat_val = mat_props[prop]
                if isinstance(mat_val, float | int):
                    properties_min_max[prop][0] = min(mini, mat_val)
                    properties_min_max[prop][1] = max(maxi, mat_val)
                elif isinstance(mat_val, Sequence):
                    properties_min_max[prop][0] = min(mini, min(mat_val))
                    properties_min_max[prop][1] = max(maxi, max(mat_val))
                else:
                    raise ValueError(f'in given material properties dict found {mat_val=} for {mat_name=} ({mat_props=})')
            else:
                for dl in ['DielectricLayer', 'DielectricLayer0']:
                    if dl in mat_props.keys():
                        mat_props_dl = mat_props[dl]
                        if prop in mat_props_dl.keys():
                            mat_val = mat_props_dl[prop]
                            if isinstance(mat_val, float | int):
                                properties_min_max[prop][0] = min(mini, mat_val)
                                properties_min_max[prop][1] = max(maxi, mat_val)
                            elif isinstance(mat_val, Sequence):
                                properties_min_max[prop][0] = min(mini, min(mat_val))
                                properties_min_max[prop][1] = max(maxi, max(mat_val))
                            else:
                                raise ValueError(f'in given material properties dict found {mat_val=} for {mat_name=} ({mat_props=})')
    return properties_min_max

def convert_material_properties_to_gray_scale(
        array_dict_mat_props: dict,
        properties_min_max: dict
    ) -> dict:
    """
    Converts material property arrays to grayscale values normalized between 0 and 1 using the corresponding 
    min and max values from `properties_min_max`. For the 'conductivity' property, normalization is performed 
    in log10 space. For all other properties, linear normalization is applied.
    Args:
        array_dict_mat_props (dict): Dictionary where keys are tuples of (property name, height)
            and values are numpy arrays representing material property values.
        properties_min_max (dict): Dictionary mapping property names to a tuple of (max, min) values
            used for normalization.
    Returns:
        dict: The input dictionary with arrays normalized to grayscale values in the range [0, 1].
    """
    for (prop, height), array in array_dict_mat_props.items():
        mini, maxi = properties_min_max[prop]
        array[np.isposinf(array)] = maxi
        array[np.isneginf(array)] = mini
        if not np.amax(array) <= maxi:
            warnings.warn(f'{prop=} values {np.amax(array)=} larger than {maxi=}')
        if not np.amin(array) >= mini:
            warnings.warn(f'{prop=} values {np.amin(array)=} smaller than {mini=}')
        if prop == 'conductivity':
            array_dict_mat_props[(prop, height)] = (np.log10(array + 1) - np.log10(mini + 1)) / (np.log10(maxi + 1) - np.log10(mini + 1))
        else:
            array_dict_mat_props[(prop, height)] = (array - mini) / (maxi - mini)
    return array_dict_mat_props
        



    
    return properties_min_max


def rasterize_project_classes(
        project : ProjectWI,
        x_max : float, 
        y_max : float, 
        x_steps : int, 
        y_steps : int,
        heights : Sequence[float],
        names_to_hide : list[str],
        materials : Sequence[str|Sequence[str]],
    ) -> dict[float,np.ndarray]:
    """
    Rasterizes a project's floorplan and object geometries per material and height slice.

    Parameters:
        project: A ProjectWI instance containing geometries to rasterize.
        x_max, y_max: Physical extent of the raster domain in X and Y (world units).
        x_steps, y_steps: Number of pixels in X and Y.
        heights: Monotonically increasing list of height values defining height intervals [height_min, height_max] we check for the geometries.
        names_to_hide: List of substrings; geometries whose names match are excluded.
        materials: List of material name prefixes to include in rasterization. Several materials can be put into one category, wich is useful for e.g. concrete and drywall to model unknown wall material.

    Returns:
        Dictionary mapping height_min to rasterized 2D numpy arrays 
        (shape: [y_steps, x_steps]) containing class values: 
        0 - empty,
        1 - materials[0],
        2 - materials[1],
        ...
    
    Raises:
        Depends on downstream rasterization errors if inputs are malformed.
    """
    floorplan, objects = project.floorplan, project.objects
    ### create list of the geometries we want to rasterize, excluding the ones pointed out by names_to_hide
    geometry_list = []
    for geom in list(floorplan.get_contained_geometries()) + [sg for o in objects for sg in o.get_contained_geometries()]:
        if not any(n in geom.name for n in names_to_hide):
            geometry_list.append(geom)
    ### reorder and split up, because last values will overwrite previous ones!
    geometry_lists_split = [[g for g in geometry_list if isinstance(g, LineStringWI) and 'wall' in g.name] 
                            + [g for g in geometry_list if isinstance(g, PolygonWI)] 
                            + [g for g in geometry_list if isinstance(g, LineStringWI) and not 'wall' in g.name]]
    assert len(geometry_list) == sum(len(l) for l in geometry_lists_split), f'{geometry_lists_split=}\n\n{geometry_list=}'

    array_dict_combined = {height_min : np.zeros((x_steps, y_steps), dtype=np.uint8) for height_min in heights}

    for idh in range(len(heights)):
        height_min, height_max = heights[idh], heights[idh + 1] if idh < len(heights) - 1 else np.inf
        for geometry_sub_list in geometry_lists_split:
            for idm, material in enumerate(materials, 1):
                if isinstance(material, str):
                    array_binary = rasterize_geometries_slice(
                                [g.geometry for g in geometry_sub_list if g.material.name.startswith(material) and g.z_min <= height_max and height_min <= g.z_max], 
                                x_max, y_max, x_steps, y_steps)
                elif isinstance(material, Sequence):
                    array_binary = rasterize_geometries_slice(
                                [g.geometry for g in geometry_list if any(g.material.name.startswith(m) for m in material) and g.z_min <= height_max and height_min <= g.z_max], 
                                x_max, y_max, x_steps, y_steps)
                array_dict_combined[height_min] = fuse_arrays(array_dict_combined[height_min], idm * array_binary, 0)
    return array_dict_combined

def fuse_arrays(
        array_previous : np.ndarray, 
        array_new : np.ndarray,
        default_value : int | float
    ) -> np.ndarray:
    """
    Fuses two arrays by updating the previous array with non-default values from the new array.

    Parameters:
        array_previous (np.ndarray): The previous array to be updated.
        array_new (np.ndarray): The new array whose non-default values will overwrite the previous array.
        default_value (int | float): Default value 

    Returns:
        np.ndarray: The updated array after fusion.

    Raises:
        ValueError: If the shapes of the input arrays do not match.
    """
    if not array_previous.shape == array_new.shape:
        raise ValueError(f'{array_previous.shape=} != {array_new.shape=}')
    array_previous[array_new != default_value] = array_new.astype(array_previous.dtype)[array_new != default_value]
    return array_previous

def rasterize_project_material_properties(
        project : ProjectWI,
        x_max : float, 
        y_max : float, 
        x_steps : int, 
        y_steps : int,
        heights : Sequence[float],
        names_to_hide : list[str],
    ) -> dict[tuple[str,float],np.ndarray]:
    """
    Rasterizes a project's floorplan and object geometries per material and height slice.

    Parameters:
        project: A ProjectWI instance containing geometries to rasterize.
        x_max, y_max: Physical extent of the raster domain in X and Y (world units).
        x_steps, y_steps: Number of pixels in X and Y.
        heights: Monotonically increasing list of height values defining height intervals [height_min, height_max] we check for the geometries.
        names_to_hide: List of substrings; geometries whose names match are excluded.
        materials: List of material name prefixes to include in rasterization. Several materials can be put into one category, wich is useful for e.g. concrete and drywall to model unknown wall material.

    Returns:
        Dictionary mapping (material_name, height_min) to rasterized 2D numpy arrays 
        (shape: [y_steps, x_steps]) containing binary masks.
    
    Raises:
        Depends on downstream rasterization errors if inputs are malformed.
    """
    floorplan, objects = project.floorplan, project.objects
    ### create list of the geometries we want to rasterize, excluding the ones pointed out by names_to_hide
    geometry_list : list[GeometryWI]= []
    for geom in list(floorplan.get_contained_geometries()) + [sg for o in objects for sg in o.get_contained_geometries()]:
        if not any(n in geom.name for n in names_to_hide) and geom.material is not None:
            geometry_list.append(geom)
    for g in geometry_list:
        assert g.material is not None
        if g.material.get_permittivity() is None:
            print(f'for {g.name=} got perm None {g.material.name=}, {g.material.get_properties()=}')
        if g.material.get_conductivity() is None:
            print(f'for {g.name=} got cond None {g.material.name=}, {g.material.get_properties()=}')
        if g.material.get_thickness() is None:
            print(f'for {g.name=} got thickness None {g.material.name=}, {g.material.get_properties()=}')

    ### reorder and split up, because last values will overwrite previous ones!
    geometry_lists_split = [[g for g in geometry_list if isinstance(g, LineStringWI) and 'wall' in g.name] 
                            + [g for g in geometry_list if isinstance(g, PolygonWI)] 
                            + [g for g in geometry_list if isinstance(g, LineStringWI) and not 'wall' in g.name]]
    assert len(geometry_list) == sum(len(l) for l in geometry_lists_split), f'{geometry_lists_split=}\n\n{geometry_list=}'

    array_dict = {(mp, h) : np.zeros((x_steps, y_steps), dtype=np.float32) for mp in ['conductivity', 'thickness'] for h in heights}
    array_dict.update({('permittivity', h) : np.ones((x_steps, y_steps), dtype=np.float32) for h in heights})

    for geometry_sub_list in geometry_lists_split:
            
        permittivities = list(set([g.material.get_permittivity() for g in geometry_sub_list if g.material is not None and g.material.get_permittivity() is not None]))
        conductivities = list(set([g.material.get_conductivity() for g in geometry_sub_list if g.material is not None and g.material.get_conductivity() is not None]))
        thicknesses = list(set([g.material.get_thickness() for g in geometry_sub_list if g.material is not None and g.material.get_thickness() is not None]))

        for idh in range(len(heights)):
            height_min, height_max = heights[idh], heights[idh + 1] if idh < len(heights) - 1 else np.inf
            
            for perm in permittivities:
                perm_slice = rasterize_geometries_slice(
                        [g.geometry for g in geometry_sub_list if g.material is not None and g.material.get_permittivity()==perm and g.z_min <= height_max and height_min <= g.z_max], 
                        x_max, y_max, x_steps, y_steps)
                array_dict[('permittivity', height_min)] = fuse_arrays(array_dict[('permittivity', height_min)], np.where(perm_slice, perm, 1), 1)
            for cond in conductivities:
                cond_slice = rasterize_geometries_slice(
                        [g.geometry for g in geometry_sub_list if g.material is not None and g.material.get_conductivity()==cond and g.z_min <= height_max and height_min <= g.z_max], 
                        x_max, y_max, x_steps, y_steps)
                array_dict[('conductivity', height_min)] = fuse_arrays(array_dict[('conductivity', height_min)], np.where(cond_slice, cond, 0), 0)
            for thickn in thicknesses:
                th_slice = rasterize_geometries_slice(
                        [g.geometry for g in geometry_sub_list if g.material is not None and g.material.get_thickness()==thickn and g.z_min <= height_max and height_min <= g.z_max], 
                        x_max, y_max, x_steps, y_steps)
                array_dict[('thickness', height_min)] = fuse_arrays(array_dict[('thickness', height_min)], np.where(th_slice, thickn, 0), 0)

    return array_dict

def rasterize_geometries_slice(
        geometry_list : Sequence[Polygon|LineString], 
        x_max : float, 
        y_max : float, 
        x_steps : int, 
        y_steps : int,
    ) -> np.ndarray:
    """
    Rasterizes a list of 2D geometries onto a pixel grid.

    Parameters:
        geometry_list: List of shapely Polygon or LineString objects.
        x_max, y_max: Physical extent of the domain in X and Y (world units).
        x_steps, y_steps: Resolution of the output raster (in pixels).
        use_rasterio: If True, uses rasterio for rasterization; otherwise uses fallback method.
Z
    Returns:
        A binary numpy array of shape (y_steps, x_steps), where 1 indicates geometry coverage.

    Raises:
        ValueError: If non-Polygon/LineString geometries are provided.
    """
    if not all(isinstance(g, (Polygon, LineString)) for g in geometry_list):
        raise ValueError(f'Only takes Polygon and LineString but got {[type(g) for g in geometry_list if not isinstance(g, (Polygon, LineString))]}')
    # define geo_to_pix transform to map geographical coordinates to pixel coordinates
    # this transform is not well document, it seems that:
    # X_pix = b * X_geo + d * Y_geo + f
    # Y_pix = a * X_geo + c * Y_geo + e
    # for a transform (a, b,..., f)
    x_step_size = x_max / x_steps
    y_step_size = y_max / y_steps

    gtp = (0, 1 / x_step_size, 1 / y_step_size, 0, 0, 0)
    return rasterize_geo(geometry_list, len(geometry_list) * [1], (x_steps, y_steps), 0, algorithm='replace', geo_to_pix=gtp)
    
def combine_materials_per_slice(
        array_dict : dict[tuple[str,float],np.ndarray],
        heights : Sequence[float],
        materials : Sequence[str|Sequence[str]]
    ) -> np.ndarray:
    """
    Combines per-material, per-height-slice raster arrays into a single labeled 3D array.

    Parameters:
        array_dict: Dictionary mapping (material_name, height_min) to 2D binary masks.
        heights: List of height breakpoints; must match keys in array_dict.
        materials: List of material names used as keys in array_dict.

    Returns:
        A 3D numpy array of shape (len(heights)-1, H, W), where each [i, :, :] slice contains
        integer labels: 0 for empty, 1..N for the i-th material.

    Raises:
        ValueError: If array_dict keys do not match the expected combinations of materials and height intervals.
    """
    materials = [m if isinstance(m, str) else '_'.join(mm for mm in m) for m in materials]
    ### make sure that we have the correct heights and materials in the dict
    dict_keys = set(array_dict.keys())
    mats_heights = set((m, h) for m in materials for h in heights)
    if not dict_keys == mats_heights:
        raise ValueError(f'dict keys and materials+heights do not match: {dict_keys - mats_heights=}, {mats_heights - dict_keys=}')
    ###  if the arrays are shape H x W and we have N_h height intervals (N_h+1 values in heights), we create an array of shape N_h x H x W with values 0, 1, 2,...,len(materials)
    arr_shape = next(iter(array_dict.values())).shape
    class_array = np.zeros((len(heights), *arr_shape), dtype=np.uint8) 
    for idm, mat in enumerate(materials, 1):
        for idh, h in enumerate(heights):
            class_array[idh, array_dict[(mat, h)].astype('bool')] = idm
    return class_array
