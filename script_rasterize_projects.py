import yaml
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import json
from joblib import delayed, Parallel
from typing import cast, Sequence

from modules import processing_to_images as pti

def main(
        project_dir : Path,
        out_dir : Path,
        x_max : float, 
        y_max : float, 
        x_steps : int, 
        y_steps : int,
        heights : Sequence[float],
        names_to_hide : list[str],
        binary : bool,
        n_jobs : int,
        config_for_materials : Path | None
    ) -> None:
    """
    Rasterizes indoor projects found in a directory and saves the resulting images and parameters.
    Args:
        project_dir (Path): Path to the directory containing project folders.
        out_dir (Path): Path to the output directory where raster images and parameters will be saved.
        x_max (float): Maximum x-coordinate for rasterization.
        y_max (float): Maximum y-coordinate for rasterization.
        x_steps (int): Number of steps along the x-axis for rasterization.
        y_steps (int): Number of steps along the y-axis for rasterization.
        heights (Sequence[float]): List of heights at which to rasterize the projects.
        names_to_hide (list[str]): List of object names to exclude from rasterization.
        binary (bool): Create binary images instead of all material classes.
        material_properties (bool): Additionally create rasters of electromagnetic amterial properties.
        n_jobs (int): Number of parallel jobs to use for processing projects.
        config_for_materials (Path|None): Config from which material property max and min values are read to create rasters with material properties.
                Ommit in order to not create them.
    Returns:
        None
    Side Effects:
        - Creates raster images for each valid project in `out_dir`.
        - Saves rasterization parameters to 'copy_parameters.json' in `out_dir`.
        - Prints the number of projects processed.
    """

    projects_found = 0

    out_dir.mkdir(exist_ok=True)

    ### this construction is a bit weird due to legacy reasons
    materials = [['concrete', 'drywall', 'wood', 'metal', 'glass']] if binary else [['concrete', 'drywall'], 'wood', 'metal', 'glass']

    params_file = out_dir / 'rasterization_parameters.json'
    params = {
            'materials' : materials,
            'x_max' : x_max,
            'y_max' : y_max,
            'x_steps' : x_steps,
            'y_steps' : y_steps,
            'heights' : heights,
            'names_to_hide' : names_to_hide
        }
    if config_for_materials is not None:
        with open(config_for_materials, 'r') as f:
            cfg_materials = yaml.safe_load(f)['materials']
        properties_min_max = pti.get_min_max_properties_from_config(cfg_materials)
        params.update(properties_min_max)
    else:
        properties_min_max = None

    if params_file.exists():
        with open(params_file, 'r') as f:
            params_previous = json.load(f)
            assert params == params_previous, f'{params=}, \nbut\n {params_previous=}\nTo rasterize projects with different parameters, choose a different directory!'
    else:
        ### save parameters of the copy process
        with open(params_file, 'w') as f:
            json.dump(params, f)

    def process_project(p : Path) -> tuple[Path, bool, str | None]:
        try:
            if not p.is_dir() or 'copies' in p.stem:
                return (p, False, None)  # no valid project processed
            pti.rasterize_project_from_dir(
                project_dir=p,
                out_dir=out_dir,
                x_max=x_max,
                y_max=y_max,
                x_steps=x_steps,
                y_steps=y_steps,
                heights=heights,
                names_to_hide=names_to_hide,
                materials=materials,
                properties_min_max=properties_min_max
            )
            return (p, True, None)  # count this project
        except Exception as e:
            return (p, False, str(e))

    results = cast(list[tuple], Parallel(n_jobs=n_jobs)(
        delayed(process_project)(p) for p in project_dir.iterdir() 
    ))

    projects_found = sum(1 for _, success, _ in results if success)
    print(f'Created raster images for {projects_found} projects each.')

    failed = [(p, msg) for p, success, msg in results if not success and msg is not None]
    if failed:
        failure_file = out_dir / 'failures.txt'
        with open(failure_file, 'w') as f:
            print("Failures:")
            for p, msg in failed:
                print(f" - {p}: {msg}")
                f.write(f" - {p}: {msg}")
        print(f'saved failures to {failure_file}')
        

    
if __name__ == "__main__":
    parser = ArgumentParser(description="Rasterize indoor projects to images. By default, creates images with classes. Use the flags -b, -c for binary respectively material property images.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--project_dir', type=str, default='./indoor_projects_filtered_250922/project_files', help='Where to find the original projects')
    parser.add_argument('-o', '--out_dir', type=str, default='./indoor_projects_filtered_250922/rasterized_projects_256x256', help='Output directory for rasterized images')
    parser.add_argument('-xmax', '--x_max', type=float, default=9.6, help='Maximum x coordinate for rasterization')
    parser.add_argument('-ymax', '--y_max', type=float, default=9.6, help='Maximum y coordinate for rasterization')
    parser.add_argument('-xs', '--x_steps', type=int, default=256, help='Number of steps in x direction')
    parser.add_argument('-ys', '--y_steps', type=int, default=256, help='Number of steps in y direction')
    parser.add_argument('-z', '--heights', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, \
                                                                           1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8], help='Heights at which to rasterize')
    parser.add_argument('-nh', '--names_to_hide', type=str, nargs='*', default=['metal_plate'], help='Names of objects to hide during rasterization')
    parser.add_argument('-b', '--binary', action='store_true', help='Activate to create binary encodings. Otherwise, classes will be used for rasterization.')
    parser.add_argument('-nj', '--n_jobs', type=int, default=24, help='Number of parallel jobs for multiprocessing')
    parser.add_argument('-c', '--config-for-materials', type=str, help='Config to find material properties to create pngs with the properties encoded (optional). Without a valid config, mp rasters are not created.')
    args = parser.parse_args()
    
    main(
        project_dir=Path(args.project_dir),
        out_dir=Path(args.out_dir),
        x_max=args.x_max,
        y_max=args.y_max,
        x_steps=args.x_steps,
        y_steps=args.y_steps,
        heights=args.heights,
        names_to_hide=args.names_to_hide,
        binary=args.binary,
        material_properties=args.material_properties,
        n_jobs=args.n_jobs,
        config_for_materials=Path(args.config_for_materials) if args.config_for_materials is not None else None
    )