from pathlib import Path
from yaml import safe_load
from shutil import rmtree, copytree
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
import json
from joblib import delayed, Parallel
from typing import cast

from modules import project_creation as pc, write_wi_files as wwf

def main(
        config: str | Path, 
        materials: bool, 
        project_properties: bool, 
        object_offset: float, 
        n_copies: int, 
        xml_glob: str, 
        project_dir: Path, 
        n_jobs: int,
        out_dir_str : str | None
    ) -> None:
    """
    Generates randomized copies of indoor projects based on the provided configuration and parameters.
    Args:
        config (str | Path): Path to the YAML configuration file used for randomizing the project and material properties.
        materials (bool): Whether to randomize materials in the copied projects.
        project_properties (bool): Whether to randomize project properties.
        object_offset (float): Maximum offset to apply to objects for randomization.
        n_copies (int): Number of copies to generate for each project.
        xml_glob (str): Glob pattern to match XML files for simulations.
        project_dir (Path): Directory containing the original projects.
        n_jobs (int): Number of parallel jobs to use for processing.
        out_dir_str (str | None): Give explicit subdir to continue generation of a set of copies already started previously. Otherwise,
            a new directory will be created.
    Raises:
        ValueError: If no randomization options are enabled (materials, project_properties, object_offset).
    Side Effects:
        - Creates a new output directory for the copies.
        - Generates randomized copies for each valid project in `project_dir`.
        - Saves the parameters of the copy process to a JSON file in the output directory.
        - Prints the number of projects processed and copies created.
    """

    if not project_properties and not materials and object_offset==0:
        raise ValueError(f'With the given parameters, we do not randomize anything')

    cfg_path = Path(config)

    with open(cfg_path, 'r') as f:
        cfg = safe_load(f)

    projects_found = 0
    

    
    if out_dir_str is None:
        out_dir = project_dir / f'copies_{time.strftime("%Y%m%d-%H%M%S")}'
        out_dir.mkdir()
    else:
        out_dir = project_dir / out_dir_str
        out_dir.mkdir(exist_ok=True)
        
    params = {
                'materials' : materials,
                'project_properties' : project_properties,
                'config' : str(config),
                'object_offset' : object_offset,
                'xml_glob' : xml_glob,
                'n_copies' : n_copies
            }
    json_file = out_dir / 'copy_parameters.json'
    if (json_file).exists():
        with open(json_file, 'r') as f:
            params_previous = json.load(f)
        ### make sure that we continue making copies with the same configuration   
        if not params_previous == params:
            raise ValueError(f'{params=}\n{params_previous=}')
    else:
        ### save parameters of the copy process
        with open(json_file, 'w') as f:
            json.dump(params, f)

    def process_project(p : Path) -> int:
        if not p.is_dir() or 'copies' in p.stem:
            return 0  # no valid project processed
        pc.create_random_copies(
            project_dir=p,
            cfg=cfg,
            materials=materials,
            project_properties=project_properties,
            object_offset=object_offset,
            n_copies=n_copies,
            xml_glob=xml_glob,
            out_dir=out_dir
        )
        return 1  # count this project
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_project)(p) for p in project_dir.iterdir()
    )

    projects_found = sum(cast(list[int],results))
    print(f'Created {n_copies} copies of {projects_found} projects each (including existing ones we skipped).')

    

    
if __name__ == "__main__":
    parser = ArgumentParser(description="Create copies of indoor projects.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='configs/config1.yml', help='path to config')
    parser.add_argument('-m', '--materials', action='store_true', help='Activate to randomize material properties')
    parser.add_argument('-pp', '--project_properties', action='store_true', help='Activate to randomize project properties')
    parser.add_argument('-oo', '--object_offset', type=float, default=0.5, help='Offset for moving objects')
    parser.add_argument('-n', '--n_copies', type=int, default=10, help='Number of copies per project to be created')
    parser.add_argument('-x', '--xml_glob', type=str, default='x3d3_3_1*.xml', help='This glob will be used to find xml files in ./wi_templates, we use each template found with each Tx-Rx-combination')
    parser.add_argument('-d', '--project_dir', type=str, default='./indoor_projects_filtered_250922/project_files', help='Where to find the original projects')
    parser.add_argument('-o', '--out_dir', type=str, default=None, help='Subdir of project_dir to save outputs, leave as None to create  a new one.')
    parser.add_argument('-nj', '--n_jobs', type=int, default=24, help='Number of kernels for multiprocessing')
    args = parser.parse_args()
    
    main(
        config=args.config, 
        materials=args.materials,
        project_properties=args.project_properties,
        object_offset=args.object_offset,
        n_copies=args.n_copies, 
        xml_glob=args.xml_glob, 
        project_dir=Path(args.project_dir), 
        n_jobs=args.n_jobs,
        out_dir_str=args.out_dir
    )