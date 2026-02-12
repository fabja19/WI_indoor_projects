from pathlib import Path
from yaml import safe_load
from modules import project_creation as pc, write_wi_files as wwf
from shutil import rmtree, copytree
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def main(config : str | Path, number : int, xml_glob : str, project_dir : Path, overwrite : bool) -> None:
    cfg_path = Path(config)

    with open(cfg_path, 'r') as f:
        cfg = safe_load(f)

    n_tx = 0

    if project_dir.is_dir() and overwrite:
        rmtree(project_dir)

    for i in range(number):
        project_dir_here = project_dir / f'project{i}'
        if project_dir_here.exists():
            if overwrite:
                rmtree(project_dir_here)
            else:
                print(f'{project_dir_here} exists already, skipping')
                continue
        project, cfg_here = pc.create_project_randomly(cfg)
        n_tx += len(project.tx)

        wwf.project_to_files(project=project, project_dir=project_dir_here, xml_glob=xml_glob, cfg_here=cfg_here)


    print(f'created {number} new random projects in {project_dir.name} with {n_tx} Tx in total')

if __name__ == "__main__":
    parser = ArgumentParser(description="Create indoor projects.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='configs/config1.yml', help='path to config')
    parser.add_argument('-n', '--number', type=int, default=1, help='Number of projects to be created (in total with the existing, up to ID number)')
    parser.add_argument('-x', '--xml_glob', type=str, default='x3d3_3_1*.xml', help='This glob will be used to find xml files in ./wi_templates, we use each template found with each Tx-Rx-combination')
    parser.add_argument('-d', '--project_dir', type=str, default='./projects', help='Where to save the projects')
    parser.add_argument('-ow', '--overwrite', action='store_true', help='Activate to overwrite existing projects, otherwise they will be skipped')
    args = parser.parse_args()
    main(
        config=args.config, 
        number=args.number, 
        xml_glob=args.xml_glob, 
        project_dir=Path(args.project_dir), 
        overwrite=args.overwrite)