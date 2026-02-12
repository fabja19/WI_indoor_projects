from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import json
from joblib import delayed, Parallel
from typing import cast, Sequence
import h5py
from PIL import Image
import time

from modules.utils import W_dBm, dBm_W

### constants
N_0 = -174

def get_pl_min_max(
            simulation_dir : Path, 
            tx_power_simulations : float
        ) -> tuple[float,float]:
        """
        Calculates the minimum and maximum path loss values from a set of HDF5 files in a given directory.

        Each HDF5 file is expected to contain a dataset named 'received_power(dBm)'. The path loss is computed
        by subtracting the transmitter power (`tx_power_simulations`) from the received power values.

        Args:
            simulation_dir (Path): Path to the directory containing the HDF5 files.
            tx_power_simulations (float): The transmitter power used in the simulations (in dBm).

        Returns:
            tuple[float, float]: A tuple containing the minimum and maximum path loss values across all files.
        """
        pl_min, pl_max = 0, np.nan
        for idf, file in enumerate(simulation_dir.glob('*.h5')):
            with h5py.File(file, 'r') as f:
                path_loss = W_dBm(np.nansum(dBm_W(np.array(f['received_power(dBm)'], dtype=np.float64)), -1)) - tx_power_simulations
                path_loss[path_loss==-np.inf] = np.array(f['received_power(dBm)'])[path_loss==-np.inf,0]
            pl_min = np.nanmin([pl_min, np.nanmin(path_loss)])
            pl_max = np.nanmax([pl_max, np.nanmax(path_loss)])
        print(f'checked {idf} files to calculate min and max PL values: {pl_min=}, {pl_max=}')

        return pl_min, pl_max

def main(
        simulation_dir : Path,
        out_dir : Path,
        n_jobs : int,
        x_max : int,
        y_max : int,
        bandwidth : float,
        tx_power_simulations : float,
        tx_power_wanted : float,
        remove_from_file_name : list[str],
        no_swapping : bool,
        pl_max : float | None,
        pl_trnc : float | None,
    ) -> None:

    noise_floor = 10 * np.log10(bandwidth) + N_0
    PL_thr = -1 * tx_power_wanted + noise_floor

    if pl_max is None or pl_trnc is None:
        PL_min, PL_max = get_pl_min_max(simulation_dir=simulation_dir, tx_power_simulations=tx_power_simulations)
        PL_trnc = PL_thr - 1/4*(PL_max - PL_thr)
    if pl_max is not None:
        PL_max = pl_max
    if pl_trnc is not None:
        PL_trnc = pl_trnc
        PL_thr = None

    out_dir.mkdir(exist_ok=True)

    params_file = out_dir / 'rm_processing_parameters.json'
    params = {
            'x_max' : x_max,
            'y_max' : y_max,
            'bandwidth' : bandwidth,
            'tx_power_simulations' : tx_power_simulations,
            'tx_power_wanted' : tx_power_wanted,
            'PL_max' : PL_max,
            'PL_thr' : PL_thr,
            'PL_trnc' : PL_trnc
        }
    if params_file.exists():
        with open(params_file, 'r') as f:
            params_previous = json.load(f)
            assert params == params_previous, f'{params=}, but {params_previous=}\nTo process simulation output with different parameters, choose a different directory!'
    else:
        ### save parameters of the copy process
        with open(params_file, 'w') as f:
            json.dump(params, f)

    
    def process_simulation(p : Path) -> tuple[Path, str | None]:
        try:
            file_name = p.stem
            for r in remove_from_file_name:
                file_name = file_name.replace(r, '')
            ### remove leading 0 in Tx ID and make Tx ID consistent starting at 0
            file_name = f'{file_name.split('-')[0]}-{int(file_name.split('-')[1]) - 1}'
            file_name = out_dir / f'{file_name}.png'
            if file_name.exists():
                return (p, None)
            
            with h5py.File(p, 'r') as f:
                try:
                    power = np.array(f['received_power(dBm)'], dtype=np.float64)
                except:
                    try:
                        power = np.array(f['power'], dtype=np.float64)
                    except  Exception as e:
                        print(f'{f.keys()=}')
                        raise e
            if power.shape[0] > x_max or power.shape[1] > y_max:
                raise ValueError(f'{power.shape=} bigger that requested {(x_max, y_max)=}')
            PL = W_dBm(np.nansum(dBm_W(power), -1)) - tx_power_simulations
            PL[np.all(np.isnan(power), -1)] = PL_trnc
            if not no_swapping:
                PL = np.swapaxes(PL, 0, 1)
            ####
            # PL_big = np.full((x_max, y_max), PL_trnc, np.float64)
            # PL_big[:PL.shape[0], :PL.shape[1]] = PL
            # PL_img = Image.fromarray(((np.clip(PL_big, PL_trnc, PL_max) - PL_trnc)/ (PL_max - PL_trnc) * 255).astype(np.uint8))
            PL_img = Image.fromarray(((np.clip(PL, PL_trnc, PL_max) - PL_trnc)/ (PL_max - PL_trnc) * 255).astype(np.uint8))
            ####
            PL_img.save(file_name)
            return (p, None)  # count this project
        except Exception as e:
            return (p, str(e))
    
    results = cast(list[tuple], Parallel(n_jobs=n_jobs)(
        delayed(process_simulation)(p) for p in simulation_dir.glob('*.h5')
    ))

    simulations_processed = sum(1 for _, msg in results if msg is None)
    print(f'Created radio maps for {simulations_processed} simulations.')

    failed = [(p, msg) for p, msg in results if msg is not None]
    if failed:
        failure_file = out_dir / 'failures.txt'
        with open(failure_file, 'w') as f:
            print("Failures:")
            for p, msg in failed:
                print(f" - {p}: {msg}")
                f.write(f" - {p}: {msg}\n")
        print(f'saved failures to {failure_file}')
    
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Process simulation data to create radio maps.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--simulation_dir', type=str, default='./indoor_projects_filtered_250922/simulation_data', help='Directory containing simulation HDF5 files')
    parser.add_argument('-o', '--out_dir', type=str, default='./indoor_projects_filtered_250922/radio_maps_-12_-71', help='Output directory for radio map images')
    parser.add_argument('-x', '--x_max', type=int, default=32, help='Maximum x coordinate for image size')
    parser.add_argument('-y', '--y_max', type=int, default=32, help='Maximum y coordinate for image size')
    parser.add_argument('-bw', '--bandwidth', type=float, default=20e6, help='Bandwidth in Hz')
    parser.add_argument('-ts', '--tx_power_simulations', type=float, default=0.0, help='Transmitter power used in simulations (dBm)')
    parser.add_argument('-tw', '--tx_power_wanted', type=float, default=23.0, help='Desired transmitter power (dBm) to be modeled for the PL threshold')
    parser.add_argument('-rm', '--remove_from_file_name', type=str, nargs='*', default=['projects-', 'x3d3_3_1-t001_', '.r006'], help='Strings to remove from output file names to make them shorter')
    parser.add_argument('-nj', '--n_jobs', type=int, default=24, help='Number of parallel jobs for multiprocessing')
    parser.add_argument('-ns', '--no_swapping', action='store_true', help='Activate this flag to NOT swap axes. By default we do this, since WI messes them up.')
    parser.add_argument('-max', '--pl_max', type=float, default=None, help='Give an explicit PL max value for the grayscale conversion, so it will not be calculated as the max in the data.')
    parser.add_argument('-trnc', '--pl_trnc', type=float, default=None, help='Give an explicit PL trnc value for the grayscale conversion, so it will not be calculated from the badnwidth and the noise power.')
    args = parser.parse_args()
    
    main(
        simulation_dir=Path(args.simulation_dir),
        out_dir=Path(args.out_dir),
        n_jobs=args.n_jobs,
        x_max=args.x_max,
        y_max=args.y_max,
        bandwidth=args.bandwidth,
        tx_power_simulations=args.tx_power_simulations,
        tx_power_wanted=args.tx_power_wanted,
        remove_from_file_name=args.remove_from_file_name,
        no_swapping=args.no_swapping,
        pl_max=args.pl_max,
        pl_trnc=args.pl_trnc,
    )