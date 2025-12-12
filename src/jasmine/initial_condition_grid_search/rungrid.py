from sympy import false

from gridfit import run_event #fit_single_lens_models
import numpy as np
import glob
import pandas as pd
import warnings


warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

def main(event_name_, events_directory_path_, satellite_files_directory_path_, alpha_grid_density_):
    """Run a grid search for a given event."""
    a1_list = [0.33, 0.4534, 0.2517]
    dataset_list = ['RomanW146sat1.dat', 'RomanZ087sat2.dat', 'RomanK213sat3.dat']
    s = '0.37 0.40 0.43 0.46 0.49 0.52 0.56 0.60 0.65 0.70 0.75 0.80 0.84 0.88 0.91 0.93 0.95 0.965 0.98 0.99 1.00 1.01 1.02 1.035 1.05 1.07 1.09 1.12 1.15 1.18 1.22 1.26 1.30 1.34 1.38 1.42 1.46 1.50 1.55 1.60 1.66 1.72 1.79 1.86 1.94 2.02 2.1 2.2 2.3 2.4 2.5 2.62 2.75'
    s = s.split(' ')
    s_grid = np.array(list(map(float, s)))
    eps1 = '1.e-10 3.16e-7 1.e-6 3.16e-6 1.e-5 3.16e-5 1.e-4 3.16e-4 1.e-3 3.16e-3 1.e-2'
    eps1 = eps1.split(' ')
    eps1_grid = np.array(list(map(float, eps1)))
    q_grid = 1 / ((1 / eps1_grid) - 1)
    #q_grid = np.array([7.900280539951e-05])
    tstar = 0.000003
    alpha_grid = np.linspace(-3.15, 3.15, alpha_grid_density_)
    event_path = f'{events_directory_path_}/{event_name_}'
    model_types_ = ['PSPL_PLX']
    #fit_single_lens_models(event_path=event_path,nprocessors=8,satellitedir=satellite_files_directory_path_,modeltypes=model_types_)
    #pspl_fit_pyLIMA(event_path,dataset_list)
    run_event(event_path=event_path, dataset_list=dataset_list, grid_s=s_grid, grid_q=q_grid, grid_alpha=alpha_grid,
              tstar=tstar, a1_list=a1_list, pspl_thresh=0, satellitedir=satellite_files_directory_path_, processors=9,
              parallax=True,use_saved_pspl=False,finite_source=False)
if __name__ == "__main__":
    # dir6 = '/Users/jmbrashe/Downloads/events'
    # event = 'event_0_87_1723_memtest'

    events_directory_path = '/Users/jmbrashe/Downloads/sample_rtmodel_v2.4'
    #'/Users/jmbrashe/VBBOrbital/ICGS_analysis/benchmark_parallel'
    satellite_files_directory_path = '/Users/jmbrashe/Downloads/satellitedir'
    alpha_density = 945
    event_name = 'event_0_603_3135'

    main(event_name, events_directory_path, satellite_files_directory_path, alpha_density)

    # with multiprocessing.Pool(1) as p:
    #    p.map(main,events)

