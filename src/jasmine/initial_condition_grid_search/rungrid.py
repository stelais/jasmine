#from gridfit import *
#import sys
#import tdqm
import numpy as np


def main(event_path):
    a1_list = [0.33,0.4534,0.2517]
    dataset_list = ['RomanW146sat1.dat','RomanZ087sat2.dat','RomanK213sat3.dat']
    s = '0.37 0.40 0.43 0.46 0.49 0.52 0.56 0.60 0.65 0.70 0.75 0.80 0.84 0.88 0.91 0.93 0.95 0.965 0.98 0.99 1.00 1.01 1.02 1.035 1.05 1.07 1.09 1.12 1.15 1.18 1.22 1.26 1.30 1.34 1.38 1.42 1.46 1.50 1.55 1.60 1.66 1.72 1.79 1.86 1.94 2.02 2.1 2.2 2.3 2.4 2.5 2.62 2.75'
    s = s.split(' ')
    s_grid = list(map(float,s))
    eps1 = '1.e-6 3.16e-6 1.e-5 3.16e-5 1.e-4 3.16e-4 1.e-3 3.16e-3 1.e-2 3.16e-2 1.e-1 3.16e-1'
    eps1 = eps1.split(' ')
    eps1_grid = np.array(list(map(float,eps1)))
    q_grid = 1/((1/eps1_grid)-1)
    tstar=0.03
    alpha_grid = np.linspace(-3.15,3.15,316)
    dir6 = '/Users/jmbrashe/VBBOrbital/NEWGULLS/6_events_for_testing'
    event_name=event_path
    event_path = f'{dir6}/{event_path}'
    satdir = '/Users/jmbrashe/VBBOrbital/NEWGULLS/6_events_for_testing/levmar_runs_good_plx/satellitedir'
    run_event(event_path=event_path, dataset_list=dataset_list, grid_s=s_grid, grid_q=q_grid, grid_alpha=alpha_grid, tstar=tstar,
              a1_list=a1_list, pspl_thresh=0, satellitedir=satdir,processors=1)
if __name__ == "__main__":
    #dir6 = '/Users/jmbrashe/VBBOrbital/NEWGULLS/6_events_for_testing/levmar_runs_redo'
    event = 'event_0_5_606'
    #with multiprocessing.Pool(1) as p:
    #    p.map(main,events)
    main(event)






