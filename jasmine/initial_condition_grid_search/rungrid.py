#from gridfit import *
#import sys
#import tdqm
import numpy as np
import multiprocessing
from jasmine.initial_condition_grid_search.gridfit import *
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
    dir6 = '/Users/jmbrashe/VBBOrbital/NEWGULLS/6_events_for_testing/levmar_runs_good_plx'
    event_name=event_path
    event_path = f'{dir6}/{event_path}'
    run_event(event_path=event_path, dataset_list=dataset_list, grid_s=s_grid, grid_q=q_grid, grid_alpha=alpha_grid, tstar=tstar,
              a1_list=a1_list, pspl_thresh=0, processors=5)
    #res=pspl_fit(event_path,dataset_list,method='lm')
    #res = np.loadtxt(f'{event_path}/pspl_pars.txt',delimiter=',')
    #plot_pspl(res[0:3], dataset_list[0], event_path)
    #xrange = [8358,8362]
    #yrange = [21.2,21.7]

    #plotname = 'ploticgsbest.png'
    #res = pd.read_csv(f'{dir6}/event_0_952_2841/FinalModels/LSfit005-5.txt', sep='\s+', header=None, nrows=1).values[0,
    #       0:7]
    #res[0:2] = np.log(res[0:2])
    #res[3]+=np.pi
    #res[4]=0.001299
    #res[4:6] = np.log(res[4:6])

    #res = pd.read_csv(f'{dir6}/{event_name}/ICGS_initconds.txt', sep='\s+', header=None).values[5
    # , :]
    #res = [0.699301951,3.194986579088e-06,0.723295004,(102.8134596-180)*np.pi/180,0.001443661,47.44172474,8352.655157]
    #res = [7.5001329231722380e-01,2.1604558753494731e-06,5.7088026497279953e-01,4.9128956213455304e+00,1.2985892234727756e-03,5.5179034293455409e+01,8.3532850815421825e+03]
    #res[0] = 1/res[0]
    #xrange = [res[-1] -0.1*res[5],res[-1]+ 0.1*res[5]]
    #yrange=[21.2,21.7]
    #res[0:2] = np.log(res[0:2])
    #res[4:6] = np.log(res[4:6])

    #plot_2L1S(res,dataset_list[0],event_path,xrange,yrange,plotname)
    #print(res)
    #print(res.x)
    #print(res.cost*2)
    #print(res.fun)
if __name__ == "__main__":
    #dir6 = '/Users/jmbrashe/VBBOrbital/NEWGULLS/6_events_for_testing/levmar_runs_redo'
    events = ['event_0_128_2350','event_0_167_1273','event_0_672_793','event_0_715_873','event_0_874_19','event_0_952_2841']
    with multiprocessing.Pool(6) as p:
        p.map(main,events)
    #main(events[-1])






