
import numpy as np


def main(event_path):
    a1_list = [0.33,0.4534,0.2517]
    dataset_list = ['RomanW146sat1.dat','RomanZ087sat2.dat','RomanK213sat3.dat']
    dir6 = '/Users/jmbrashe/VBBOrbital/NEWGULLS/6_events_for_testing/levmar_runs_good_plx'
    event_name=event_path
    event_path = f'{dir6}/{event_path}'

    #res=pspl_fit(event_path,dataset_list,method='lm')
    #res = np.loadtxt(f'{event_path}/pspl_pars.txt',delimiter=',')
    #plot_pspl(res[0:3], dataset_list[0], event_path)
    #xrange = [8358,8362]
    #yrange = [21.2,21.7]

    plotname = 'plot1.png'
    #res = pd.read_csv(f'{dir6}/event_0_952_2841/FinalModels/LSfit005-5.txt', sep='\s+', header=None, nrows=1).values[0,
    #       0:7]
    #res[0:2] = np.log(res[0:2])
    #res[3]+=np.pi
    #res[4]=0.001299
    #res[4:6] = np.log(res[4:6])

    res = pd.read_csv(f'{dir6}/{event_name}/ICGS_initconds.txt', sep='\s+', header=None).values[0
     , :]
    #res = [0.699301951,3.194986579088e-06,0.723295004,(102.8134596+360)*np.pi/180,0.001443661,47.44172474,8352.655157]
    #res = [2.09760877,0.000282696,0.000612442,(193.999114)*np.pi/180,0.005084603,1.836156641,8234+325.7754641,0.035078942,0.05949086,0.0025594472218641493,0.0026140948867198607,0.002488074018596892]
    #res = [7.5001329231722380e-01,2.1604558753494731e-06,5.7088026497279953e-01,4.9128956213455304e+00,1.2985892234727756e-03,5.5179034293455409e+01,8.3532850815421825e+03]
    #res[0] = 1/res[0]
    xrange = [res[6] -0.1e1*res[5],res[6]+ 0.1e1*res[5]]
    #xrange = [8558,8561]
    #yrange=[18,21.3]
    res[0:2] = np.log(res[0:2])
    res[4:6] = np.log(res[4:6])
    #satellitedir='/Users/jmbrashe/VBBOrbital/NEWGULLS/6_events_for_testing/levmar_runs_good_plx/satellitedir'
    plot_2L1S(pars=res,dataset=dataset_list[0],event_path=event_path,xrange=xrange,yrange=None,evname=plotname)
    #print(res)
    #print(res.x)
    #print(res.cost*2)
    #print(res.fun)
if __name__ == "__main__":
    #dir6 = '/Users/jmbrashe/VBBOrbital/NEWGULLS/6_events_for_testing/levmar_runs_redo'
    events = ['event_0_128_2350','event_0_167_1273','event_0_672_793','event_0_715_873','event_0_874_19','event_0_952_2841']
    main('event_0_874_19')






