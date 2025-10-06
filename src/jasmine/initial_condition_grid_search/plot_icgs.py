import glob

import VBMicrolensing
import numpy as np
import matplotlib.pyplot as plt
from gridfit import *
#[np.log(fsblpars[0]), np.log(fsblpars[1]), psplpars[0], fsblpars[2],
#                np.log(fsblpars[3] / psplpars[1]), np.log(psplpars[1]), psplpars[2]]
def make_comparison_plot(model_type,event_path,initconds_type,xrange,yrange):
    a1_list = [0.33,0.4534,0.2517] # limb darkening
    # load data
    dataset_list = ['RomanW146sat1.dat','RomanZ087sat2.dat','RomanK213sat3.dat']
    ndatasets = len(dataset_list)
    data_list = []
    for i in range(ndatasets):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
    ###

    event_name = event_path.split('/')[-1]
    VBMInstance = VBMicrolensing.VBMicrolensing()
    if initconds_type == 'icgs':
        icgs_init_conds = np.loadtxt(f'{event_path}/Data/ICGS_initconds.txt')#_chi2
        #print(icgs_init_conds)
        model_files_list = glob.glob(f'{event_path}/Models/*{model_type}*')
        models_list = []
        initcond_indices = []
        for file in model_files_list:
            file = file.split('/')[-1]
            models_list.append(file)
            index = int(file[5:8])
            initcond_indices.append(index)
        for i in range(len(models_list)):
            model_path = model_files_list[i]
            model_name = models_list[i]
            icgs_index = initcond_indices[i]
            init_cond = icgs_init_conds[icgs_index,:]
            pars = init_cond[0:7]

            log_pars = np.copy(pars)
            log_pars[0] = np.log(pars[0])
            log_pars[1] = np.log(pars[1])
            log_pars[4] = np.log(pars[4])
            log_pars[5] = np.log(pars[5])


            tmag = np.linspace(pars[-1] - 5 * pars[5], pars[-1] + 5 * pars[5], 80000)

            magnitude_list_init, flux_list_init, fluxdata_list_init, fluxerr_list_init, source_flux_list_init, blend_flux_list_init = calculate_magnifications_plots(log_pars, a1_list, data_list, tmag, ndatasets, VBMInstance, parallax=False)
            fig, ax = plt.subplots(dpi=100, layout='tight')

            ax.errorbar(data_list[0][:, -1], y=data_list[0][:, 0], yerr=data_list[0][:, 1], marker='.', markersize=0.75,
                        linestyle=' ', label='W146',color='springgreen')

            ax.plot(tmag, magnitude_list_init[0], zorder=10, linestyle ='--', label='Initial Condition', color='red')


            if xrange is not None:
                ax.set_xlim(xrange[0],xrange[1])
                ax.set_ylim(yrange[0],yrange[1])
            else:ax.set_xlim(tmag[0],tmag[-1])

            ax.yaxis.set_inverted(True)
            ax.set_xlabel('HJD - 2450000')
            ax.set_ylabel('W146')

            # read in RTModel solution
            rtm_pars = np.loadtxt(model_path, max_rows=1)
            log_rtm_pars = np.copy(rtm_pars)
            log_rtm_pars[0] = np.log(rtm_pars[0])
            log_rtm_pars[1] = np.log(rtm_pars[1])
            log_rtm_pars[4] = np.log(rtm_pars[4])
            log_rtm_pars[5] = np.log(rtm_pars[5])
            magnitude_list_rtm, flux_list_rtm, fluxdata_list_rtm, fluxerr_list_rtm, source_flux_list_rtm, blend_flux_list_rtm = calculate_magnifications_plots(log_rtm_pars[0:7], a1_list, data_list, tmag, ndatasets,
                                                                  VBMInstance, parallax=False)
            ax.plot(tmag, magnitude_list_rtm[0], zorder=10, linestyle='-', label='Levenberg-Marquardt', color='black')
            ax.legend()
            ax.set_title(f'{event_name} - {model_name}')
            plt.show()
    else:
        print('not implemented')
        return None # for now

if __name__ == "__main__":
    model_type_ = 'LS'
    event_path_='/Users/jmbrashe/Downloads/events/event_0_719_3030'
    initconds_type = 'icgs'
    make_comparison_plot(model_type=model_type_,event_path=event_path_,initconds_type=initconds_type,xrange=None,yrange=None)