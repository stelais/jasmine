import VBMicrolensing
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize,least_squares
import RTModel
import shutil
import os
import matplotlib.pyplot as plt
import logging
import sys

def minimize_linear_pars(y,err,x):
    """
    Exact solution to minimize linear eq chi2
    """
    alpha1 = np.sum(x/err**2)
    alpha2 = np.sum(1/err**2)
    alpha3 = np.sum(y/err**2)
    alpha4 = np.sum((x/err)**2)
    alpha5 = np.sum(x*y/err**2)
    a = (alpha5*alpha2 - alpha1*alpha3)/(alpha4*alpha2-alpha1**2)
    b = (alpha4*alpha3 - alpha5*alpha1)/(alpha4*alpha2-alpha1**2)
    return a,b

def get_chi2(magnitude_list,mag_data_list,error_list,ndatasets):
    """calculate chi2 for model on each dataset provided"""
    chi2_list = []
    for i in range(ndatasets):
        res = (magnitude_list[i] - mag_data_list[i]) / error_list[i]
        chi2 = np.sum(np.power(res,2))
        chi2_list.append(chi2)
    chi2_sum = np.sum(chi2_list)
    return chi2_list,chi2_sum

def get_residuals(magnitude_list,mag_data_list,error_list,ndatasets):
    """calculate chi2 for model on each dataset provided"""
    residue_list = []
    for i in range(ndatasets):
        res = (magnitude_list[i] - mag_data_list[i]) / error_list[i]
        residue_list.append(res)
    residues = np.concatenate(residue_list)
    return residues

def calculate_magnifications(pars,a1_list,data_list,ndatasets,VBMInstance):
    """Calculate LC magnitudes in each passband provided for 2L1S"""
    magnitude_list = []
    source_flux_list = []
    blend_flux_list = []
    for i in range(ndatasets):
        VBMInstance.a1 = a1_list[i]
        data = data_list[i]
        magnifications, y1, y2 = VBMInstance.BinaryLightCurve(pars,data[:,-1])
        meas_flux = 10**(-0.4*data[:,0])
        meas_flux_err = 0.4*np.log(10)*10**(-0.4*data[:,0])*data[:,1]
        #Fix so minimize_linear_parameters uses flux_err not mag_err
        source_flux,blend_flux = minimize_linear_pars(meas_flux,meas_flux_err,magnifications)
        sim_flux = np.array(magnifications)*source_flux+blend_flux
        source_flux_list.append(source_flux)
        blend_flux_list.append(blend_flux)
        sim_magnitudes = -2.5*np.log10(sim_flux)
        magnitude_list.append(sim_magnitudes)
    return magnitude_list,source_flux_list,blend_flux_list

def calculate_magnifications_pspl(pars,data_list,data_mag,ndatasets,VBMInstance):
    """Calculate LC magnitudes in each passband provided for PSPL model"""
    magnitude_list = []
    source_flux_list = []
    blend_flux_list = []
    pars_log = [np.log(pars[0]), np.log(pars[1]), pars[2]]
    for i in range(ndatasets):
        data = data_list[i]
        magnifications, y1, y2 = VBMInstance.PSPLLightCurve(pars_log,data[:,-1])
        meas_flux = 10**(-0.4*data[:,0])
        meas_flux_err = 0.4*np.log(10)*10**(-0.4*data[:,0])*data[:,1]
        source_flux,blend_flux = minimize_linear_pars(meas_flux,meas_flux_err,magnifications)
        magnifications, y1, y2 = VBMInstance.PSPLLightCurve(pars_log,data_mag)
        sim_flux = np.array(magnifications)*source_flux+blend_flux
        source_flux_list.append(source_flux)
        blend_flux_list.append(blend_flux)
        sim_magnitudes = -2.5*np.log10(sim_flux)
        magnitude_list.append(sim_magnitudes)
    return magnitude_list,source_flux_list,blend_flux_list

def evaluate_model(psplpars, fsblpars, a1_list, data_list, VBMInstance, pspl_chi2):
    """ Calculate chi2 for one model in the grid
        fsblpars = [s,q,alpha,tstar]
        psplpars = follows VBMicrolensing format
        data_list list with a dataframe for each band in RTModel data format
        ndatasets = number of datasets
    """
    #blend_flux_list = []
    #source_flux_list = []
    ndatasets = len(data_list)
    pars = [np.log(fsblpars[0]), np.log(fsblpars[1]), psplpars[0], fsblpars[2], np.log(fsblpars[3] / psplpars[1]), math.log(psplpars[1]), psplpars[2]]
    magnitude_list,source_flux_list,blend_flux_list = calculate_magnifications(pars,a1_list,data_list,ndatasets,VBMInstance)
    magdata_list = []
    magerr_list = []
    for i in range(ndatasets):
        magdata_list.append(data_list[i][:,0])
        magerr_list.append(data_list[i][:,1])
    chi2_list,chi2_sum = get_chi2(magnitude_list, magdata_list, magerr_list, ndatasets)
    return_list = np.concatenate((pars, source_flux_list,blend_flux_list, chi2_list,[chi2_sum, chi2_sum-pspl_chi2]))
    return return_list

def grid_fit(event_path, dataset_list, pspl_pars, grid_s, grid_q, grid_alpha, tstar, a1_list, pspl_chi2,logger):
    """
        Fit grid of models to determine initial conditions
    """
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
    VBMInstance = VBMicrolensing.VBMicrolensing()
    VBMInstance.RelTol = 1e-03
    VBMInstance.Tol=1e-03
    s,q,alpha = np.meshgrid(grid_s,grid_q,grid_alpha)
    s= s.flatten()
    q = q.flatten()
    alpha = alpha.flatten()
    #print(s.shape[0])
    grid_results = np.zeros(shape=(s.shape[0],9+3*len(a1_list)))
    for i in range(s.shape[0]):
        fsblpars = [s[i],q[i],alpha[i],tstar]
        output = evaluate_model(pspl_pars, fsblpars, a1_list, data_list, VBMInstance, pspl_chi2)
        grid_results[i,:] = output
        if i%5000==0: logger.info(i)
    return grid_results

def pspl_fit(event_path,dataset_list,logger,p0=None,init_ind = 0,method='lm'):
    """ Find best fitting PSPL Model"""
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
    VBMInstance = VBMicrolensing.VBMicrolensing()
    VBMInstance.RelTol = 1e-03
    VBMInstance.Tol=1e-03
    if p0 is None:
        logger.info('Setting default initial conditions')
        t0_init = data_list[init_ind][np.argmin(data_list[init_ind][:,0]),-1]
        p0 = [0.1,10,t0_init]
    else:
        logger.info('Using provided initial conditions')
    #magnitude_list = []
    #source_flux_list = []
    #blend_flux_list = []
    if method =='lm':
        logger.info('Using Levenberg-Marquardt')
        res = least_squares(fun=calc_pspl_residuals, x0=p0, args=[[data_list,len(data_list),VBMInstance,logger]],method='lm',ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=50000)
    else:
        logger.info('Using some scipy.minimize')
        res = minimize(calc_pspl_chi2, p0,[data_list,len(data_list),VBMInstance,logger],method=method)
    logger.info('Done Fitting PSPL!')
    return res

def calc_pspl_chi2(pars,args):
    """ Objective function to optimize for pspl_fit() """
    magnitude_list = []
    source_flux_list = []
    blend_flux_list = []
    data_list = args[0]
    ndatasets = args[1]
    VBMInstance = args[2]
    logger = args[3]
    pars_log = [np.log(pars[0]),np.log(pars[1]),pars[2]]
    #logger.info(pars)
    for i in range(ndatasets):
        data = data_list[i]
        magnifications,y1,y2 = VBMInstance.PSPLLightCurve(pars_log,data[:,-1])
        meas_flux = 10**(-0.4*data[:,0])
        meas_flux_err = 0.4*np.log(10)*10**(-0.4*data[:,0])*data[:,1]
        source_flux,blend_flux = minimize_linear_pars(meas_flux,meas_flux_err,magnifications)
        sim_flux = np.array(magnifications)*source_flux+blend_flux
        source_flux_list.append(source_flux)
        blend_flux_list.append(blend_flux)
        sim_magnitudes = -2.5*np.log10(sim_flux)
        magnitude_list.append(sim_magnitudes)
    magdata_list = []
    magerr_list = []
    for i in range(ndatasets):
        magdata_list.append(data_list[i][:,0])
        magerr_list.append(data_list[i][:,1])
    chi2_list, chi2 = get_chi2(magnitude_list, magdata_list, magerr_list, ndatasets)
    logger.info(chi2)
    return chi2

def calc_pspl_residuals(pars,args):
    """ Objective function to optimize for pspl_fit() """
    magnitude_list = []
    source_flux_list = []
    blend_flux_list = []
    data_list = args[0]
    ndatasets = args[1]
    VBMInstance = args[2]
    logger = args[3]
    pars_log = [np.log(pars[0]),np.log(pars[1]),pars[2]]
    #print(pars)
    for i in range(ndatasets):
        data = data_list[i]
        magnifications,y1,y2 = VBMInstance.PSPLLightCurve(pars_log,data[:,-1])
        meas_flux = 10**(-0.4*data[:,0])
        meas_flux_err = 0.4*np.log(10)*10**(-0.4*data[:,0])*data[:,1]
        source_flux,blend_flux = minimize_linear_pars(meas_flux,meas_flux_err,magnifications)
        sim_flux = np.array(magnifications)*source_flux+blend_flux
        source_flux_list.append(source_flux)
        blend_flux_list.append(blend_flux)
        sim_magnitudes = -2.5*np.log10(sim_flux)
        magnitude_list.append(sim_magnitudes)
    magdata_list = []
    magerr_list = []
    for i in range(ndatasets):
        magdata_list.append(data_list[i][:,0])
        magerr_list.append(data_list[i][:,1])
    residuals = get_residuals(magnitude_list, magdata_list, magerr_list, ndatasets)
    chi2 = np.sum(np.power(residuals,2))
    logger.info(chi2)
    return residuals


def filter_by_q(grid_result, logger,pspl_thresh=0):
    """
    Finds best initial conditions on grid for each value of q.
    Does a very simple check for s>1 s<1, but this often will not find an s/1/s degeneracy.
    """
    logger.info('Picking best models for each q.')
    q_list = list(grid_result['log(q)'].unique())
    best_grid_models = []
    for q in q_list:
        #print(q)
        subgrid = grid_result[grid_result['log(q)'] == q]  # filter results for 1 q value.
        best_model = subgrid.sort_values('delta_pspl_chi2').reset_index(drop=True).iloc[0,
                     :]  # get best grid model for this q
        if best_model['delta_pspl_chi2'] <= pspl_thresh:
            best_grid_models.append(best_model)  # add to list of models to be run
        # Simple check for any s<->1/s degeneracy. Just looks any good solution with s> or < 1
       #
         # s = np.exp(best_model['log(s)'])
        # if s > 1:
        #     subgrid_s_degen = subgrid[np.exp(subgrid['log(s)']) <= 1]  # filter out s>=1 to get s<1 df
        #     best_model_s_degen = subgrid_s_degen.sort_values('delta_pspl_chi2').reset_index(drop=True).iloc[0,
        #                          :]  # check for s<1
        #     if best_model_s_degen['delta_pspl_chi2'] <= pspl_thresh:
        #         best_grid_models.append(best_model_s_degen)  # add to list of models to be run
        # elif s < 1:
        #     subgrid_s_degen = subgrid[np.exp(subgrid['log(s)']) >= 1]  # filter out s<=1 to get s>1 df
        #     best_model_s_degen = subgrid_s_degen.sort_values('delta_pspl_chi2').reset_index(drop=True).iloc[0,
        #                          :]  # check for s>1
        #     if best_model_s_degen['delta_pspl_chi2'] <= pspl_thresh:
        #         best_grid_models.append(best_model_s_degen)  # add to list of models to be run
        # elif s == float(1):
        #     check for both s>1 s<1
            # subgrid_s_degen = subgrid[np.exp(subgrid['log(s)']) <= 1]  # filter out s>=1 to get s<1 df
            # best_model_s_degen = subgrid_s_degen.sort_values('delta_pspl_chi2').reset_index(drop=True).iloc[0,
            #                      :]  # check for s<1
            # if best_model_s_degen['delta_pspl_chi2'] <= pspl_thresh:
            #     best_grid_models.append(best_model_s_degen)  # add to list of models to be run
            # subgrid_s_degen = subgrid[np.exp(subgrid['log(s)']) >= 1]  # filter out s<=1 to get s>1 df
            # best_model_s_degen = subgrid_s_degen.sort_values('delta_pspl_chi2').reset_index(drop=True).iloc[0,
            #                      :]  # check for s>1
            # if best_model_s_degen['delta_pspl_chi2'] <= pspl_thresh:
            #     best_grid_models.append(best_model_s_degen)  # add to list of models to be run

    best_grid_models = pd.concat(best_grid_models, axis=1,
                                 ignore_index=True).T  # create a DataFrame of the best grid models
    # convert to initial condition input format used by RTModel
    init_cond = best_grid_models.iloc[:, 0:7].values.copy()
    init_cond[:, 0] = np.exp(init_cond[:, 0])
    init_cond[:, 1] = np.exp(init_cond[:, 1])
    init_cond[:, 4] = np.exp(init_cond[:, 4])
    init_cond[:, 5] = np.exp(init_cond[:, 5])
    logger.info('Done!')
    return best_grid_models, init_cond

def run_event(event_path,dataset_list,grid_s,grid_q,grid_alpha,tstar,a1_list,pspl_thresh,processors,method='lm'):
    """ Wrapper Function to go from pspl_fit to final RTModel runs."""
    #Remove old log file if it exists. Mostly for quick troubleshoots.
    try:
        os.remove(path=f'{event_path}/ICGS.log')
    except FileNotFoundError: print('No log file exists. Continuing.')
    #Create logging object
    logger = logging.getLogger()
    logging.basicConfig(filename=f'{event_path}/ICGS.log',level=logging.INFO)
    #Send errors and stdout to logger.
    sys.stderr.write = logger.error
    sys.stdout.write = logger.info
    print('Printing to logger!')
    # First do the PSPL fit
    pspl_results = pspl_fit(event_path=event_path,dataset_list=dataset_list,method=method,logger=logger)
    if method == 'lm':
        pspl_chi2 = pspl_results.cost*2
    else: pspl_chi2 = pspl_results.chi2
    pspl_pars = pspl_results.x
    #save pspl fit to a txt file
    grid_fit_results = grid_fit(event_path=event_path, dataset_list=dataset_list, pspl_pars=pspl_pars,
                                grid_s=grid_s,grid_q=grid_q,grid_alpha=grid_alpha,tstar=tstar,a1_list=a1_list,pspl_chi2=pspl_chi2,logger=logger)
    names =  ['log(s)','log(q)','u0','alpha','log(rho)','log(tE)','t0','fs0','fb0','fs1','fb1','fs2','fb2','chi20','chi21','chi22','chi2sum','delta_pspl_chi2']
    grid_result_df = pd.DataFrame(grid_fit_results,columns=names)
    filtered_df,init_conds = filter_by_q(grid_result=grid_result_df,logger=logger,pspl_thresh=pspl_thresh)
    #Now run these in RTModel
    # Have RTModel prints go to log not stdout
    rtm = RTModel.RTModel()
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    rtm.recover_options() # Use same options as Stela on NCCS
    rtm.archive_run()
    shutil.copyfile(src = f'{event_path}/run-0001/ICGS.log',dst=f'{event_path}/ICGS.log')
    shutil.rmtree(f'{event_path}/run-0001' ,) # have to remove old stuff or it affects the InitConds for everything.
    #Write some outputs after clearing the directory
    with open(f'{event_path}/pspl_pars.txt','w') as f:
        f.write(f'{pspl_pars[0]},{pspl_pars[1]},{pspl_pars[2]},{pspl_chi2}')
    np.savetxt(fname=f'{event_path}/ICGS_initconds.txt', X=init_conds)  # save init conds to a text file
    np.savetxt(f'{event_path}/grid_fit.txt', grid_fit_results)
    rtm.Reader()
    rtm.InitCond()
    num_init_cond = init_conds.shape[0]
    for n in range(num_init_cond):
        init_cond = list(init_conds[n,:])
        #launch each fit from the init conds
        rtm.LevMar(f'LSfit{n:03}',parameters = init_cond)
    rtm.ModelSelector('LS')
    rtm.launch_fits('LX')
    rtm.ModelSelector('LX')
    rtm.launch_fits('LO')
    rtm.ModelSelector('LO')
    rtm.Finalizer()
    return None

def plot_pspl(pars,dataset,event_path):
    data = np.loadtxt(f'{event_path}/Data/{dataset}')
    VBMInstance = VBMicrolensing.VBMicrolensing()
    VBMInstance.RelTol = 1e-03
    VBMInstance.Tol = 1e-03
    tmag = np.linspace(data[0,-1],data[-1,-1],80000)
    magnitude_list, source_flux_list, blend_flux_list = calculate_magnifications_pspl(pars,[data],tmag,1,VBMInstance)
    magnitude = magnitude_list[0]
    fig, ax = plt.subplots(dpi=100, layout='tight')
    ax.plot(tmag, magnitude, zorder=10, label='PSPL Fit', color='black')
    #magnitude_list, source_flux_list, blend_flux_list = calculate_magnifications_pspl([0.6291636,50.20121621,pars[-1]], [data],tmag, 1, VBMInstance)
    #magnitude = magnitude_list[0]
    #ax.plot(tmag, magnitude, zorder=10, label='PSPL True', color='red')
    ax.errorbar(data[:, -1], y=data[:, 0], yerr=data[:, 1], marker='.', markersize=0.75,
                linestyle=' ', label='W146')
    ax.set_xlim(pars[-1]-1*pars[1],pars[-1]+1*pars[1])
    ax.yaxis.set_inverted(True)
    ax.legend()
    plt.savefig(f'{event_path}/pspl_plot.png')
    plt.show()

    return None