import VBMicrolensing
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize,least_squares
import RTModel
import shutil
import os
import matplotlib.pyplot as plt
import time
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

def grid_fit(event_path, dataset_list, pspl_pars, grid_s, grid_q, grid_alpha, tstar, a1_list, pspl_chi2):
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
    print(f'PSPL pars: {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]}')
    print(f'Checking {s.shape[0]} Models on grid!')
    for i in range(s.shape[0]):
        fsblpars = [s[i],q[i],alpha[i],tstar]
        print(f'{i} {s[i]} {q[i]} {alpha[i]} {tstar}')
        output = evaluate_model(pspl_pars, fsblpars, a1_list, data_list, VBMInstance, pspl_chi2)
        grid_results[i,:] = output
        #if i%5000==0: print(f'{i} Models checked')
    print('Done checking models!')
    return grid_results

def pspl_fit(event_path,dataset_list,p0=None,init_ind = 0,method='lm'):
    """ Find best fitting PSPL Model"""
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
    VBMInstance = VBMicrolensing.VBMicrolensing()
    VBMInstance.RelTol = 1e-03
    VBMInstance.Tol=1e-03
    if p0 is None:
        print('Setting default initial conditions')
        t0_init = data_list[init_ind][np.argmin(data_list[init_ind][:,0]),-1]
        p0 = [0.1,10,t0_init]
    else:
        print('Using provided initial conditions')
    #magnitude_list = []
    #source_flux_list = []
    #blend_flux_list = []
    if method =='lm':
        print('Using Levenberg-Marquardt')
        res = least_squares(fun=calc_pspl_residuals, x0=p0, args=[[data_list,len(data_list),VBMInstance]],method='lm',ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=50000)
    else:
        print('Using some scipy.minimize')
        res = minimize(calc_pspl_chi2, p0,[data_list,len(data_list),VBMInstance],method=method)
    print('Done Fitting PSPL!')
    return res

def calc_pspl_chi2(pars,args):
    """ Objective function to optimize for pspl_fit() """
    magnitude_list = []
    source_flux_list = []
    blend_flux_list = []
    data_list = args[0]
    ndatasets = args[1]
    VBMInstance = args[2]
    #logger = args[3]
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
    print(chi2)
    return chi2

def calc_pspl_residuals(pars,args):
    """ Objective function to optimize for pspl_fit() """
    magnitude_list = []
    source_flux_list = []
    blend_flux_list = []
    data_list = args[0]
    ndatasets = args[1]
    VBMInstance = args[2]
    #logger = args[3]
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
    print(f'chi2: {chi2}')
    return residuals


def filter_by_q(grid_result,pspl_thresh=0):
    """
    Finds best initial conditions on grid for each value of q.
    Does a very simple check for s>1 s<1, but this often will not find an s/1/s degeneracy.
    """
    print('Picking best models for each mass ratio')
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
    print('Done')
    return best_grid_models, init_cond

def run_event(event_path,dataset_list,grid_s,grid_q,grid_alpha,tstar,a1_list,pspl_thresh,processors,satellitedir,method='lm'):
    """ Wrapper Function to go from pspl_fit to final RTModel runs."""
    #Remove old log file if it exists. Mostly for quick troubleshoots.
    #try:
    #    os.remove(path=f'{event_path}/Data/ICGS.log')
    #except FileNotFoundError: print('No log file exists. Continuing.')
    #Create logging object
    #logger = logging.getLogger()
    #logging.basicConfig(filename=f'{event_path}/Data/ICGS.log',level=logging.INFO)
    #Send errors and stdout to logger.
    #sys.stderr.write = logger.error
    #sys.stdout.write = logger.info
    #print('Printing to logger!')
    # First do the PSPL fit
    pspl_results = pspl_fit(event_path=event_path,dataset_list=dataset_list,method=method)
    if method == 'lm':
        pspl_chi2 = pspl_results.cost*2
    else: pspl_chi2 = pspl_results.chi2
    pspl_pars = pspl_results.x
    #save pspl fit to a txt file
    time0 = time.time()
    grid_fit_results = grid_fit(event_path=event_path, dataset_list=dataset_list, pspl_pars=pspl_pars,
                                grid_s=grid_s,grid_q=grid_q,grid_alpha=grid_alpha,tstar=tstar,a1_list=a1_list,pspl_chi2=pspl_chi2)
    time1 = time.time()
    print(f'ICGS time: {time1-time0}')
    time0 = time.time()
    names =  ['log(s)','log(q)','u0','alpha','log(rho)','log(tE)','t0','fs0','fb0','fs1','fb1','fs2','fb2','chi20','chi21','chi22','chi2sum','delta_pspl_chi2']
    grid_result_df = pd.DataFrame(grid_fit_results,columns=names)
    filtered_df,init_conds = filter_by_q(grid_result=grid_result_df,pspl_thresh=pspl_thresh)
    #Now run these in RTModel
    # Have RTModel prints go to log not stdout
    rtm = RTModel.RTModel()
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    rtm.archive_run()
    shutil.rmtree(f'{event_path}/run-0001') # have to remove old stuff or it affects the InitConds for everything.
    #Write some outputs after clearing the directory
    with open(f'{event_path}/Data/pspl_pars.txt','w') as f:
        f.write(f'{pspl_pars[0]},{pspl_pars[1]},{pspl_pars[2]},{pspl_chi2}')
    np.savetxt(fname=f'{event_path}/Data/ICGS_initconds.txt', X=init_conds)  # save init conds to a text file
    np.savetxt(f'{event_path}/Data/grid_fit.txt', grid_fit_results)
    modeltypes = ['PS','LS','LX','LO']
    rtm.set_satellite_dir(satellitedir=satellitedir)
    peak_threshold = 5
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    rtm.set_satellite_dir(satellitedir=satellitedir)
    rtm.config_Reader(otherseasons=0, binning=1000000)
    rtm.config_InitCond(usesatellite=1, peakthreshold=peak_threshold,modelcategories=modeltypes)
    rtm.Reader()
    rtm.InitCond()
    #Do FSPL fit for comparison
    print('Launching PS Fits')
    rtm.launch_fits('PS')
    rtm.ModelSelector('PS')
    print('Launching LS Fits')
    num_init_cond = init_conds.shape[0]
    for n in range(num_init_cond):
        init_cond = list(init_conds[n,:])
        #launch each fit from the init conds
        rtm.LevMar(f'LSfit{n:03}',parameters = init_cond)
    rtm.ModelSelector('LS')
    print('Launching LX and LO fits')
    rtm.launch_fits('LX')
    rtm.ModelSelector('LX')
    rtm.launch_fits('LO')
    rtm.ModelSelector('LO')
    rtm.Finalizer()
    print('Done')
    time1 = time.time()
    print(f'RTModel time: {time1 - time0}')
    return None

#def run_event_from_file(event_path,pspl_thresh,processors,satellitedir,method='lm'):
#    """ Wrapper Function to go from pspl_fit to final RTModel runs."""
    #Create logging object
#    logger = logging.getLogger()
#     logging.basicConfig(filename=f'{event_path}/Data/ICGSfits.log',level=logging.INFO)
#     #Send errors and stdout to logger.
#     sys.stderr.write = logger.error
#     sys.stdout.write = logger.info
#     print('Printing to logger!')
#     #Now run these in RTModel
#     # Have RTModel prints go to log not stdout
#     init_conds = pd.read_csv(f'{event_path}/Data/I)
#
#
#     rtm = RTModel.RTModel()
#     rtm.set_processors(nprocessors=processors)
#     rtm.set_event(event_path)
#     rtm.archive_run()
#     shutil.rmtree(f'{event_path}/run-0001') # have to remove old stuff or it affects the InitConds for everything.
#     #Write some outputs after clearing the directory
#     modeltypes = ['PS','LS','LX','LO']
#     rtm.set_satellite_dir(satellitedir=satellitedir)
#     peak_threshold = 5
#     rtm.set_processors(nprocessors=processors)
#     rtm.set_event(event_path)
#     rtm.set_satellite_dir(satellitedir=satellitedir)
#     rtm.config_Reader(otherseasons=0, binning=1000000)
#     rtm.config_InitCond(usesatellite=1, peakthreshold=peak_threshold,modelcategories=modeltypes)
#     rtm.Reader()
#     rtm.InitCond()
#     #Do FSPL fit for comparison
#     logger.info('Launching PS Fits')
#     rtm.launch_fits('PS')
#     rtm.ModelSelector('PS')
#     logger.info('Launching LS Fits')
#     num_init_cond = init_conds.shape[0]
#     for n in range(num_init_cond):
#         init_cond = list(init_conds[n,:])
#         #launch each fit from the init conds
#         rtm.LevMar(f'LSfit{n:03}',parameters = init_cond)
#     rtm.ModelSelector('LS')
#     logger.info('Launching LX and LO fits')
#     rtm.launch_fits('LX')
#     rtm.ModelSelector('LX')
#     rtm.launch_fits('LO')
#     rtm.ModelSelector('LO')
#     rtm.Finalizer()
#     logger.info('Done')
#     return None

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




def calculate_magnifications_plots(pars,a1_list,data_list,data_mag,ndatasets,VBMInstance,parallax=False):
    """Calculate LC magnitudes in each passband provided for 2L1S"""
    magnitude_list = []
    source_flux_list = []
    blend_flux_list = []
    for i in range(ndatasets):
        VBMInstance.a1 = a1_list[i]
        data = data_list[i]
        if parallax == False:
            magnifications, y1, y2 = VBMInstance.BinaryLightCurve(pars,data[:,-1])
        else:
            magnifications, y1, y2, sorb = VBMInstance.BinaryLightCurveOrbital(pars, data[:, -1])
        meas_flux = 10**(-0.4*data[:,0])
        meas_flux_err = 0.4*np.log(10)*10**(-0.4*data[:,0])*data[:,1]
        #Fix so minimize_linear_parameters uses flux_err not mag_err
        source_flux,blend_flux = minimize_linear_pars(meas_flux,meas_flux_err,magnifications)
        magnifications, y1, y2 = VBMInstance.BinaryLightCurve(pars, data_mag)
        sim_flux = np.array(magnifications)*source_flux+blend_flux
        source_flux_list.append(source_flux)
        blend_flux_list.append(blend_flux)
        sim_magnitudes = -2.5*np.log10(sim_flux)
        magnitude_list.append(sim_magnitudes)
    return magnitude_list,source_flux_list,blend_flux_list

def plot_2L1S(pars,dataset,event_path,xrange,yrange,evname):
    data = np.loadtxt(f'{event_path}/Data/{dataset}')
    VBMInstance = VBMicrolensing.VBMicrolensing()
    VBMInstance.RelTol = 1e-04
    VBMInstance.Tol = 1e-05
    tmag = np.linspace(pars[-1]-5*pars[5],pars[-1]+5*pars[5],80000)
    a1 = 0.33
    magnitude_list, source_flux_list, blend_flux_list = calculate_magnifications_plots(pars,[a1],[data],tmag,1,VBMInstance)
    magnitude = magnitude_list[0]
    fig, ax = plt.subplots(dpi=100, layout='tight')
    ax.plot(tmag, magnitude, zorder=10, label='2L1S Fit', color='black')
    #magnitude_list, source_flux_list, blend_flux_list = calculate_magnifications_pspl([0.6291636,50.20121621,pars[-1]], [data],tmag, 1, VBMInstance)
    #magnitude = magnitude_list[0]
    #ax.plot(tmag, magnitude, zorder=10, label='PSPL True', color='red')
    ax.errorbar(data[:, -1], y=data[:, 0], yerr=data[:, 1], marker='.', markersize=0.75,
                linestyle=' ', label='W146')
    if yrange == None:
        ind_0 = np.where(tmag >= xrange[0])
        ind_1 = np.where(tmag <= xrange[1])

        ind_data_0 = np.where(data[:,-1] >= xrange[0])
        ind_data_1 = np.where(data[:, -1] <= xrange[1])
        ind_plot = np.intersect1d(ind_0,ind_1)
        ind_data = np.intersect1d(ind_data_0,ind_data_1)

        ymin = np.min([np.min(magnitude[ind_plot]) - 0.1,np.min(data[ind_data,0]) - 0.1])
        ymax = np.max([np.max(magnitude[ind_plot]) + 0.1,np.max(data[ind_data,0]) + 0.1])
        yrange=[ymin,ymax]
    ax.set_xlim(xrange[0],xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    ax.yaxis.set_inverted(True)
    ax.set_ylabel('W146')
    ax.set_xlabel("HJD'")
    ax.legend()
    plt.savefig(f'{event_path}/{evname}')
    plt.show()
    return None

def plot_2L1S_parallax(pars,dataset,event_path,xrange,yrange,evname,satellitedir):
    data = np.loadtxt(f'{event_path}/Data/{dataset}')
    VBMInstance = VBMicrolensing.VBMicrolensing()
    VBMInstance.RelTol = 1e-04
    VBMInstance.Tol = 1e-05
    coordinatefile = f'{event_path}/Data/event.coordinates'
    VBMInstance.SetObjectCoordinates(coordinatefile, satellitedir)
    VBMInstance.satellite=1
    tmin = pars[6]-5*pars[5]
    tmax = pars[6]+5*pars[5]
    if tmin < data[0,-1]:
        tmin = data[0,-1]
    if tmax > data[-1,-1]:
        tmax = data[-1,-1]
    tmag = np.linspace(tmin,tmax,80000)
    a1 = 0.33
    magnitude_list, source_flux_list, blend_flux_list = calculate_magnifications_plots(pars,[a1],[data],tmag,1,VBMInstance,parallax=True)
    magnitude = magnitude_list[0]
    fig, ax = plt.subplots(dpi=100, layout='tight')
    ax.plot(tmag, magnitude, zorder=10, label='2L1S Fit', color='black')
    #magnitude_list, source_flux_list, blend_flux_list = calculate_magnifications_pspl([0.6291636,50.20121621,pars[-1]], [data],tmag, 1, VBMInstance)
    #magnitude = magnitude_list[0]
    #ax.plot(tmag, magnitude, zorder=10, label='PSPL True', color='red')
    ax.errorbar(data[:, -1], y=data[:, 0], yerr=data[:, 1], marker='.', markersize=0.75,
                linestyle=' ', label='W146')
    # Set some good ylim based on the xlim if yrange==None.
    if yrange == None:
        ind_0 = np.where(tmag >= xrange[0])
        ind_1 = np.where(tmag <= xrange[1])

        ind_data_0 = np.where(data[:,-1] >= xrange[0])
        ind_data_1 = np.where(data[:, -1] <= xrange[1])
        ind_plot = np.intersect1d(ind_0,ind_1)
        ind_data = np.intersect1d(ind_data_0,ind_data_1)

        ymin = np.min([np.min(magnitude[ind_plot]) - 0.1,np.min(data[ind_data,0]) - 0.1])
        ymax = np.max([np.max(magnitude[ind_plot]) + 0.1,np.max(data[ind_data,0]) + 0.1])
        yrange=[ymin,ymax]
    ax.set_xlim(xrange[0],xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    ax.yaxis.set_inverted(True)
    ax.set_ylabel('W146')
    ax.set_xlabel("HJD'")
    ax.legend()
    plt.savefig(f'{event_path}/{evname}')
    plt.show()
    return None