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
import multiprocessing as mp

from jasmine.files_organizer import ra_and_dec_conversions as radec
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.models import PSPL_model,FSPL_model
from pyLIMA.fits import DE_fit,TRF_fit

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
    """calculate residuals for model on each dataset provided"""
    residue_list = []
    for i in range(ndatasets):
        res = (magnitude_list[i] - mag_data_list[i]) / error_list[i]
        residue_list.append(res)
    residues = np.concatenate(residue_list)
    return residues

def calculate_magnifications(pars,a1_list,data_list,ndatasets,VBMInstance,parallax=False):
    """Calculate LC magnitudes in each passband provided for 2L1S"""
    magnitude_list = []
    source_flux_list = []
    blend_flux_list = []
    #loop over bands
    print(pars)
    for i in range(ndatasets):
        #must update a1 each time
        VBMInstance.a1 = a1_list[i]
        data = data_list[i]
        if parallax:
            #print(pars)
            #set satellite for each band if PLX
            VBMInstance.satellite = i + 1
            #print(VBMInstance.satellite)
            magnifications, y1, y2 = VBMInstance.BinaryLightCurveParallax(pars,data[:,-1])
        else:
            magnifications, y1, y2 = VBMInstance.BinaryLightCurve(pars, data[:, -1])
        # get the fluxes from measured lightcurve magnitudes
        meas_flux = 10**(-0.4*data[:,0])
        meas_flux_err = 0.4*np.log(10)*10**(-0.4*data[:,0])*data[:,1]
        #Fix so minimize_linear_parameters uses flux_err not mag_err
        # finds optimal linear source + blend flux parameters
        source_flux,blend_flux = minimize_linear_pars(meas_flux,meas_flux_err,magnifications)
        sim_flux = np.array(magnifications)*source_flux+blend_flux # model lightcurve fluxes
        source_flux_list.append(source_flux)
        blend_flux_list.append(blend_flux)
        sim_magnitudes = -2.5*np.log10(sim_flux)  # model lightcurve magnitudes
        magnitude_list.append(sim_magnitudes)

        #
    return magnitude_list,source_flux_list,blend_flux_list


def evaluate_model(psplpars, fsblpars, a1_list, data_list, VBMInstance, pspl_chi2,parallax=False):
    """ Calculate chi2 for one model in the grid
        fsblpars = [s,q,alpha,tstar]
        psplpars = follows VBMicrolensing format
        data_list list with a dataframe for each band in RTModel data format
        ndatasets = number of datasets
    """
    #blend_flux_list = []
    #source_flux_list = []
    ndatasets = len(data_list)
    # create parameter array for VBM.BinaryLightCurve()
    if parallax:
        pars = [np.log(fsblpars[0]), np.log(fsblpars[1]), psplpars[0], fsblpars[2], np.log(fsblpars[3] / psplpars[1]),
                np.log(psplpars[1]), psplpars[2],psplpars[3],psplpars[4]]
    else:
        pars = [np.log(fsblpars[0]), np.log(fsblpars[1]), psplpars[0], fsblpars[2], np.log(fsblpars[3] / psplpars[1]), np.log(psplpars[1]), psplpars[2]]
    # Now go to calculate_magnifications
    # This function will use VBM to get mags, then calculate source + blend fluxes.
    magnitude_list,source_flux_list,blend_flux_list = calculate_magnifications(pars,a1_list,data_list,ndatasets,VBMInstance,parallax=parallax)
    # Just the way I passed measured magnitudes to get_chi2(). Maybe I should change it because it's really terrible.
    magdata_list = []
    magerr_list = []
    for i in range(ndatasets):
        magdata_list.append(data_list[i][:,0])
        magerr_list.append(data_list[i][:,1])
    print(source_flux_list)

    chi2_list,chi2_sum = get_chi2(magnitude_list, magdata_list, magerr_list, ndatasets)
    # return this, it is one line of the grid output file from grid_fit()
    return_list = np.concatenate((pars, source_flux_list,blend_flux_list, chi2_list,[chi2_sum, chi2_sum-pspl_chi2]))

    fig, ax = plt.subplots()
    ax.errorbar(data_list[0][:,-1],data_list[0][:,0],yerr = data_list[0][:,1], marker='.', markersize=0.75,
                linestyle=' ', label='W146')
    ax.plot(data_list[0][:,-1], magnitude_list[0], zorder=10, label='2L1S Fit', color='black')

    ax.annotate(f'W146 chi2 = {chi2_list[0]}\nTotal chi2 = {chi2_sum}',xy=(0.8,0.9))
    ax.invert_yaxis()
    ax.set_xlim(10000, 10050)
    ax.legend()
    plt.show()
    print(' ')


    return return_list

def grid_fit(event_path, dataset_list, pspl_pars, grid_s, grid_q, grid_alpha, tstar, a1_list, pspl_chi2,parallax=False,satellitedir=None):
    """
        Fit grid of models to determine initial conditions
    """
    # read datasets from /Data
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
    #create VBM instance
    VBMInstance = VBMicrolensing.VBMicrolensing()
    VBMInstance.parallaxsystem = 1 # :( :( :( :(
    VBMInstance.t0_par_fixed = 1
    VBMInstance.t0_par = 10025.111593759
    #if Parallax set coordinates and read in satellite tables
    if parallax:
        VBMInstance.SetObjectCoordinates(f'{event_path}/Data/event.coordinates',satellitedir)
        #VBMInstance.SetObjectCoordinates('17:55:35.00561287 -30:12:38.19570995')
    VBMInstance.RelTol = 1e-03
    VBMInstance.Tol=1e-03
    # I was experimenting with fixing the t0_par to exactly what pyLIMA uses, didn't fix anything.

    # create parameter meshgrid
    s,q,alpha = np.meshgrid(grid_s,grid_q,grid_alpha)
    s= s.flatten()
    q = q.flatten()
    alpha = alpha.flatten()
    #print(s.shape[0])
    # Initialize results array, with size depending on if static
    # N parameters + source+blend fluxes + 3 bands chi2 + total chi2 + delta chi2 = 7/9 + 2+3*len(al_list)
    if parallax:
        grid_results = np.zeros(shape=(s.shape[0], 11 + 3 * len(a1_list)))
        print(f'PSPL + PLX pars: {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]} {pspl_pars[3]} {pspl_pars[4]}')
    else:
        grid_results = np.zeros(shape=(s.shape[0], 9 + 3 * len(a1_list)))
        print(f'PSPL pars: {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]}')
    print(f'Checking {s.shape[0]} Models on grid!')
    # loop over each model in the grid and call evaluate model
    for i in range(s.shape[0]):
        fsblpars = [s[i],q[i],alpha[i],tstar]
        #evaluate model is where chi2 is calculated.
        output = evaluate_model(pspl_pars, fsblpars, a1_list, data_list, VBMInstance, pspl_chi2, parallax=parallax)
        grid_results[i,:] = output
        print(f'{i} {s[i]} {q[i]} {alpha[i]} {tstar} {output[-2]}') # print this to see chi2 of each model.
        #if i%5000==0: print(f'{i} Models checked')
    print('Done checking models!')
    return grid_results

def pspl_fit_pyLIMA(event_path,dataset_list):
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
        data_list[i] = data_list[i][:, [2, 0, 1]]
        data_list[i][:,0]+=2450000 # add +2_450_000 for pyLIMA
    print('Using pyLIMA DE')

    file_path = f'{event_path}/Data/event.coordinates'
    event_coordinates = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['RA_HMS', 'Dec_DMS'])
    # Convert RA and Dec to decimal degrees
    ra_deg = radec.ra_hms_to_deg(event_coordinates['RA_HMS'].iloc[0])
    dec_deg = radec.dec_dms_to_deg(event_coordinates['Dec_DMS'].iloc[0])

    your_event = event.Event(ra=ra_deg, dec=dec_deg)
    your_event.name = f'PSPL Fit'
    telescope_1 = telescopes.Telescope(name='RomanW146',
                                       camera_filter='W146',
                                       lightcurve=data_list[0],
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])
    telescope_2 = telescopes.Telescope(name='RomanZ087',
                                       camera_filter='Z087',
                                       lightcurve=data_list[1],
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])
    telescope_3 = telescopes.Telescope(name='RomanK213',
                                       camera_filter='K213',
                                       lightcurve=data_list[2],
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])
    your_event.telescopes.append(telescope_1)
    your_event.telescopes.append(telescope_2)
    your_event.telescopes.append(telescope_3)
    your_event.find_survey('RomanW146')
    your_event.check_event()
    print(telescope_1.bad_data)
    pspl = PSPL_model.PSPLmodel(your_event, parallax=['None', 0.0])

    diffev_fit = DE_fit.DEfit(pspl, loss_function='chi2')
    diffev_fit.fit_parameters['u0'][1] = [0., 5.]
    diffev_fit.fit()
    # Do gradient fit to fine tune solution
    gradient_fit = TRF_fit.TRFfit(pspl, loss_function='chi2')
    gradient_fit.fit_parameters['u0'][1] = [0, 5.]  # PyLima has a short limit for u0

    gradient_fit.model_parameters_guess = diffev_fit.fit_results['best_model'][0:3]
    gradient_fit.fit()


    results = gradient_fit.fit_results['best_model']
    chi2 = gradient_fit.fit_results['chi2']
    return results,chi2

def psplPLX_fit_pyLIMA(event_path,dataset_list,satellitedir):
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
        data_list[i] = data_list[i][:, [2, 0, 1]]
        data_list[i][:,0]+=2450000 # add +2_450_000 for pyLIMA
    print('Using pyLIMA DE')

    file_path = f'{event_path}/Data/event.coordinates'
    event_coordinates = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['RA_HMS', 'Dec_DMS'])
    # Convert RA and Dec to decimal degrees
    ra_deg = radec.ra_hms_to_deg(event_coordinates['RA_HMS'].iloc[0])
    dec_deg = radec.dec_dms_to_deg(event_coordinates['Dec_DMS'].iloc[0])

    your_event = event.Event(ra=ra_deg, dec=dec_deg)
    your_event.name = f'PSPL + PLX Fit'
    telescope_1 = telescopes.Telescope(name='RomanW146',
                                       camera_filter='W146',
                                       lightcurve=data_list[0],
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])
    telescope_2 = telescopes.Telescope(name='RomanZ087',
                                       camera_filter='Z087',
                                       lightcurve=data_list[1],
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])
    telescope_3 = telescopes.Telescope(name='RomanK213',
                                       camera_filter='K213',
                                       lightcurve=data_list[2],
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'])

    ### IMPORTANT: Tell the code that Roman is in space!
    # read in ephemerides files
    names = ['time', 'RA','Dec', 'dist', 'deldot']
    t1_ephem = pd.read_csv(f'{satellitedir}/satellite1.txt',names=names,comment='$',sep = '\s+')
    t2_ephem = pd.read_csv(f'{satellitedir}/satellite2.txt', names=names, comment='$',sep = '\s+')
    t3_ephem = pd.read_csv(f'{satellitedir}/satellite3.txt', names=names, comment='$',sep = '\s+')

    telescope_1.location = 'Space'
    telescope_1.spacecraft_name = 'Roman'
    telescope_1.spacecraft_positions['photometry'] = t1_ephem[['time','RA','Dec','dist']].values
    telescope_2.location = 'Space'
    telescope_2.spacecraft_name = 'Roman'
    telescope_2.spacecraft_positions['photometry'] = t2_ephem[['time','RA', 'Dec', 'dist']].values
    telescope_3.location = 'Space'
    telescope_3.spacecraft_name = 'Roman'
    telescope_3.spacecraft_positions['photometry'] = t3_ephem[['time','RA', 'Dec', 'dist']].values


    your_event.telescopes.append(telescope_1)
    your_event.telescopes.append(telescope_2)
    your_event.telescopes.append(telescope_3)
    your_event.find_survey('RomanW146')
    your_event.check_event()
    print(telescope_1.bad_data)

    print(telescope_1.bad_data)
    pspl_noplx = PSPL_model.PSPLmodel(your_event, parallax=['None',0.0 ])

    diffev_fit = DE_fit.DEfit(pspl_noplx, loss_function='chi2')
    diffev_fit.fit_parameters['u0'][1] = [0., 5.]
    pool = mp.Pool(processes=5)
    diffev_fit.fit(computational_pool=pool)
    no_plx_t0 = diffev_fit.fit_results['best_model'][0]

    pspl_plx = PSPL_model.PSPLmodel(your_event, parallax=['Full', no_plx_t0])

    diffev_fit = DE_fit.DEfit(pspl_plx, loss_function='chi2')
    diffev_fit.fit_parameters['u0'][1] = [0., 5.]
    diffev_fit.fit(computational_pool=pool)
    # Do gradient fit to fine tune solution
    gradient_fit = TRF_fit.TRFfit(pspl_plx, loss_function='chi2')
    gradient_fit.fit_parameters['u0'][1] = [0, 5.]  # PyLima has a short limit for u0

    gradient_fit.model_parameters_guess = diffev_fit.fit_results['best_model'][0:5]
    gradient_fit.fit()


    results = gradient_fit.fit_results['best_model']
    chi2 = gradient_fit.fit_results['chi2']
    return results,chi2



def filter_by_q(grid_result,parallax=False,pspl_thresh=-50):
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
        #if best_model['delta_pspl_chi2'] <= pspl_thresh: removed for now
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
    if parallax:
        init_cond = best_grid_models.iloc[:, 0:9].values.copy()
    else:
        init_cond = best_grid_models.iloc[:, 0:7].values.copy()
    init_cond[:, 0] = np.exp(init_cond[:, 0])
    init_cond[:, 1] = np.exp(init_cond[:, 1])
    init_cond[:, 4] = np.exp(init_cond[:, 4])
    init_cond[:, 5] = np.exp(init_cond[:, 5])
    print('Done')
    return best_grid_models, init_cond

def run_event(event_path,dataset_list,grid_s,grid_q,grid_alpha,tstar,a1_list,pspl_thresh,processors,satellitedir,parallax=False,method='lm'):
    """ Wrapper Function to go from pspl_fit to final RTModel runs."""
    # First do the PSPL fit
    #results=pspl_fit_pyLIMA(event_path=event_path,dataset_list=dataset_list)
    #pspl_pars = [results['u0'],results['tE'],results['t0']]
    #pspl_chi2 = results['chi2']
    if parallax:
        pspl_results, pspl_chi2 = psplPLX_fit_pyLIMA(event_path=event_path, dataset_list=dataset_list,satellitedir=satellitedir)
        pspl_pars = [pspl_results[1], pspl_results[2], pspl_results[0] - 2_450_000,pspl_results[3],pspl_results[4]]
    else:
        pspl_results,pspl_chi2 = pspl_fit_pyLIMA(event_path=event_path,dataset_list=dataset_list)
        pspl_pars = [pspl_results[1], pspl_results[2], pspl_results[0] - 2_450_000]
    #if method == 'lm':
    #    pspl_chi2 = pspl_results.cost*2
    #else: pspl_chi2 = pspl_results.chi2

    print(pspl_chi2)
    print(pspl_pars)
    #save pspl fit to a txt file
    #
    time0 = time.time()
    grid_fit_results = grid_fit(event_path=event_path, dataset_list=dataset_list, pspl_pars=pspl_pars,
                                grid_s=grid_s,grid_q=grid_q,grid_alpha=grid_alpha,tstar=tstar,a1_list=a1_list,pspl_chi2=pspl_chi2,parallax=parallax)
    time1 = time.time()
    print(f'ICGS time: {time1-time0}')
    time0 = time.time()
    if parallax:
        names = ['log(s)', 'log(q)', 'u0', 'alpha', 'log(rho)', 'log(tE)', 't0','piEN','piEE', 'fs0', 'fb0', 'fs1', 'fb1', 'fs2',
                 'fb2', 'chi20', 'chi21', 'chi22', 'chi2sum', 'delta_pspl_chi2']
    else:
        names =  ['log(s)','log(q)','u0','alpha','log(rho)','log(tE)','t0','fs0','fb0','fs1','fb1','fs2','fb2','chi20','chi21','chi22','chi2sum','delta_pspl_chi2']
    grid_result_df = pd.DataFrame(grid_fit_results,columns=names)
    filtered_df,init_conds = filter_by_q(grid_result=grid_result_df,parallax=parallax,pspl_thresh=pspl_thresh)
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
    modeltypes = ['PS','PX','LS','LX','LO']
    if parallax:
        modeltypes = ['PX', 'LX', 'LO']
    rtm.set_satellite_dir(satellitedir=satellitedir)
    peak_threshold = 5
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    rtm.set_satellite_dir(satellitedir=satellitedir)
    rtm.config_Reader(otherseasons=0, binning=1000000,renormalize=0)
    rtm.config_InitCond(usesatellite=1, peakthreshold=peak_threshold,modelcategories=modeltypes)
    rtm.Reader()
    rtm.InitCond()
    #Do FSPL fit for comparison


    if parallax:
        print('Launching PX Fits')
        rtm.launch_fits('PX')
        rtm.ModelSelector('PX')

        print('Launching LX Fits')
        num_init_cond = init_conds.shape[0]
        for n in range(num_init_cond):
            init_cond = list(init_conds[n, :])
            # launch each fit from the init conds
            rtm.LevMar(f'LXfit{n:03}', parameters=init_cond)
            rtm.ModelSelector('LX')
    else:
        print('Launching PS Fits')
        rtm.launch_fits('PS')
        rtm.ModelSelector('PS')

        print('Launching PX Fits')
        rtm.launch_fits('PX')
        rtm.ModelSelector('PX')

        print('Launching LS Fits')
        num_init_cond = init_conds.shape[0]
        for n in range(num_init_cond):
            init_cond = list(init_conds[n,:])
            #launch each fit from the init conds
            rtm.LevMar(f'LSfit{n:03}',parameters = init_cond)
        rtm.ModelSelector('LS')
        print('Launching LX fits')
        rtm.launch_fits('LX')
        rtm.ModelSelector('LX')

    print('Launching LO fits')
    rtm.launch_fits('LO')
    rtm.ModelSelector('LO')
    rtm.Finalizer()
    print('Done')
    time1 = time.time()
    print(f'RTModel time: {time1 - time0}')
    return None


def run_event_from_crash(event_path,processors,satellitedir):
    """ run from an error (user) to final RTModel runs."""

    #Now run these in RTModel
    # Have RTModel prints go to log not stdout
    time0 = time.time()
    init_conds = np.loadtxt(fname=f'{event_path}/Data/ICGS_initconds.txt')
    rtm = RTModel.RTModel()
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    modeltypes = ['PS','LS','LX','LO']
    rtm.set_satellite_dir(satellitedir=satellitedir)
    peak_threshold = 5
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    rtm.set_satellite_dir(satellitedir=satellitedir)
    rtm.config_Reader(otherseasons=0, binning=1000000,renormalize=0)
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





