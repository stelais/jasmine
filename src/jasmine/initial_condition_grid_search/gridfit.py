import VBMicrolensing
import pandas as pd
import numpy as np
import RTModel
import os
import shutil
import matplotlib.pyplot as plt
import time

import glob
from joblib import Parallel, delayed
from jasmine.files_organizer import ra_and_dec_conversions as radec
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.models import PSPL_model
from pyLIMA.fits import DE_fit, TRF_fit


def minimize_linear_pars(y, err, x):
    """
    Exact solution to minimize linear eq chi2
    """
    alpha1 = np.sum(x / err ** 2)
    alpha2 = np.sum(1 / err ** 2)
    alpha3 = np.sum(y / err ** 2)
    alpha4 = np.sum((x / err) ** 2)
    alpha5 = np.sum(x * y / err ** 2)
    a = (alpha5 * alpha2 - alpha1 * alpha3) / (alpha4 * alpha2 - alpha1 ** 2)
    b = (alpha4 * alpha3 - alpha5 * alpha1) / (alpha4 * alpha2 - alpha1 ** 2)
    return a, b


def get_chi2(magnitude_list, mag_data_list, error_list, ndatasets):
    """calculate chi2 for model on each dataset provided"""
    chi2_list = []
    for i in range(ndatasets):
        res = (magnitude_list[i] - mag_data_list[i]) / error_list[i]
        chi2 = np.sum(np.power(res, 2))
        chi2_list.append(chi2)
    chi2_sum = np.sum(chi2_list)

    return chi2_list, chi2_sum


def get_residuals(magnitude_list, mag_data_list, error_list, ndatasets):
    """calculate residuals for model on each dataset provided"""
    residue_list = []
    for i in range(ndatasets):
        res = (magnitude_list[i] - mag_data_list[i]) / error_list[i]
        residue_list.append(res)
    residues = np.concatenate(residue_list)
    return residues


def calc_pspl_chi2(pars, args):
    """ Objective function to optimize for pspl_fit() """
    magnitude_list = []
    source_flux_list = []
    blend_flux_list = []
    data_list = args[0]
    ndatasets = args[1]
    VBMInstance = args[2]
    # logger = args[3]
    pars_log = [np.log(pars[0]), np.log(pars[1]), pars[2]]
    # logger.info(pars)
    fluxdata_list = []
    fluxerr_list = []
    flux_list = []
    for i in range(ndatasets):
        data = data_list[i]
        magnifications, y1, y2 = VBMInstance.PSPLLightCurve(pars_log, data[:, -1])
        meas_flux = 10 ** (-0.4 * data[:, 0])
        meas_flux_err = 0.4 * np.log(10) * 10 ** (-0.4 * data[:, 0]) * data[:, 1]
        fluxdata_list.append(meas_flux)
        fluxerr_list.append(meas_flux_err)
        source_flux, blend_flux = minimize_linear_pars(meas_flux, meas_flux_err, magnifications)
        sim_flux = np.array(magnifications) * source_flux + blend_flux
        flux_list.append(sim_flux)
        source_flux_list.append(source_flux)
        blend_flux_list.append(blend_flux)
        sim_magnitudes = -2.5 * np.log10(sim_flux)
        magnitude_list.append(sim_magnitudes)
    magdata_list = []
    magerr_list = []
    for i in range(ndatasets):
        magdata_list.append(data_list[i][:, 0])
        magerr_list.append(data_list[i][:, 1])
    chi2_list, chi2 = get_chi2(magnitude_list, magdata_list, magerr_list, ndatasets)
    chi2_listf, chi2f = get_chi2(flux_list, fluxdata_list, fluxerr_list, ndatasets)
    print('VBM PSPL chi2 with mag ', chi2)
    print('VBM PSPL chi2 with flux ', chi2f)
    print('Source Fluxes ', source_flux_list)
    print('Blend Fluxes ', blend_flux_list)
    return chi2


def calculate_magnifications(pars, a1_list, data_list, ndatasets, VBMInstance, parallax=False):
    """Calculate LC magnitudes in each passband provided for 2L1S"""
    magnitude_list = []
    flux_list = []
    fluxdata_list = []
    fluxerr_list = []
    source_flux_list = []
    blend_flux_list = []
    # loop over bands
    # print(pars)
    for i in range(ndatasets):
        # must update a1 each time
        VBMInstance.a1 = a1_list[i]
        data = data_list[i]
        if parallax:
            # print(pars)
            # set satellite for each band if PLX
            VBMInstance.satellite = i + 1
            # print(VBMInstance.satellite)
            magnifications, y1, y2 = VBMInstance.BinaryLightCurveParallax(pars, data[:, -1])
        else:
            magnifications, y1, y2 = VBMInstance.BinaryLightCurve(pars, data[:, -1])
        # get the fluxes from measured lightcurve magnitudes
        meas_flux = 10 ** (-0.4 * data[:, 0])
        meas_flux_err = 0.4 * np.log(10) * 10 ** (-0.4 * data[:, 0]) * data[:, 1]
        fluxdata_list.append(meas_flux)
        fluxerr_list.append(meas_flux_err)
        # Fix so minimize_linear_parameters uses flux_err not mag_err
        # finds optimal linear source + blend flux parameters
        source_flux, blend_flux = minimize_linear_pars(meas_flux, meas_flux_err, magnifications)
        sim_flux = np.array(magnifications) * source_flux + blend_flux  # model lightcurve fluxes
        source_flux_list.append(source_flux)
        blend_flux_list.append(blend_flux)
        sim_magnitudes = -2.5 * np.log10(sim_flux)  # model lightcurve magnitudes
        magnitude_list.append(sim_magnitudes)
        flux_list.append(sim_flux)

        #
    return magnitude_list, flux_list, fluxdata_list, fluxerr_list, source_flux_list, blend_flux_list


def calculate_magnifications_plots(pars, a1_list, data_list, data_mag, ndatasets, VBMInstance, parallax=False):
    """Calculate LC magnitudes in each passband provided for 2L1S"""
    magnitude_list = []
    flux_list = []
    fluxdata_list = []
    fluxerr_list = []
    source_flux_list = []
    blend_flux_list = []
    for i in range(ndatasets):
        VBMInstance.a1 = a1_list[i]
        data = data_list[i]
        if parallax == False:
            magnifications, y1, y2 = VBMInstance.BinaryLightCurve(pars, data[:, -1])
        else:
            magnifications, y1, y2, sorb = VBMInstance.BinaryLightCurveOrbital(pars, data[:, -1])
        meas_flux = 10 ** (-0.4 * data[:, 0])
        meas_flux_err = 0.4 * np.log(10) * 10 ** (-0.4 * data[:, 0]) * data[:, 1]
        fluxdata_list.append(meas_flux)
        fluxerr_list.append(meas_flux_err)
        # Fix so minimize_linear_parameters uses flux_err not mag_err
        source_flux, blend_flux = minimize_linear_pars(meas_flux, meas_flux_err, magnifications)
        magnifications, y1, y2 = VBMInstance.BinaryLightCurve(pars, data_mag)
        sim_flux = np.array(magnifications) * source_flux + blend_flux
        source_flux_list.append(source_flux)
        blend_flux_list.append(blend_flux)
        sim_magnitudes = -2.5 * np.log10(sim_flux)
        magnitude_list.append(sim_magnitudes)
        flux_list.append(sim_flux)
    return magnitude_list, flux_list, fluxdata_list, fluxerr_list, source_flux_list, blend_flux_list


def evaluate_model(psplpars, fsblpars, a1_list, data_list, VBMInstance, pspl_chi2, parallax=False):
    """ Calculate chi2 for one model in the grid
        fsblpars = [s,q,alpha,tstar]
        psplpars = follows VBMicrolensing format
        data_list list with a dataframe for each band in RTModel data format
        ndatasets = number of datasets
    """
    # blend_flux_list = []
    # source_flux_list = []
    ndatasets = len(data_list)
    # create parameter array for VBM.BinaryLightCurve()
    if parallax:
        pars = [np.log(fsblpars[0]), np.log(fsblpars[1]), psplpars[0], fsblpars[2], np.log(fsblpars[3] / psplpars[1]),
                np.log(psplpars[1]), psplpars[2], psplpars[3], psplpars[4]]
    else:
        pars = [np.log(fsblpars[0]), np.log(fsblpars[1]), psplpars[0], fsblpars[2],
                np.log(fsblpars[3] / psplpars[1]), np.log(psplpars[1]), psplpars[2]]
    # Now go to calculate_magnifications
    # This function will use VBM to get mags, then calculate source + blend fluxes.
    magnitude_list, flux_list, fluxdata_list, fluxerr_list, source_flux_list, blend_flux_list = calculate_magnifications(
        pars, a1_list, data_list, ndatasets, VBMInstance, parallax=parallax)
    # Just the way I passed measured magnitudes to get_chi2(). Maybe I should change it because it's really terrible.
    magdata_list = []
    magerr_list = []
    for i in range(ndatasets):
        magdata_list.append(data_list[i][:, 0])
        magerr_list.append(data_list[i][:, 1])
    # print(source_flux_list)

    chi2_list, chi2_sum = get_chi2(flux_list, fluxdata_list, fluxerr_list, ndatasets)
    # return this, it is one line of the grid output file from grid_fit()
    return_list = np.concatenate((pars, source_flux_list, blend_flux_list, chi2_list, [chi2_sum, chi2_sum - pspl_chi2]))

    # fig, ax = plt.subplots()
    # ax.errorbar(data_list[0][:,-1],data_list[0][:,0],yerr = data_list[0][:,1], marker='.', markersize=0.75,
    #            linestyle=' ', label='W146')
    # ax.plot(data_list[0][:,-1], magnitude_list[0], zorder=10, label='2L1S Fit', color='black')

    # ax.annotate(f'W146 chi2 = {chi2_list[0]}\nTotal chi2 = {chi2_sum}',xy=(0.8,0.9))
    # ax.invert_yaxis()
    # ax.set_xlim(10000, 10050)
    # ax.legend()
    # plt.show()
    # print(' ')
    return return_list


def grid_fit_pass_meshgrid(event_path, dataset_list, pspl_pars, s, q, alpha, tstar, a1_list, pspl_chi2):
    """
        Fit grid of models to determine initial conditions
        pass preconstructed meshgrid to the grid fit
    """
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
    VBMInstance = VBMicrolensing.VBMicrolensing()
    VBMInstance.RelTol = 1e-03
    VBMInstance.Tol = 1e-03
    # print(s.shape[0])
    grid_results = np.zeros(shape=(s.shape[0], 9 + 3 * len(a1_list)))
    print(f'PSPL pars: {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]}')
    print(f'Checking {s.shape[0]} Models on grid!')
    for i in range(s.shape[0]):
        fsblpars = [s[i], q[i], alpha[i], tstar]
        print(f'{i} {s[i]} {q[i]} {alpha[i]} {tstar}')
        output = evaluate_model(pspl_pars, fsblpars, a1_list, data_list, VBMInstance, pspl_chi2)
        grid_results[i, :] = output
        # if i%5000==0: print(f'{i} Models checked')
    print('Done checking models!')
    return grid_results


def make_grid(grid_s, grid_q, grid_alpha, use_croin=False):
    if use_croin:
        print('Implementation in progress')
    else:
        s, q, alpha = np.meshgrid(grid_s, grid_q, grid_alpha)
        s = s.flatten()
        q = q.flatten()
        alpha = alpha.flatten()
    return s, q, alpha


# Functions for compressing 3 nested loops into one and recovering the three indices
def CALCULATE_INDEX(i, j, k, ni, nj, nk):
    IND = i * nj * nk + j * nk + k
    return IND


def RECOVER_INDICES(IND, ni, nj, nk):
    i = np.int64(IND / (nj * nk))
    left = IND - (i * nj * nk)
    j = np.int64(left / nk)
    k = left - j * nk
    return i, j, k


def create_random_indices(event_path, nmodels, ncores):
    '''
    Divide nmodels on the grid into ncore batches to run.
    Randomize index order so one job doesn't get stuck on all the long ones
    '''
    index = np.arange(0, nmodels)
    nper_job = int(index.shape[0] / ncores)
    nper = nper_job * np.ones(ncores)
    left = index.shape[0] - ncores * nper_job
    for i in range(left):
        nper[i] += 1
    randomized_index = np.random.choice(index, nmodels, replace=False)
    start_ind = np.zeros(ncores, dtype=int)
    start_ind[1:ncores] = np.cumsum(nper, dtype=int)[0:ncores - 1]
    end_ind = np.cumsum(nper, dtype=int)[0:ncores]
    for i in range(ncores):
        sub_indices = randomized_index[start_ind[i]:end_ind[i]]
        #np.save(f'{event_path}/Data/sub_index_list_{i}.npy', sub_indices)
    return randomized_index, start_ind, end_ind


def evaluation_loop(common_parameters, index_list, iteration):
    # unpack values. This is why we need a class instead
    event_path = common_parameters[0]
    satellitedir = common_parameters[1]
    grid_q = common_parameters[2]
    grid_s = common_parameters[3]
    grid_alpha = common_parameters[4]
    tstar = common_parameters[5]
    pspl_pars = common_parameters[6]
    a1_list = common_parameters[7]
    data_list = common_parameters[8]
    pspl_chi2 = common_parameters[9]
    parallax = common_parameters[10]
    del common_parameters  # get that outta here

    VBMInstance = VBMicrolensing.VBMicrolensing()
    VBMInstance.parallaxsystem = 1  # :( :( :( :(
    # VBMInstance.t0_par_fixed = 1
    # VBMInstance.t0_par = 10025.111593759
    # if Parallax set coordinates and read in satellite tables
    if parallax:
        VBMInstance.SetObjectCoordinates(f'{event_path}/Data/event.coordinates', satellitedir)
    VBMInstance.Tol = 1.0e-02
    VBMInstance.RelTol = 1.0e-03  # leave at 0 for ICGS
    VBMInstance.minannuli = 2

    with open(f'{event_path}/Data/grid_fit{iteration}.txt', 'w') as grid_file:
        for index in index_list:
            grid_calculation_time_start = time.time() # Adding time track for each model evaluation/calculation
            i, j, k = RECOVER_INDICES(index, grid_q.shape[0], grid_s.shape[0], grid_alpha.shape[0])
            fsblpars = [grid_s[j], grid_q[i], grid_alpha[k], tstar]
            # evaluate model is where chi2 is calculated.
            output = evaluate_model(pspl_pars, fsblpars, a1_list, data_list, VBMInstance, pspl_chi2, parallax=parallax)
            grid_calculation_time_end = time.time()
            grid_calculation_time = grid_calculation_time_end - grid_calculation_time_start
            grid_file.write(f'{index} {output[-2]} {grid_calculation_time}\n')


def grid_fit_parallelized(event_path, dataset_list, pspl_pars, grid_s, grid_q, grid_alpha, tstar, a1_list, pspl_chi2,
                          parallax=False, satellitedir=None, use_croin=False, nprocessors=2):
    """
        Fit grid of models to determine initial conditions
    """
    # read datasets from /Data
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
    # create VBM instance

    # I was experimenting with fixing the t0_par to exactly what pyLIMA uses, didn't fix anything.
    # create parameter meshgrid
    # s,q,alpha = make_grid(grid_s,grid_q, grid_alpha, use_croin)
    # print(s.shape[0])
    n_models = grid_q.shape[0] * grid_s.shape[0] * grid_alpha.shape[0]
    randomized_index, start, end = create_random_indices(event_path, n_models, nprocessors)

    # Initialize results array, with size depending on if static
    # N parameters + source+blend fluxes + 3 bands chi2 + total chi2 + delta chi2 = 7/9 + 2+3*len(al_list)
    # calc_pspl_chi2(pspl_pars,[data_list,3,VBMInstance])

    print(f'Checking {n_models} Models on grid with {nprocessors} cores!')
    # loop over each model in the grid and call evaluate model
    # This is where the parallelization must be implemented. So write a function here
    # INDICES for grids
    common_parameters = [event_path, satellitedir, grid_q, grid_s, grid_alpha, tstar, pspl_pars, a1_list,
                         data_list, pspl_chi2, parallax]
    Parallel(n_jobs=nprocessors)(
        delayed(evaluation_loop)(common_parameters, randomized_index[start[iteration]:end[iteration]], iteration) for
        iteration in range(nprocessors))
    print('Done checking models!')
    return None


def grid_fit(event_path, dataset_list, pspl_pars, grid_s, grid_q, grid_alpha, tstar, a1_list, pspl_chi2, parallax=False,
             satellitedir=None, use_croin=False):
    """
        Fit grid of models to determine initial conditions
    """
    # read datasets from /Data
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
    # create VBM instance
    VBMInstance = VBMicrolensing.VBMicrolensing()
    VBMInstance.parallaxsystem = 1  # :( :( :( :(
    # VBMInstance.t0_par_fixed = 1
    # VBMInstance.t0_par = 10025.111593759
    # if Parallax set coordinates and read in satellite tables
    if parallax:
        VBMInstance.SetObjectCoordinates(f'{event_path}/Data/event.coordinates', satellitedir)
        # VBMInstance.SetObjectCoordinates('17:55:35.00561287 -30:12:38.19570995')
    VBMInstance.Tol = 1e-02
    VBMInstance.RelTol = 0  # leave at 0 for ICGS
    VBMInstance.minannuli = 2
    # I was experimenting with fixing the t0_par to exactly what pyLIMA uses, didn't fix anything.
    # create parameter meshgrid
    # s,q,alpha = make_grid(grid_s,grid_q, grid_alpha, use_croin)
    # print(s.shape[0])
    n_models = grid_q.shape[0] * grid_s.shape[0] * grid_alpha.shape[0]
    # Initialize results array, with size depending on if static
    # N parameters + source+blend fluxes + 3 bands chi2 + total chi2 + delta chi2 = 7/9 + 2+3*len(al_list)
    # calc_pspl_chi2(pspl_pars,[data_list,3,VBMInstance])
    if parallax:
        # grid_results = np.zeros(shape=(n_models, 11 + 3 * len(a1_list)))
        print(f'PSPL + PLX pars: {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]} {pspl_pars[3]} {pspl_pars[4]}')
    else:
        # grid_results = np.zeros(shape=(n_models, 9 + 3 * len(a1_list)))
        print(f'PSPL pars: {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]}')
    print(f'Checking {n_models} Models on grid!')
    # loop over each model in the grid and call evaluate model
    # This is where the parallelization must be implemented. So write a function here
    # INDICES for grids
    iter = 0
    with open(f'{event_path}/Data/grid_fit{iter}.txt', 'w') as grid_file:
        for index in range(n_models):
            grid_calculation_time_start = time.time() # Adding time track for each model evaluation/calculation
            i, j, k = RECOVER_INDICES(index, grid_q.shape[0], grid_s.shape[0], grid_alpha.shape[0])
            fsblpars = [grid_s[j], grid_q[i], grid_alpha[k], tstar]
            # evaluate model is where chi2 is calculated.
            output = evaluate_model(pspl_pars, fsblpars, a1_list, data_list, VBMInstance, pspl_chi2, parallax=parallax)
            grid_calculation_time_end = time.time()
            grid_calculation_time = grid_calculation_time_end - grid_calculation_time_start
            grid_file.write(f'{index} {output[-2]} {grid_calculation_time}\n')
            # grid_results[i,:] = output
        # print(f'{index} {s[i]} {q[i]} {alpha[i]} {tstar} {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]} {output[-2]}') # print this to see chi2 of each model.
        # if i%10==0: print(f'{i} Models checked')
    print('Done checking models!')
    return None


# def evaluate_multi()

def pspl_fit_pyLIMA(event_path, dataset_list):
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
        data_list[i] = data_list[i][:, [2, 0, 1]]
        data_list[i][:, 0] += 2450000  # add +2_450_000 for pyLIMA
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
    # print(telescope_1.bad_data)
    pspl = PSPL_model.PSPLmodel(your_event, parallax=['None', 0.0])

    diffev_fit = DE_fit.DEfit(pspl, loss_function='chi2')
    diffev_fit.fit_parameters['u0'][1] = [0., 5.]
    diffev_fit.fit_parameters['tE'][1] = [0.1, 200.0]
    diffev_fit.fit()
    # Do gradient fit to fine tune solution
    gradient_fit = TRF_fit.TRFfit(pspl, loss_function='chi2')
    gradient_fit.fit_parameters['u0'][1] = [0, 5.]  # PyLima has a short limit for u0
    gradient_fit.fit_parameters['tE'][1] = [0.1, 200.0]

    gradient_fit.model_parameters_guess = diffev_fit.fit_results['best_model'][0:3]
    gradient_fit.fit()

    results = gradient_fit.fit_results['best_model']
    chi2 = gradient_fit.fit_results['chi2']
    return results, chi2


def psplPLX_fit_pyLIMA(event_path, dataset_list, satellitedir):
    data_list = []
    for i in range(len(dataset_list)):
        data_list.append(np.loadtxt(f'{event_path}/Data/{dataset_list[i]}'))
        data_list[i] = data_list[i][:, [2, 0, 1]]
        data_list[i][:, 0] += 2450000  # add +2_450_000 for pyLIMA
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
    names = ['time', 'RA', 'Dec', 'dist', 'deldot']
    t1_ephem = pd.read_csv(f'{satellitedir}/satellite1.txt', names=names, comment='$', sep='\s+')
    t2_ephem = pd.read_csv(f'{satellitedir}/satellite2.txt', names=names, comment='$', sep='\s+')
    t3_ephem = pd.read_csv(f'{satellitedir}/satellite3.txt', names=names, comment='$', sep='\s+')

    telescope_1.location = 'Space'
    telescope_1.spacecraft_name = 'Roman'
    telescope_1.spacecraft_positions['photometry'] = t1_ephem[['time', 'RA', 'Dec', 'dist']].values
    telescope_2.location = 'Space'
    telescope_2.spacecraft_name = 'Roman'
    telescope_2.spacecraft_positions['photometry'] = t2_ephem[['time', 'RA', 'Dec', 'dist']].values
    telescope_3.location = 'Space'
    telescope_3.spacecraft_name = 'Roman'
    telescope_3.spacecraft_positions['photometry'] = t3_ephem[['time', 'RA', 'Dec', 'dist']].values

    your_event.telescopes.append(telescope_1)
    your_event.telescopes.append(telescope_2)
    your_event.telescopes.append(telescope_3)
    your_event.find_survey('RomanW146')
    your_event.check_event()
    # print(telescope_1.bad_data)

    # print(telescope_1.bad_data)
    pspl_noplx = PSPL_model.PSPLmodel(your_event, parallax=['None', 0.0])

    diffev_fit = DE_fit.DEfit(pspl_noplx, loss_function='chi2')
    diffev_fit.fit_parameters['u0'][1] = [0., 5.]
    pool = None  # mp.Pool(processes=5)
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

    # results = []
    # chi2 = []

    results = gradient_fit.fit_results['best_model']
    chi2 = gradient_fit.fit_results['chi2']

    # Now fit - u0 and add to the results array
    # diffev_fit.fit_parameters['u0'][1] = [-5., 0.]
    # diffev_fit.fit(computational_pool=pool)
    # Do gradient fit to fine tune solution
    # gradient_fit = TRF_fit.TRFfit(pspl_plx, loss_function='chi2')
    # gradient_fit.fit_parameters['u0'][1] = [-5., 0.]  # PyLima has a short limit for u0

    # gradient_fit.model_parameters_guess = diffev_fit.fit_results['best_model'][0:5]
    # gradient_fit.fit()

    # results.append(gradient_fit.fit_results['best_model'])
    # chi2.append(gradient_fit.fit_results['chi2'])
    return results, chi2


def filter_by_q(event_path, pspl_pars, pspl_chi2, tstar, grid_q, grid_s, grid_alpha, a1_list, parallax=False,
                pspl_thresh=-50):
    """
    Finds best initial conditions on grid for each value of q.
    Does a very simple check for s>1 s<1, but this often will not find an s/1/s degeneracy.
    """
    nq = grid_q.shape[0]
    ns = grid_s.shape[0]
    na = grid_alpha.shape[0]
    n_models = nq * ns * na
    grid_file = np.loadtxt(f'{event_path}/Data/grid_fit.txt')
    if parallax:
        best_grid_models = np.zeros(shape=(nq, 11))
        # print(f'PSPL + PLX pars: {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]} {pspl_pars[3]} {pspl_pars[4]}')
    else:
        best_grid_models = np.zeros(shape=(nq, 9))
        # print(f'PSPL pars: {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]}')
    index = np.arange(0, n_models)
    chi2 = grid_file[:, 1]
    q_index, s_index, alpha_index = RECOVER_INDICES(index, nq, ns, na)

    print(q_index.dtype, s_index.dtype, alpha_index.dtype)
    q_values = grid_q[q_index]
    s_values = grid_s[s_index]
    alpha_values = grid_alpha[alpha_index]
    print('Picking best models for each mass ratio')
    q_unique = list(np.unique(q_values))
    # print(q_unique)
    best_grid_model_indices = []
    for q in q_unique:
        # print(q)
        subgrid = index[np.where(q_values == q)[0]]
        print(subgrid.shape)
        # filter results for 1 q value.
        best_model_argsort_index = np.argsort(chi2[subgrid])[0]  # get index of best chi2
        best_model_index = subgrid[best_model_argsort_index]  # get
        best_grid_model_indices.append(best_model_index)
    # pars = [np.log(fsblpars[0]), np.log(fsblpars[1]), psplpars[0], fsblpars[2],
    # np.log(fsblpars[3] / psplpars[1]), np.log(psplpars[1]), psplpars[2]]

    # now reconstruct actual inital condition
    for i in range(len(best_grid_model_indices)):
        ind = best_grid_model_indices[i]
        best_grid_models[i, 0] = s_values[ind]
        best_grid_models[i, 1] = q_values[ind]
        best_grid_models[i, 2] = pspl_pars[0]
        best_grid_models[i, 3] = alpha_values[ind]
        best_grid_models[i, 4] = tstar / pspl_pars[1]
        best_grid_models[i, 5] = pspl_pars[1]
        best_grid_models[i, 6] = pspl_pars[2]
        if parallax:
            best_grid_models[i, 7] = pspl_pars[3]
            best_grid_models[i, 8] = pspl_pars[4]
            best_grid_models[i, 9] = chi2[ind]
            best_grid_models[i, 10] = pspl_chi2 - chi2[ind]
        else:
            best_grid_models[i, 7] = chi2[ind]
            best_grid_models[i, 8] = pspl_chi2 - chi2[ind]

    print('Done')
    return best_grid_models


def filter_by_q_and_s(event_path, pspl_pars, pspl_chi2, tstar, grid_q, grid_s, grid_alpha,
                      parallax=False, pspl_thresh=0,
                      number_of_best_models_before_q_filter=13, number_of_final_models=20,
                      number_of_minimum_unique_q=4):
    """
    Finds best initial conditions on grid, with only one model allowed for each q and s.
    Tries to ensure 4 unique mass ratios, but will stop at 20 total initial conditions.
    Mini steps:
    1. Saves 13 best models with unique combination of q and s.
    2a. If < 4 unique mass ratios, check for 3 best models with different mass ratios, then 2, then 2 again.
    In a way there are 4 unique mass ratios.
    2b. If >= 4 unique mass ratios, add best models with unique q,s until 20 initial conditions.
    """

    # Number of models in the grid
    nq, ns, na = grid_q.shape[0], grid_s.shape[0], grid_alpha.shape[0]
    n_models = nq * ns * na
    # Read in grid file
    grid_file = np.loadtxt(f'{event_path}/Data/grid_fit.txt')
    chi2 = grid_file[:, 1]
    delta_chi2 = pspl_chi2 - chi2
    # Recover q,s,alpha values for each model
    index = np.arange(0, n_models)
    q_index, s_index, alpha_index = RECOVER_INDICES(index, nq, ns, na)
    print(q_index.dtype, s_index.dtype, alpha_index.dtype)
    q_values, s_values, alpha_values = grid_q[q_index], grid_s[s_index], grid_alpha[alpha_index]



    # Sort all models by chi2 (ascending)
    argsort_chi2_indices = np.argsort(chi2)
    delta_chi2_le_0_flag = 0  # flag to set if all models that improve the chi2 have been found.
    best_grid_model_indices = []
    best_grid_model_q_s_tuples = []

    # ----------------------
    # Step 1: select top 13 (number_of_best_models_before_q_filter) unique (q,s)
    # ----------------------
    print('Picking best models. 1 alpha per q,s pair allowed.')
    saved_skipped_indices = []
    last_step1_model_index = 0
    for model_index in range(n_models):
        if len(best_grid_model_indices) == number_of_best_models_before_q_filter:
            last_step1_model_index = model_index - 1  # Just to be true to the variable name
            #print(f'Step 1: {len(best_grid_model_indices)} initial conditions selected')
            break
        current_index = argsort_chi2_indices[model_index]
        q, s = q_values[current_index], s_values[current_index]
        if (q, s) in best_grid_model_q_s_tuples:
            saved_skipped_indices.append(model_index)
            continue
        else:
            if delta_chi2[current_index] > pspl_thresh:
                best_grid_model_indices.append(current_index)
                best_grid_model_q_s_tuples.append((q, s))
            else:
                print("No more models improve the chi2 over the pspl!")
                delta_chi2_le_0_flag = 1
                break
    # If we already have all models that improve chi2, print and go to best_model_indices part:
    if delta_chi2_le_0_flag:
        print(f'{len(best_grid_model_indices)} initial conditions selected')

    # ----------------------
    # Step 2: fill remaining slots to reach number_of_final_models (20)
    # ----------------------
    else:
        # Track unique q (mass ratios)
        q_in_init_conds = []
        for qs in best_grid_model_q_s_tuples:
            q_in_init_conds.append(qs[0])
        unique_q_in_init_conds = np.unique(q_in_init_conds)

        # if there are already 4 unique mass ratios, continue adding best (q,s) models
        if len(unique_q_in_init_conds) >= number_of_minimum_unique_q:
            start_i = last_step1_model_index + 1
            for remaining_model_index in range(start_i, n_models):
                if len(best_grid_model_indices) == number_of_final_models:
                    #print(f'Step 2: {len(best_grid_model_indices)} initial conditions selected')
                    break
                current_index = argsort_chi2_indices[remaining_model_index]
                s = s_values[current_index]
                q = q_values[current_index]
                if (q, s) in best_grid_model_q_s_tuples:
                    saved_skipped_indices.append(remaining_model_index)
                    continue
                else:
                    if delta_chi2[current_index] > pspl_thresh:
                        best_grid_model_indices.append(current_index)
                        best_grid_model_q_s_tuples.append((q, s))
                    else:
                        print("No more models improve the chi2 over the pspl!")
                        delta_chi2_le_0_flag = 1
                        break
        # however, if there are less than 4 unique mass ratios, try to add more
        else:
            print("Warning! Less than 4 unique mass ratios in the initial conditions")
            print("Adding more initial conditions")
            # batch_sizes = [3, 2, 2]
            # That's what we have /\ , but writing it more generic:
            number_of_models_that_are_needed = number_of_final_models - number_of_best_models_before_q_filter
            batch_sizes = [number_of_models_that_are_needed // (number_of_minimum_unique_q - 1) +
                           (i < (number_of_models_that_are_needed) % (number_of_minimum_unique_q - 1))
                           for i in range(number_of_minimum_unique_q - 1)]
            start_i = last_step1_model_index + 1
            nstop = number_of_best_models_before_q_filter
            for batch_counter in range(3):
                nstop += batch_sizes[batch_counter]
                #print('nstop',nstop)
                if len(unique_q_in_init_conds) < number_of_minimum_unique_q:
                    for new_model_index in range(start_i, n_models):
                        if len(best_grid_model_indices) == nstop:
                            # If batch is complete, it recalculates again the number of unique mass ratios
                            # and moves the start index to the last model added and breaks to start a new batch
                            q_in_init_conds = []
                            for qs in best_grid_model_q_s_tuples:
                                q_in_init_conds.append(qs[0])
                            unique_q_in_init_conds = np.unique(q_in_init_conds)
                            start_i = new_model_index
                            break
                        else:
                            # While batch is not complete, keep adding models
                            # if unique q is still not reached, only add models with new mass ratios:
                            current_index = argsort_chi2_indices[new_model_index]
                            q, s = q_values[current_index], s_values[current_index]
                            if ((q, s) in best_grid_model_q_s_tuples) or (q in unique_q_in_init_conds):
                                saved_skipped_indices.append(new_model_index)
                                continue
                            else:
                                if delta_chi2[current_index] > pspl_thresh:
                                    best_grid_model_indices.append(current_index)
                                    best_grid_model_q_s_tuples.append((q, s))
                                    #print(f'Step 3a: {len(best_grid_model_indices)} initial conditions selected')
                                else:
                                    print("No more models improve the chi2 over the pspl!")
                                    #print(f'Step 3b: {len(best_grid_model_indices)} initial conditions selected')
                                    delta_chi2_le_0_flag = 1
                                    break
                else:
                    print('Enough unique q added, adding skipped models.')
                    # if minimum unique q is reached, complete with next models from skipped models
                    # calculate how many more models are needed to reach 20
                    number_of_models_needed = number_of_final_models - len(best_grid_model_indices)
                    for skipped_index in range(0, number_of_models_needed):
                        if len(best_grid_model_indices) == nstop:
                            # If batch is complete, it breaks to start a new batch
                            #print(f'Step 3c: {len(best_grid_model_indices)} initial conditions selected')
                            break
                        else:
                            if skipped_index < len(saved_skipped_indices):
                                saved_index = saved_skipped_indices[skipped_index]
                            else:
                                saved_index = start_i  # fallback if no more skipped indices

                            current_index = argsort_chi2_indices[saved_index]
                            q, s = q_values[current_index], s_values[current_index]
                            if delta_chi2[current_index] > pspl_thresh:
                                best_grid_model_indices.append(current_index)
                                best_grid_model_q_s_tuples.append((q, s))
                                #print(f'Step 3d: {len(best_grid_model_indices)} initial conditions selected')
                            else:
                                print("No more models improve the chi2 over the pspl!")
                                #print(f'Step 3b: {len(best_grid_model_indices)} initial conditions selected')
                                delta_chi2_le_0_flag = 1
                                break

    # ----------------------
    # Step 3: populate best_grid_models array
    # ----------------------
    # now reconstruct actual initial conditions
    # Initialize best grid models array, with size depending on if static
    if parallax:
        best_grid_models = np.zeros(shape=(len(best_grid_model_indices), 11))
        # print(f'PSPL + PLX pars: {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]} {pspl_pars[3]} {pspl_pars[4]}')
    else:
        best_grid_models = np.zeros(shape=(len(best_grid_model_indices), 9))
        # print(f'PSPL pars: {pspl_pars[0]} {pspl_pars[1]} {pspl_pars[2]}')

    for best_model_index in range(len(best_grid_model_indices)):
        ind = best_grid_model_indices[best_model_index]
        best_grid_models[best_model_index, 0] = s_values[ind]
        best_grid_models[best_model_index, 1] = q_values[ind]
        best_grid_models[best_model_index, 2] = pspl_pars[0]
        best_grid_models[best_model_index, 3] = alpha_values[ind]
        best_grid_models[best_model_index, 4] = tstar / pspl_pars[1]
        best_grid_models[best_model_index, 5] = pspl_pars[1]
        best_grid_models[best_model_index, 6] = pspl_pars[2]
        if parallax:
            best_grid_models[best_model_index, 7] = pspl_pars[3]
            best_grid_models[best_model_index, 8] = pspl_pars[4]
            best_grid_models[best_model_index, 9] = chi2[ind]
            best_grid_models[best_model_index, 10] = pspl_chi2 - chi2[ind]
        else:
            best_grid_models[best_model_index, 7] = chi2[ind]
            best_grid_models[best_model_index, 8] = pspl_chi2 - chi2[ind]

    print('Done')
    return best_grid_models


def combine_grid_files(event_path, parallel):
    data_path = f'{event_path}/Data'
    if parallel:
        if os.path.exists(f'{data_path}/grid_fit.txt'):
            os.remove(f'{data_path}/grid_fit.txt')
        grid_files = glob.glob(f'{data_path}/grid_fit*')
        dataframes = []
        for file in grid_files:
            print(file)
            #print(len(dataframes))
            dataframes.append(pd.read_csv(file, names=['index', 'chi2', 'cal_time'], sep='\s+'))
            os.remove(file) # remove now unecessary file
            #print(len(dataframes))
        #print(dataframes)
        grid_fit = pd.concat(dataframes)
        del dataframes
        grid_fit = grid_fit.sort_values('index')
        grid_fit.to_csv(f'{data_path}/grid_fit.txt', index=False, header=False, sep=' ')
    else:
        if os.path.exists(f'{data_path}/grid_fit0.txt'):
            shutil.move(f'{event_path}/Data/grid_fit0.txt', f'{event_path}/Data/grid_fit.txt')
        else:
            raise FileNotFoundError
    return None

def write_inital_conditions(event_path,initial_conditions,parallax=False):
    ninit_conds = initial_conditions.shape[0]
    if parallax:
        model = 'LX'
    else:
        model = 'LS'
    temppath = f'{event_path}/InitCond/InitCond{model}.tmp'
    initpath = f'{event_path}/InitCond/InitCond{model}.txt'
    with open(temppath,'w') as tempinit:
        with open(initpath, 'r') as oldinit:
            npeaks_nconds = oldinit.readline()
            npeaks = int(npeaks_nconds.split(' ')[0])
            npeaks_nconds = f'{npeaks} {ninit_conds}\n'
            tempinit.write(npeaks_nconds)
            # write peaks from old file
            for n in range(npeaks):
                peak = oldinit.readline()
                tempinit.write(peak)
    #now write initial conditons to file
    initial_conditions.iloc[:,0:7].to_csv(temppath,sep=' ', mode='a',header=False, index = False)
    os.remove(initpath)
    os.rename(temppath,initpath)
    return None



def run_event(event_path, dataset_list, grid_s, grid_q, grid_alpha, tstar, a1_list, pspl_thresh, processors,
              satellitedir, parallax=False, method='lm', use_saved_pspl=False):
    """ Wrapper Function to go from pspl_fit to final RTModel runs."""
    # First do the PSPL fit
    # results=pspl_fit_pyLIMA(event_path=event_path,dataset_list=dataset_list)
    # pspl_pars = [results['u0'],results['tE'],results['t0']]
    # pspl_chi2 = results['chi2']
    if use_saved_pspl:
        pspl = np.loadtxt(f'{event_path}/Data/pspl_pars.txt', delimiter=',')
        if parallax:
            pspl_pars = pspl[0:5]
        else:
            pspl_pars = pspl[0:3]
        pspl_chi2 = pspl[-1]
    else:
        if parallax:
            pspl_results, pspl_chi2 = psplPLX_fit_pyLIMA(event_path=event_path, dataset_list=dataset_list,
                                                         satellitedir=satellitedir)
            pspl_pars = [pspl_results[1], pspl_results[2], pspl_results[0] - 2_450_000, pspl_results[3],
                         pspl_results[4]]
        else:
            pspl_results, pspl_chi2 = pspl_fit_pyLIMA(event_path=event_path, dataset_list=dataset_list)
            pspl_pars = [pspl_results[1], pspl_results[2], pspl_results[0] - 2_450_000]
    # if method == 'lm':
    #    pspl_chi2 = pspl_results.cost*2
    # else: pspl_chi2 = pspl_results.chi2

    print(pspl_chi2)

    print(pspl_pars)
    # save pspl fit to a txt file
    #
    time0 = time.time()
    if processors == 1:
        parallel = False
        # grid_file_name = f'{event_path}/Data/grid_fit.txt'
        grid_fit(event_path=event_path, dataset_list=dataset_list, pspl_pars=pspl_pars,
                 grid_s=grid_s, grid_q=grid_q, grid_alpha=grid_alpha, tstar=tstar, a1_list=a1_list, pspl_chi2=pspl_chi2,
                 parallax=parallax)
    else:
        parallel = True
        grid_fit_parallelized(event_path=event_path, dataset_list=dataset_list, pspl_pars=pspl_pars,
                              grid_s=grid_s, grid_q=grid_q, grid_alpha=grid_alpha, tstar=tstar, a1_list=a1_list,
                              pspl_chi2=pspl_chi2,
                              parallax=parallax, use_croin=False, nprocessors=processors)
    time1 = time.time()
    print(f'ICGS time: {time1 - time0}')

    time0 = time.time()

    if parallax:
        names = ['log(s)', 'log(q)', 'u0', 'alpha', 'log(rho)', 'log(tE)', 't0', 'piEN', 'piEE', 'chi2sum',
                 'delta_pspl_chi2']
    else:
        names = ['log(s)', 'log(q)', 'u0', 'alpha', 'log(rho)', 'log(tE)', 't0', 'chi2sum', 'delta_pspl_chi2']

    # Now run these in RTModel
    # Have RTModel prints go to log not stdout
    rtm = RTModel.RTModel()
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    # if os.path.exists(f'{event_path}/Nature.txt'):
    rtm.archive_run()
    # have to remove old stuff or it affects the InitConds for everything.
    archive_list = glob.glob(f'{event_path}/run-*')
    for archived_run in archive_list:
        shutil.rmtree(archived_run)
    # Write some outputs after clearing the directory
    with open(f'{event_path}/Data/pspl_pars.txt', 'w') as f:
        f.write(f'{pspl_pars[0]},{pspl_pars[1]},{pspl_pars[2]},{pspl_chi2}')
    # np.savetxt(fname=f'{event_path}/Data/ICGS_initconds.txt', X=init_conds)  # save init conds to a text file
    # np.savetxt(f'{event_path}/Data/grid_fit.txt', grid_fit_results) # change to npy or parquet later
    
    combine_grid_files(event_path, parallel=parallel)
    init_conds = filter_by_q_and_s(event_path, pspl_pars, pspl_chi2, tstar, grid_q, grid_s, grid_alpha, parallax,
                                   pspl_thresh=0)
    filtered_df = pd.DataFrame(init_conds, columns=names)
    np.savetxt(f'{event_path}/Data/ICGS_initconds_chi2.txt', filtered_df.values)

    if parallax:
        # nostatic = True
        modeltypes = ['PS', 'PX', 'LX', 'LO']
    else:
        # nostatic = False
        modeltypes = ['PS', 'PX', 'LS', 'LX', 'LO']

    rtm.set_satellite_dir(satellitedir=satellitedir)
    peak_threshold = 5
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    rtm.set_satellite_dir(satellitedir=satellitedir)
    rtm.config_Reader(otherseasons=0, binning=1000000, renormalize=0)
    rtm.config_InitCond(usesatellite=1, peakthreshold=peak_threshold, modelcategories=modeltypes),  # nostatic=nostatic)
    rtm.Reader()
    rtm.InitCond()


    # Do FSPL fit for comparison

    if parallax:

        print('Launching PS Fits')
        rtm.launch_fits('PS')
        rtm.ModelSelector('PS')

        print('Launching PX Fits')
        rtm.launch_fits('PX')
        rtm.ModelSelector('PX')

        write_inital_conditions(event_path, filtered_df, parallax=parallax)
        print('Launching LX Fits')
        rtm.launch_fits('LX')
        rtm.ModelSelector('LX')


    else:
        print('Launching PS Fits')
        rtm.launch_fits('PS')
        rtm.ModelSelector('PS')

        print('Launching PX Fits')
        rtm.launch_fits('PX')
        rtm.ModelSelector('PX')

        write_inital_conditions(event_path, filtered_df, parallax=parallax)
        print('Launching LS Fits')
        rtm.launch_fits('LS')
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


def run_event_from_crash(event_path, processors, satellitedir):
    """ run from an error (user) to final RTModel runs."""

    # Now run these in RTModel
    # Have RTModel prints go to log not stdout
    time0 = time.time()
    init_conds = np.loadtxt(fname=f'{event_path}/Data/ICGS_initconds.txt')
    rtm = RTModel.RTModel()
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    modeltypes = ['PS', 'LS', 'LX', 'LO']
    rtm.set_satellite_dir(satellitedir=satellitedir)
    peak_threshold = 5
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    rtm.set_satellite_dir(satellitedir=satellitedir)
    rtm.config_Reader(otherseasons=0, binning=1000000, renormalize=0)
    rtm.config_InitCond(usesatellite=1, peakthreshold=peak_threshold, modelcategories=modeltypes)
    rtm.Reader()
    rtm.InitCond()
    # Do FSPL fit for comparison
    print('Launching PS Fits')
    rtm.launch_fits('PS')
    rtm.ModelSelector('PS')
    print('Launching LS Fits')
    num_init_cond = init_conds.shape[0]
    for n in range(num_init_cond):
        init_cond = list(init_conds[n, :])
        # launch each fit from the init conds
        rtm.LevMar(f'LSfit{n:03}', parameters=init_cond)
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


def update_event(event_path, processors, satellitedir):
    """ update rtmodel run."""

    rtm = RTModel.RTModel()
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    # if os.path.exists(f'{event_path}/Nature.txt'):
    rtm.archive_run()
    rtm.set_satellite_dir(satellitedir=satellitedir)
    peak_threshold = 5
    rtm.set_processors(nprocessors=processors)
    rtm.set_event(event_path)
    rtm.set_satellite_dir(satellitedir=satellitedir)
    rtm.config_Reader(otherseasons=0, binning=1000000, renormalize=0)
    rtm.config_InitCond(usesatellite=1, onlyupdate=True, oldmodels=10,
                        modelcategories=['PS', 'LX', 'LO']),  # nostatic=nostatic)
    rtm.run()
    return None


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
