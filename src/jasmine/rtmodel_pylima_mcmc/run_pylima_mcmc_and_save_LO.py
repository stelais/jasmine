import csv
import time

import numpy as np
import pandas as pd

from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.models import FSBL_model
from pyLIMA.fits import MCMCHDF5_fit,TRF_fit
from pyLIMA.models import pyLIMA_fancy_parameters

from src.jasmine.classes_and_files_reader import RTModel_results_cls as rtm_results
from src.jasmine.files_organizer import ra_and_dec_conversions as radec
from src.jasmine.classes_and_files_reader.best_models_wrap_up import finding_best_rtmodel_fit_name

def main(general_path_for_rtmodel_run_,
         pylima_data_folder_,
         model_path_for_initial_conditions_,
         satellite_directory_,
         event_name_,
         number_of_processors_=5,
         number_of_steps_=5000,
         number_of_walkers_=2,
         run_name=''):
    # WHERE YOUR DATA IS LOCATED:
    data_folder_path = f'{general_path_for_rtmodel_run_}/Data'
    ### Create a new EVENT object and give it a name.
    # Create new files
    #new_data_files(data_folder_path=data_folder_path,
    #               data_output_folder_path=pylima_data_folder_,
    #               event_number=event_name_)

    # Load previous information from RTModel
    print('Loading previous information from RTModel...')
    # Using jasmine
    rtm_model = rtm_results.ModelResults(file_to_be_read=model_path_for_initial_conditions_)
    # Coordinates
    file_path = f'{data_folder_path}/event.coordinates'  # Replace with the actual file path
    event_coordinates = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['RA_HMS', 'Dec_DMS'])

    # Convert RA and Dec to decimal degrees
    ra_deg = radec.ra_hms_to_deg(event_coordinates['RA_HMS'].iloc[0])
    dec_deg = radec.dec_dms_to_deg(event_coordinates['Dec_DMS'].iloc[0])

    your_event = event.Event(ra=ra_deg, dec=dec_deg)
    your_event.name = f'rep_sample_{event_name_}'

    ### Load up the data
    print('Loading data...')
    names = ['index','HJD','flux','err','sat']
    lc_to_fit = pd.read_csv(f'{general_path_for_rtmodel_run_}/LCToFit.txt', names=names,sep='\s+',skiprows=1)
    lc_to_fit['HJD']+=2_450_000 #pylima wants hjd not hjd - 2450000
    filters = []
    with open(f'{general_path_for_rtmodel_run_}/FilterToData.txt','r') as f:
        for line in f:
            s=line.strip('\n')
            filters.append(s)
    ald_coefficients = []
    with open(f'{general_path_for_rtmodel_run_}/Data/LimbDarkening.txt','r') as f:
        for line in f:
            s = float(line.strip('\n'))
            ald_coefficients.append(s)

    data_1 = lc_to_fit[lc_to_fit['sat']==1]
    data_2 = lc_to_fit[lc_to_fit['sat']==2]
    data_3 = lc_to_fit[lc_to_fit['sat']==3]
    index_1 = np.unique(data_1['index'])[0]
    index_2 = np.unique(data_2['index'])[0]
    index_3 = np.unique(data_3['index'])[0]
    ald_1 = ald_coefficients[index_1]
    ald_2 = ald_coefficients[index_2]
    ald_3 = ald_coefficients[index_3]
    print(index_3)



    ### IMPORTANT: Tell the code that Roman is in space!
    # read in ephemerides files
    names = ['time', 'RA','Dec', 'dist', 'deldot']
    t1_ephem = pd.read_csv(f'{satellite_directory_}/satellite1.txt',names=names,comment='$',sep = '\s+')
    t2_ephem = pd.read_csv(f'{satellite_directory_}/satellite2.txt', names=names, comment='$',sep = '\s+')
    t3_ephem = pd.read_csv(f'{satellite_directory_}/satellite3.txt', names=names, comment='$',sep = '\s+')

    data_2 = data_2.reset_index()
    data_3 = data_3.reset_index()
    data_1.loc[:,'HJD'] = t1_ephem.loc[:,'time']
    data_2.loc[:,'HJD'] = t2_ephem.loc[:,'time']
    data_3.loc[:,'HJD'] = t3_ephem.loc[:,'time']
    #print(data_3)
    #     [time,mag,err_mag]
    telescope_1 = telescopes.Telescope(name='RomanW146',
                                       camera_filter='W146',
                                       lightcurve=data_1[['HJD', 'flux', 'err']].values,
                                       lightcurve_names=['time', 'flux', 'err_flux'],
                                       lightcurve_units=['HJD', 'flux', 'flux'])
    telescope_2 = telescopes.Telescope(name='RomanZ087',
                                       camera_filter='Z087',
                                       lightcurve=data_2[['HJD', 'flux', 'err']].values,
                                       lightcurve_names=['time', 'flux', 'err_flux'],
                                       lightcurve_units=['HJD', 'flux', 'flux'])
    telescope_3 = telescopes.Telescope(name='RomanK213',
                                       camera_filter='K213',
                                       lightcurve=data_3[['HJD', 'flux', 'err']].values,
                                       lightcurve_names=['time', 'flux', 'err_flux'],
                                       lightcurve_units=['HJD', 'flux', 'flux'])




    #print(telescope_1.spacecraft_positions)

    telescope_1.location = 'Space'
    telescope_1.spacecraft_name = 'Roman'
    telescope_1.spacecraft_positions['photometry'] = t1_ephem[['time','RA','Dec','dist']].values
    telescope_2.location = 'Space'
    telescope_2.spacecraft_name = 'Roman'
    telescope_2.spacecraft_positions['photometry'] = t2_ephem[['time','RA', 'Dec', 'dist']].values
    telescope_3.location = 'Space'
    telescope_3.spacecraft_name = 'Roman'
    telescope_3.spacecraft_positions['photometry'] = t3_ephem[['time','RA', 'Dec', 'dist']].values
    #print(telescope_1.spacecraft_positions)
    # Adding limb darkening
    telescope_1.ld_a1 = ald_1
    telescope_2.ld_a1 = ald_2
    telescope_3.ld_a1 = ald_3


    ### Append these two telescope data sets to your EVENT object.
    print('Appending data...')
    your_event.telescopes.append(telescope_1)
    your_event.telescopes.append(telescope_2)
    your_event.telescopes.append(telescope_3)
    print(telescope_1.name,telescope_1.ld_a1)
    print(telescope_2.name, telescope_2.ld_a1)
    print(telescope_3.name, telescope_3.ld_a1)
    ### Define the survey telescope that you want to use to align all other data sets to.
    ### We recommend using the data set with the most measurements covering the gretest
    ### time span of observations:
    print('Defining survey telescope...')
    your_event.find_survey('RomanW146')

    ### Run a quick sanity check on your input.
    print('Checking event...')
    your_event.check_event()
    print(rtm_model.model_parameters.t0 + 2450000)
    ### CHOSE MODEL
    print('Choosing model...')
    # parallax should be close to t0
    #define log parameters with fancy parameters
    #my_pars = {'log_tE': 'tE','log_rho':'rho','log_mass_ratio':'mass_ratio','log_separation':'separation'}

    fancy = pyLIMA_fancy_parameters.StandardFancyParameters()
    if rtm_model.model_type == 'LS':
        model_to_mcmc = FSBL_model.FSBLmodel(your_event, parallax=['None', 0.0],
                                             orbital_motion=['None',0.0],fancy_parameters=fancy)
        guess_parameters = np.array(
            [rtm_model.model_parameters.t0 + 2450000, rtm_model.model_parameters.u0,
             np.log10(rtm_model.model_parameters.tE), np.log10(rtm_model.model_parameters.rho),
             np.log10(rtm_model.model_parameters.separation), np.log10(rtm_model.model_parameters.mass_ratio),
             rtm_model.model_parameters.alpha])


    elif  rtm_model.model_type == 'LX':
        model_to_mcmc = FSBL_model.FSBLmodel(your_event, parallax=['Full', rtm_model.model_parameters.t0 + 2450000],
                                             orbital_motion=['None', 0.0],fancy_parameters=fancy)
        guess_parameters = np.array(
            [rtm_model.model_parameters.t0 + 2450000, rtm_model.model_parameters.u0,
             np.log10(rtm_model.model_parameters.tE), np.log10(rtm_model.model_parameters.rho),
             np.log10(rtm_model.model_parameters.separation), np.log10(rtm_model.model_parameters.mass_ratio),
             rtm_model.model_parameters.alpha, rtm_model.model_parameters.piEN, rtm_model.model_parameters.piEE])

    elif rtm_model.model_type == 'LO':
        model_to_mcmc = FSBL_model.FSBLmodel(your_event, parallax=['Full', rtm_model.model_parameters.t0 + 2450000],
                                             orbital_motion=['Circular',rtm_model.model_parameters.t0 + 2450000],fancy_parameters=fancy)
        guess_parameters = np.array(
            [rtm_model.model_parameters.t0 + 2450000, rtm_model.model_parameters.u0,
             np.log10(rtm_model.model_parameters.tE), np.log10(rtm_model.model_parameters.rho),
             np.log10(rtm_model.model_parameters.separation), np.log10(rtm_model.model_parameters.mass_ratio),
             rtm_model.model_parameters.alpha, rtm_model.model_parameters.piEN, rtm_model.model_parameters.piEE,
             365.25*rtm_model.model_parameters.gamma1,365.25*rtm_model.model_parameters.gamma2,365.25*rtm_model.model_parameters.gammaz])
    else:
        print('Model not found - only supported LS, LO and LX ')
        return


    print('Gradient Fitting...')
    #my_fit = MCMC_fit.MCMCfit(model_to_mcmc, MCMC_links=number_of_steps_, MCMC_walkers=number_of_walkers_,
    #                          loss_function='chi2')
    gradient_fit = TRF_fit.TRFfit(model_to_mcmc,loss_function='chi2')
    print('Defining guess parameters...')
    gradient_fit.model_parameters_guess = guess_parameters
    print('Defining constraints...')
    #use very wide constraints.
    # my_fit.fit_parameters['t0'][1] = [2458000, 2464000]  # PyLima limits for t0 doesnt allow roman simulations
    gradient_fit.fit_parameters['u0'][1] = [-5.0, 5.0]  # PyLima has a short limit for u0
    gradient_fit.fit_parameters['log_rho'][1] = [-5.3, -0.3]  # PyLima has a low limit for rho
    gradient_fit.fit_parameters['log_separation'][1] = [-5.3, 2.]
    gradient_fit.fit_parameters['log_mass_ratio'][1] = [-7., 0.]
    print(rtm_model.model_type)
    if (rtm_model.model_type == 'LX') or (rtm_model.model_type == 'LO'):
        gradient_fit.fit_parameters['piEN'][1] = [-2.0, 2.0]
        gradient_fit.fit_parameters['piEE'][1] = [-2.0, 2.0]
        if rtm_model.model_type == 'LO':
            gradient_fit.fit_parameters['v_para'][1] = [-365.25, 365.25]
            gradient_fit.fit_parameters['v_perp'][1] = [-365.25, 365.25]
            gradient_fit.fit_parameters['v_radial'][1] = [1E-7, 365.25]

    #print(my_fit.model_chi2(parameters=guess_parameters))
    #print(gradient_fit.fit_parameters)

    gradient_fit.fit()#computational_pool=pool
    #my_fit.fit_outputs()
    print('TR Fit Done!')
    # File #1
    # Gather necessary data
    print('MCMC Fitting')
    hdf5path = f'{pylima_data_folder_}/{event_name_}_mcmc_chain{run_name}_{rtm_model.model_type}.hdf5'
    my_fit = MCMCHDF5_fit.MCMCfit(model_to_mcmc, MCMC_links=number_of_steps_, MCMC_walkers=number_of_walkers_,
                              loss_function='chi2',hdf5path=hdf5path)
    print('Defining guess parameters...')
    my_fit.model_parameters_guess = gradient_fit.fit_results['best_model']
    print('Defining constraints...')
    # use very wide constraints.
    # my_fit.fit_parameters['t0'][1] = [2458000, 2464000]  # PyLima limits for t0 doesnt allow roman simulations
    my_fit.fit_parameters['u0'][1] = [-5.0, 5.0]  # PyLima has a short limit for u0
    my_fit.fit_parameters['log_rho'][1] = [-5.3, -0.3]  # PyLima has a low limit for rho
    my_fit.fit_parameters['log_separation'][1] = [-5.3, 2.]
    my_fit.fit_parameters['log_mass_ratio'][1] = [-7., 0.]
    print(rtm_model.model_type)
    if (rtm_model.model_type == 'LX') or (rtm_model.model_type == 'LO'):
        my_fit.fit_parameters['piEN'][1] = [-2.0, 2.0]
        my_fit.fit_parameters['piEE'][1] = [-2.0, 2.0]
        if rtm_model.model_type == 'LO':
            my_fit.fit_parameters['v_para'][1] = [-365.25, 365.25]
            my_fit.fit_parameters['v_perp'][1] = [-365.25, 365.25]
            my_fit.fit_parameters['v_radial'][1] = [1E-7, 365.25]

    #pool = mul.Pool(processes=number_of_processors_)
    my_fit.fit(computational_pool=None)
    print('MCMC Done')
    event_name = my_fit.model.event.name
    ra = my_fit.model.event.ra
    dec = my_fit.model.event.dec
    east = my_fit.model.event.East
    north = my_fit.model.event.North
    parameters_names = list(my_fit.model.model_dictionnary.keys())
    best_model_ndarray = my_fit.fit_results['best_model']
    fit_time = my_fit.fit_results['fit_time']
    MCMC_links = my_fit.MCMC_links
    MCMC_walkers = my_fit.MCMC_walkers
    MCMC_chains_with_fluxes = my_fit.fit_results['MCMC_chains_with_fluxes']
    #print(my_fit.model_chi2(parameters=best_model_ndarray))
    #print(my_fit.model_likelihood(parameters=best_model_ndarray))
    saving_results_path = pylima_data_folder_
    # Write File #1
    #pyLIMA_plots.plot_lightcurves(model_to_mcmc,guess_parameters)
    #my_fit.fit_outputs(bokeh_plot=True,bokeh_plot_name=f'{pylima_data_folder_}/{event_name}plot.html')
    print('Saving best model...')
    with open(f'{saving_results_path}/{event_name_}_best_model{run_name}_{rtm_model.model_type}.csv', 'w', newline='') as csvfile1:
        csvfile1.write(f"# Event name: {event_name}\n")
        csvfile1.write(f"# RA: {ra}\n")
        csvfile1.write(f"# Dec: {dec}\n")
        csvfile1.write(f"# East: {east}\n")
        csvfile1.write(f"# North: {north}\n")

        writer = csv.writer(csvfile1)
        # Write header
        writer.writerow(parameters_names)
        # Write data
        writer.writerow(best_model_ndarray)

    # Write File #2
    print('Saving MCMC chains...')
    with open(f'{saving_results_path}/{event_name_}_mcmc_chain{run_name}_{rtm_model.model_type}.csv', 'w', newline='') as csvfile2:
        csvfile2.write(f"# Event name: {event_name}\n")
        csvfile2.write(f"# RA: {ra}\n")
        csvfile2.write(f"# Dec: {dec}\n")
        csvfile2.write(f"# East: {east}\n")
        csvfile2.write(f"# North: {north}\n")
        csvfile2.write(f"# Fit time: {fit_time}\n")
        csvfile2.write(f"# MCMC links: {MCMC_links}\n")
        csvfile2.write(f"# MCMC walkers: {MCMC_walkers}\n")
        writer = csv.writer(csvfile2)
        # Write header
        writer.writerow(parameters_names + ['likelihood','prior'])
        # Write data
        for quoted_arrays in MCMC_chains_with_fluxes:
            for array in quoted_arrays:
                writer.writerow(array)
    sampler = my_fit.fit_results['fit_object']
    print(f'Autocorr Time = {sampler.get_autocorr_time(tol=0)}')
    print(f'Acceptance Ratio =  {sampler.acceptance_fraction}')
    print('All done!')


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    start = time.time()
    event_name = 'event_0_90_1748'
    best_rtmodel_fit_table_path = '/RTModel_runs/sample_rtmodel_v2.4_LX_LO_summary.csv'
    model_name = finding_best_rtmodel_fit_name(event_name_=event_name,
                                               model_type_='LX',
                                               best_rtmodel_fit_table_path_=best_rtmodel_fit_table_path)
    number_of_process = 1
    number_of_steps = 5000
    number_of_walkers = 2
    # ###########################
    #     gs66-ponyta:
    # general_path_for_rtmodel_run = (f'/Users/sishitan/Documents/Scripts/RTModel_project/'
    #                                 f'rtmodel_pylima/datachallenge_events/event_{event_number:03}/')
    # pylima_data_folder = ('/Users/sishitan/Documents/Scripts/RTModel_project/'
    #                       'rtmodel_pylima/datachallenge_events/pylima_format_data')
    # ############################


    general_path_for_rtmodel_run = f'/Users/jmbrashe/VBBOrbital/NEWGULLS/MCMC/sample_rtmodel_v2.4/{event_name}'
    # where the data in pylima format will be saved
    pylima_data_folder = '/Users/jmbrashe/VBBOrbital/NEWGULLS/MCMC/pylima_outputs'
    model_path_for_initial_conditions = f'{general_path_for_rtmodel_run}/Models/{model_name}.txt'
    satellite_directory = '/Users/jmbrashe/VBBOrbital/NEWGULLS/MCMC/satellitedir'
    main(event_name_=event_name,
         general_path_for_rtmodel_run_=general_path_for_rtmodel_run,
         pylima_data_folder_=pylima_data_folder,
         model_path_for_initial_conditions_=model_path_for_initial_conditions,
         satellite_directory_=satellite_directory,
         number_of_processors_=number_of_process,
         number_of_steps_=number_of_steps,
         number_of_walkers_=number_of_walkers,
         run_name=f'_{number_of_steps}_{number_of_walkers}_{model_name}')
    finish = time.time()
    print('Pylima MCMC took', finish - start, f'seconds for event {event_name} using {number_of_process} processors,'
                                              f'{number_of_steps} number_of_steps '
                                              f'and {number_of_walkers} number_of_walkers.')