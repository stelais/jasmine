import pandas as pd
import numpy as np


def bound_planet_master_reader(filepath_='./OMPLLD_croin_cassan.sample.csv'):
    """
    This function reads the bound planet file and returns a pandas dataframe with the columns
    :param filepath_:
    :return: bound_planet_df
    """
    bound_planet_df = pd.read_csv(filepath_)
    return bound_planet_df


def lightcurve_data_reader(SubRun, Field, ID, *, folder_path_='data', include_ephem=True):
    """
    This function reads the lightcurve data file and returns a pandas dataframe with the BJD Magnitude Error columns
    :param folder_path_:
    :param data_challenge_lc_number_:
    :param filter_:
    :return: lightcurve_data_df
    """
    fname = f'OMPLDG_croin_cassan_{SubRun}_{Field}_{ID}.det.lc'

    with open(f'{folder_path_}/{fname}', 'r') as f:
        fs = f.readline()
        fs = np.array(fs.split(' ')[1:4]).astype(float)
        f.readline()
        f.readline()
        m_source = f.readline()
        m_source = np.array(m_source.split(' ')[1:4]).astype(float)
    if include_ephem:
        columns = ['Simulation_time', 'measured_relative_flux', 'measured_relative_flux_error', 'true_relative_flux',
                   'true_relative_flux_error', 'observatory_code', 'saturation_flag', 'best_single_lens_fit',
                   'parallax_shift_t', 'parallax_shift_u', 'BJD', 'source_x', 'source_y', 'lens1_x', 'lens1_y',
                   'lens2_x', 'lens2_y', 'SAT_X', 'SAT_Y', 'SAT_Z']
    else:
        columns = ['Simulation_time', 'measured_relative_flux', 'measured_relative_flux_error', 'true_relative_flux',
                   'true_relative_flux_error', 'observatory_code', 'saturation_flag', 'best_single_lens_fit',
                   'parallax_shift_t', 'parallax_shift_u', 'BJD', 'source_x', 'source_y', 'lens1_x', 'lens1_y',
                   'lens2_x', 'lens2_y']
    lightcurve_data_df = pd.read_csv(f'{folder_path_}/{fname}', names=columns, comment='#', sep='\s+')
    lightcurve_data_df['days'] = lightcurve_data_df['BJD'] - 2450000
    lightcurve_list = []
    for observatory_code in range(3):
        obs_data = lightcurve_data_df[lightcurve_data_df['observatory_code'] == observatory_code]
        mag_constant = m_source[observatory_code] + 2.5 * np.log10(fs[observatory_code])
        if np.min(obs_data['measured_relative_flux']) < 0:
            obs_data.loc[:, 'measured_relative_flux'] = obs_data['measured_relative_flux'] - np.min(
                obs_data['measured_relative_flux']) + 1e-5
        measured_relative_flux_error = obs_data['measured_relative_flux_error']
        mag = - 2.5 * np.log10(obs_data['measured_relative_flux']) + mag_constant
        mag_err = (2.5 / (np.log(10))) * measured_relative_flux_error / obs_data['measured_relative_flux']
        obs_data.insert(loc=obs_data.shape[1], column='mag', value=mag)
        obs_data.insert(loc=obs_data.shape[1], column='mag_err', value=mag_err)
        lightcurve_list.append(obs_data[['mag', 'mag_err', 'days']])
    return lightcurve_list


def microlensing_signal_reader(SubRun, Field, ID, *, folder_path_='data', include_ephem=True):
    fname = f'OMPLDG_croin_cassan_{SubRun}_{Field}_{ID}.det.lc'
    if include_ephem:
        columns = ['Simulation_time', 'measured_relative_flux', 'measured_relative_flux_error', 'true_relative_flux',
                   'true_relative_flux_error', 'observatory_code', 'saturation_flag', 'best_single_lens_fit',
                   'parallax_shift_t', 'parallax_shift_u', 'BJD', 'source_x', 'source_y', 'lens1_x', 'lens1_y',
                   'lens2_x',
                   'lens2_y']
    else:
        columns = ['Simulation_time', 'measured_relative_flux', 'measured_relative_flux_error', 'true_relative_flux',
                   'true_relative_flux_error', 'observatory_code', 'saturation_flag', 'best_single_lens_fit',
                   'parallax_shift_t', 'parallax_shift_u', 'BJD', 'source_x', 'source_y', 'lens1_x', 'lens1_y',
                   'lens2_x', 'lens2_y']
    lightcurve_data_df = pd.read_csv(f'{folder_path_}/{fname}', names=columns, comment='#', sep='\s+')
    lightcurve_list = []
    for observatory_code in range(3):
        obs_data = lightcurve_data_df[lightcurve_data_df['observatory_code'] == observatory_code]
        lightcurve_list.append(obs_data[['Simulation_time', 'true_relative_flux', 'true_relative_flux_error']])
    return lightcurve_list


def whole_columns_lightcurve_reader(SubRun, Field, ID, *, folder_path_='data', include_ephem=True):
    """
    This function reads the lightcurve data file and returns a pandas dataframe with the BJD Magnitude Error columns
    :param folder_path_:
    :param data_challenge_lc_number_:
    :param filter_:
    :return: lightcurve_data_df
    """
    fname = f'OMPLDG_croin_cassan_{SubRun}_{Field}_{ID}.det.lc'

    with open(f'{folder_path_}/{fname}', 'r') as f:
        fs = f.readline()
        fs = np.array(fs.split(' ')[1:4]).astype(float)
        f.readline()
        f.readline()
        m_source = f.readline()
        m_source = np.array(m_source.split(' ')[1:4]).astype(float)
    if include_ephem:
        columns = ['Simulation_time', 'measured_relative_flux', 'measured_relative_flux_error', 'true_relative_flux',
                   'true_relative_flux_error', 'observatory_code', 'saturation_flag', 'best_single_lens_fit',
                   'parallax_shift_t', 'parallax_shift_u', 'BJD', 'source_x', 'source_y', 'lens1_x', 'lens1_y',
                   'lens2_x', 'lens2_y', 'SAT_X', 'SAT_Y', 'SAT_Z']
    else:
        columns = ['Simulation_time', 'measured_relative_flux', 'measured_relative_flux_error', 'true_relative_flux',
                   'true_relative_flux_error', 'observatory_code', 'saturation_flag', 'best_single_lens_fit',
                   'parallax_shift_t', 'parallax_shift_u', 'BJD', 'source_x', 'source_y', 'lens1_x', 'lens1_y',
                   'lens2_x', 'lens2_y']
    lightcurve_data_df = pd.read_csv(f'{folder_path_}/{fname}', names=columns, comment='#', sep='\s+')
    lightcurve_data_df['days'] = lightcurve_data_df['BJD'] - 2450000
    return lightcurve_data_df


if __name__ == '__main__':
    print('')
    # lc_type = what_type_of_lightcurve(2)
    # print(lc_type)
