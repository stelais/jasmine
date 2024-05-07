import pandas as pd


def binary_star_master_reader(filepath_='data/binary_star.csv'):
    """
    This function reads the binary star file and returns a pandas dataframe with the columns
    :param filepath_:
    :return: binary_star_df
    """
    binary_star_df = pd.read_csv(filepath_)
    return binary_star_df


def bound_planet_master_reader(filepath_='data/bound_planet.csv'):
    """
    This function reads the bound planet file and returns a pandas dataframe with the columns
    :param filepath_:
    :return: bound_planet_df
    """
    bound_planet_df = pd.read_csv(filepath_)
    return bound_planet_df


def cataclysmic_variables_master_reader(filepath_='data/cataclysmic_variables.csv'):
    """
    This function reads the cataclysmic variables file and returns a pandas dataframe with the columns
    :param filepath_:
    :return: cataclysmic_variables_df
    """
    cataclysmic_variables_df = pd.read_csv(filepath_)
    return cataclysmic_variables_df


def single_lens_master_reader(filepath_='data/single_lens.csv'):
    """
    This function reads the single lens file and returns a pandas dataframe with the columns
    :param filepath_:
    :return: single_lens_df
    """
    single_lens_df = pd.read_csv(filepath_)
    return single_lens_df


def what_type_of_lightcurve(data_challenge_lc_number_, *,
                            binary_star_path_='data/binary_star.csv',
                            bound_planet_path_='data/bound_planet.csv',
                            cataclysmic_variables_path_='data/cataclysmic_variables.csv',
                            single_lens_path_='data/single_lens.csv'):
    """
    This function reads the master files and checks in which file the lightcurve is.
    :param single_lens_path_:
    :param cataclysmic_variables_path_:
    :param bound_planet_path_:
    :param binary_star_path_:
    :param data_challenge_lc_number_:
    :return: master_name
    """
    binary_star_master_df = binary_star_master_reader(binary_star_path_)
    bound_planet_master_df = bound_planet_master_reader(bound_planet_path_)
    cataclysmic_variables_master_df = cataclysmic_variables_master_reader(cataclysmic_variables_path_)
    single_lens_master_df = single_lens_master_reader(single_lens_path_)
    masters_dict = {'binary_star': binary_star_master_df,
                    'bound_planet': bound_planet_master_df,
                    'cataclysmic_variables': cataclysmic_variables_master_df,
                    'single_lens': single_lens_master_df}

    for master_name, dataframe in masters_dict.items():
        LC_check = dataframe[dataframe['data_challenge_lc_number'] == data_challenge_lc_number_]['data_challenge_lc_number'].values
        if LC_check.size > 0:
            print(f'Ligthcurve {data_challenge_lc_number_} found on {master_name}.csv file.')
            return master_name

    raise ValueError(f'Lightcurve {data_challenge_lc_number_} not found in any of the master files.')


def lightcurve_data_reader(data_challenge_lc_number_, *, filter_='W149', folder_path_='data'):
    """
    This function reads the lightcurve data file and returns a pandas dataframe with the BJD Magnitude Error columns
    :param folder_path_:
    :param data_challenge_lc_number_:
    :param filter_:
    :return: lightcurve_data_df
    """
    lightcurve_data_df = pd.read_csv(f'{folder_path_}/ulwdc1_{data_challenge_lc_number_:03}_{filter_}.txt',
                                     names=['BJD', 'Magnitude', 'Error'], sep=' ')
    lightcurve_data_df['days'] = lightcurve_data_df['BJD'] - 2450000
    return lightcurve_data_df


if __name__ == '__main__':
    lc_type = what_type_of_lightcurve(2)
    print(lc_type)

