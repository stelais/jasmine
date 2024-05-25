import os
import pandas as pd
import numpy as np
import jasmine.classes_and_files_reader.datachallenge_lightcurve_cls as lc
import jasmine.files_organizer.RTModel_ephemerides_tools as ephtools
from astropy.coordinates import SkyCoord
from astropy import units


def creating_all_directories(*, event_type_,
                             folder_path_='/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events',
                             master_input_folder_path_='/Users/sishitan/Documents/Scripts/jasmine/data',
                             data_input_folder_path_='/Users/sishitan/Documents/Scripts/jasmine/data'):
    # event_type  should be a list object containing the types of objects in the data challenge you want observed.
    # Choose between 'binary' 'planet' 'CV','single', or you may type 'all'.
    # valid examples: ['binary','planet'], ['all']
    if os.path.isdir(folder_path_):
        print(f' Found directory {folder_path_}! Continuing.')
    else:
        os.mkdir(folder_path_)
        print(f'Folder {folder_path_} created!')
    ##########
    # Check contents of event_type, if necessary prompt for correct inputs
    ##########
    valid_names = ['binary', 'planet', 'cv', 'single', 'all']
    i = 0
    while i < len(event_type_):
        if event_type_[i].lower() not in valid_names:
            name = input(
                f' Event type {event_type_[i]} not recognized.\nPlease re-enter one of these names {valid_names}'
                f' or type exit to close the program.\n')
            if name.lower() == 'exit':
                return None
            else:
                event_type_[i] = name.lower()
                i -= 1
        else:
            event_type_[i] = event_type_[i].lower()
            if event_type_[i] == 'all' and len(event_type_) > 1:
                event_type_ = ['all']
                break
        i += 1
    if 'all' in event_type_:
        event_type_ = valid_names[0:4]  #
    event_type_ = np.unique(event_type_)
    ##########
    # get lc numbers from data master files created by the data_challenge prep module
    ##########
    csv_paths_ = {'binary': f'{data_input_folder_path_}/binary_star.csv',
                  'planet': f'{data_input_folder_path_}/bound_planet.csv',
                  'cv': f'{data_input_folder_path_}/cataclysmic_variables.csv',
                  'single': f'{data_input_folder_path_}/single_lens.csv'}
    data_challenge_lc_number_list = []
    for ev_type in event_type_:
        event_type_master = pd.read_csv(csv_paths_[ev_type])
        data_challenge_lc_number_list.append(event_type_master['data_challenge_lc_number'].values)
    data_challenge_lc_number_list = np.concatenate(data_challenge_lc_number_list)
    print(data_challenge_lc_number_list)
    for dc_lc_num in data_challenge_lc_number_list:
        creating_directories(data_challenge_lc_number_=dc_lc_num,
                             folder_path_=folder_path_)
        creating_files(data_challenge_lc_number_=dc_lc_num,
                       output_folder_path_=folder_path_,
                       master_input_folder_path_=master_input_folder_path_,
                       data_input_folder_path_=data_input_folder_path_)

    return None


def creating_directories(*, data_challenge_lc_number_,
                         folder_path_='/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events'):
    """
    Create directories for the data challenge event following RTModel structure
    """
    event_folder_ = f'{folder_path_}/event_{data_challenge_lc_number_:03}'
    data_folder_ = f'{folder_path_}/event_{data_challenge_lc_number_:03}/Data'
    if os.path.isdir(event_folder_):
        raise FileExistsError(f'Directory {event_folder_} already exists!')
    else:
        os.mkdir(event_folder_)
        print(f'Folder {event_folder_} created!')
    if os.path.isdir(data_folder_):
        raise FileExistsError(f'Directory {data_folder_} already exists!')
    else:
        os.mkdir(data_folder_)
        print(f'Folder {data_folder_} created!')


def creating_files(*, data_challenge_lc_number_,
                   output_folder_path_='/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events',
                   master_input_folder_path_='/Users/sishitan/Documents/Scripts/jasmine/data',
                   data_input_folder_path_='/Users/sishitan/Documents/Scripts/jasmine/data'):
    """
    Create files for the data challenge event following RTModel structure
    """
    data_folder_ = f'{output_folder_path_}/event_{data_challenge_lc_number_:03}/Data'
    filters = ['W149', 'Z087']
    the_lightcurve = lc.LightcurveEventDataChallenge(data_challenge_lc_number_, data_folder=master_input_folder_path_)
    i = 1
    for filter_ in filters:
        lightcurve_data_df = the_lightcurve.lightcurve_data(filter_=filter_, folder_path_=data_input_folder_path_)
        lightcurve_data_df_out = lightcurve_data_df[['Magnitude', 'Error', 'days']]
        if os.path.exists(f'{data_folder_}/Roman{filter_}sat{i}.dat'):
            raise FileExistsError(f'File Roman{filter_}sat{i}.dat already exists in directory {data_folder_}')
        else:
            with open(f'{data_folder_}/Roman{filter_}sat{i}.dat', 'w') as f:
                f.write('# Mag err HJD-2450000\n')
            lightcurve_data_df_out.to_csv(f'{data_folder_}/Roman{filter_}sat{i}.dat', sep=' ',
                                          index=False, header=False, mode='a')
            print(f'File Roman{filter_}sat{i}.dat created in {data_folder_}.')
        i += 1

    # Create the file with the event information
    event_ra = the_lightcurve.event_info.event_right_ascension__deg
    event_dec = the_lightcurve.event_info.event_declination__deg
    event_coordinates = SkyCoord(event_ra, event_dec, unit=(units.deg, units.deg), obstime='J2000')
    rtmodel_event_coordinates = (event_coordinates.to_string('hmsdms').replace('h', ':').replace('m', ':')
                                 .replace('d', ':').replace('s', ''))
    if os.path.exists(f'{data_folder_}/event.coordinates'):
        raise FileExistsError(f'File event.coordinates already exists in directory {data_folder_}')
    else:
        with open(f'{data_folder_}/event.coordinates', 'w') as f:
            f.write(f'{rtmodel_event_coordinates}')
        print(f'File event.coordinates created in {data_folder_}.')


if __name__ == '__main__':
    folder_to_be_created = '/Users/jmbrashe/VBBOrbital/event_test'
    input_folder = '/Users/jmbrashe/VBBOrbital/data'
    satellite_folder = '/Users/jmbrashe/VBBOrbital/satellitedir'
    creating_all_directories(event_type_=['binary', 'planet'], folder_path_=folder_to_be_created,
                             master_input_folder_path_=input_folder, data_input_folder_path_=input_folder)
    ephtools.creating_ephemerides(satellite_folder_path_=satellite_folder, ephemerides_path_=input_folder)
