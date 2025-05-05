"""
This code creates the directories and files for the GULLS events launched in Nov2024 in the RTModel structure.
You also need to use
"""
import os
import pandas as pd
import jasmine.classes_and_files_reader.new_gullsrges_lightcurve_cls as lc
from jasmine.constants.limb_darkening_parameters import *
from astropy.coordinates import SkyCoord
from astropy import units


def creating_all_directories(*,
                             folder_to_be_created_path_='/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events',
                             master_input_folder_path_='/Users/sishitan/Documents/Scripts/jasmine/data',
                             data_input_folder_path_='/Users/sishitan/Documents/Scripts/jasmine/data'):
    if os.path.isdir(folder_to_be_created_path_):
        print(f' Found directory {folder_to_be_created_path_}! Continuing.')
    else:
        print(f' Error: {folder_to_be_created_path_} not found.')
    ##########
    # get lc numbers from data master files created by the data_challenge prep module
    ##########
    # pass dataframe index, subrun, event_id to CreatingDirectories/Files
    light_curve_master_file = f'{master_input_folder_path_}/OMPLDG_croin_cassan.sample.csv'
    light_curve_master = pd.read_csv(light_curve_master_file)
    for _, row in light_curve_master.iterrows():
        subrun = row['SubRun']
        event_id = row['EventID']
        lc_filename = row['lcname']
        field = row['Field']
        creating_directories(subrun=subrun, event_id=event_id, field=field,
                             general_folder_path_=folder_to_be_created_path_)
        creating_files(subrun=subrun, event_id=event_id, field=field,
                       lightcurve_name=lc_filename,
                       general_output_folder_path_=folder_to_be_created_path_,
                       master_file_path_=light_curve_master_file,
                       data_input_folder_path_=data_input_folder_path_)
        print()

    return None


def creating_directories(*, subrun, event_id, field,
                         general_folder_path_='/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events'):
    """
    Create directories for the gulls event following RTModel structure
    """
    event_folder_ = f'{general_folder_path_}/event_{subrun}_{field}_{event_id}'
    data_folder_ = f'{event_folder_}/Data'
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


def creating_files(*, subrun, event_id, field, lightcurve_name,
                   general_output_folder_path_='/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events',
                   master_file_path_='/Users/sishitan/Documents/Scripts/jasmine/data',
                   data_input_folder_path_='/Users/sishitan/Documenxts/Scripts/jasmine/data'):
    """
    Create files for the data challenge event following RTModel structure
    """
    data_to_be_created_folder_ = f'{general_output_folder_path_}/event_{subrun}_{field}_{event_id}/Data'
    filters = ['W146', 'Z087', 'K213']
    ld_par = [W149, Z087, K213] # Order must match order in FilterToData.txt
    the_lightcurve = lc.LightcurveEventGULLSRGES_NameBased(lightcurve_name=lightcurve_name,
                                                           data_folder=data_input_folder_path_,
                                                           master_file_path=master_file_path_)
    lightcurve_data_df_list = the_lightcurve.lightcurve_data()
    for i in range(1, 4):
        filter_ = filters[i-1]
        if os.path.exists(f'{data_to_be_created_folder_}/Roman{filter_}sat{i}.dat'):
            raise FileExistsError(f'File Roman{filter_}sat{i}.dat already exists in directory {data_to_be_created_folder_}')
        else:
            with open(f'{data_to_be_created_folder_}/Roman{filter_}sat{i}.dat', 'w') as f:
                f.write('# Mag err HJD-2450000\n')
            lightcurve_data_df_list[i-1].to_csv(f'{data_to_be_created_folder_}/Roman{filter_}sat{i}.dat', sep=' ',
                                                index=False, header=False, mode='a')
            print(f'File Roman{filter_}sat{i}.dat created in {data_to_be_created_folder_}.')
        i += 1
    # write LimbDarkening.txt file
    with open(data_to_be_created_folder_+'/LimbDarkening.txt','w') as f:
        for mu in ld_par:
            f.write(f'{mu}\n')


    # Create the file with the event information
    event_ra = the_lightcurve.event_info.event_right_ascension__deg
    event_dec = the_lightcurve.event_info.event_declination__deg
    event_coordinates = SkyCoord(event_ra, event_dec, unit=(units.deg, units.deg), obstime='J2000')
    rtmodel_event_coordinates = (event_coordinates.to_string('hmsdms').replace('h', ':').replace('m', ':')
                                 .replace('d', ':').replace('s', ''))
    if os.path.exists(f'{data_to_be_created_folder_}/event.coordinates'):
        raise FileExistsError(f'File event.coordinates already exists in directory {data_to_be_created_folder_}')
    else:
        with open(f'{data_to_be_created_folder_}/event.coordinates', 'w') as f:
            f.write(f'{rtmodel_event_coordinates}')
        print(f'File event.coordinates created in {data_to_be_created_folder_}.')


if __name__ == '__main__':
    general_path = '/discover/nobackup/sishitan/orbital_task'
    # general_path_outputs = '/Users/stela/Documents/Scripts/orbital_task'
    folder_to_be_created = f'{general_path}/RTModel_runs/sample_rtmodel_v2.4'
    master_input_folder = f'{general_path}/data/gulls_orbital_motion_extracted'
    data_input_folder = f'{general_path}/data/gulls_orbital_motion_extracted/OMPLDG_croin_cassan_sample'
    creating_all_directories(folder_to_be_created_path_=folder_to_be_created,
                             master_input_folder_path_=master_input_folder,
                             data_input_folder_path_=data_input_folder)
