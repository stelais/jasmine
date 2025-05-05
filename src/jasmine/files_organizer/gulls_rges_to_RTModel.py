import os
import pandas as pd
import src.jasmine.classes_and_files_reader.gullsrges_lightcurve_cls as lc
from src.jasmine.constants.limb_darkening_parameters import *
from astropy.coordinates import SkyCoord
from astropy import units


def creating_all_directories(*,
                             folder_path_='/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events',
                             master_input_folder_path_='/Users/sishitan/Documents/Scripts/jasmine/data',
                             data_input_folder_path_='/Users/sishitan/Documents/Scripts/jasmine/data'):
    if os.path.isdir(folder_path_):
        print(f' Found directory {folder_path_}! Continuing.')
    else:
        print(f' Error: {folder_path_} not found.')
    ##########
    # get lc numbers from data master files created by the data_challenge prep module
    ##########
    # pass dataframe index, subrun, event_id to CreatingDirectories/Files
    light_curve_master = pd.read_csv(f'{master_input_folder_path_}/OMPLLD_croin_cassan_0.sample.csv')
    sample_index = light_curve_master.index
    subrun = light_curve_master['SubRun']
    event_id = light_curve_master['EventID']
    for index in range(sample_index.shape[0]):
        creating_directories(sample_index=sample_index[index],subrun=subrun[index],event_id=event_id[index],
                             folder_path_=folder_path_)
        creating_files(sample_index=sample_index[index],subrun=subrun[index],event_id=event_id[index],
                       output_folder_path_=folder_path_,
                       master_input_folder_path_=master_input_folder_path_,
                       data_input_folder_path_=data_input_folder_path_)

    return None


def creating_directories(*, sample_index,subrun,event_id,
                         folder_path_='/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events'):
    """
    Create directories for the gulls event following RTModel structure
    """
    event_folder_ = f'{folder_path_}/event_{sample_index:04}_{subrun:03}_{event_id:04}'
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


def creating_files(*, sample_index,subrun,event_id,
                   output_folder_path_='/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events',
                   master_input_folder_path_='/Users/sishitan/Documents/Scripts/jasmine/data',
                   data_input_folder_path_='/Users/sishitan/Documents/Scripts/jasmine/data'):
    """
    Create files for the data challenge event following RTModel structure
    """
    data_folder_ = f'{output_folder_path_}/event_{sample_index:04}_{subrun:03}_{event_id:04}/Data'
    filters = ['W146', 'Z087', 'K213']
    ld_par = [W149, Z087, K213] # Order must match order in FilterToData.txt
    the_lightcurve = lc.LightcurveEventGULLSRGES(sample_index, data_folder=master_input_folder_path_)
    lightcurve_data_df_list = the_lightcurve.lightcurve_data()
    for i in range(3):
        filter_ = filters[i]
        if os.path.exists(f'{data_folder_}/Roman{filter_}sat{i}.dat'):
            raise FileExistsError(f'File Roman{filter_}sat{i}.dat already exists in directory {data_folder_}')
        else:
            with open(f'{data_folder_}/Roman{filter_}sat{i}.dat', 'w') as f:
                f.write('# Mag err HJD-2450000\n')
            lightcurve_data_df_list[i].to_csv(f'{data_folder_}/Roman{filter_}sat{i}.dat', sep=' ',
                                          index=False, header=False, mode='a')
            print(f'File Roman{filter_}sat{i}.dat created in {data_folder_}.')
        i += 1
    # write LimbDarkening.txt file    
    with open(data_folder_+'/LimbDarkening.txt','w') as f:
        for mu in ld_par:
            f.write(f'{mu}\n')



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
    folder_to_be_created = '/Users/jmbrashe/VBBOrbital/GULLS/events'
    input_folder = '/Volumes/JonDrive/GULLS/gulls_orbital_motion/repsample'
    satellite_folder = '/Users/jmbrashe/VBBOrbital/GULLS/satellitedir'
    creating_all_directories(folder_path_=folder_to_be_created,
                             master_input_folder_path_=input_folder, data_input_folder_path_=input_folder)
    #ephtools.creating_ephemerides(satellite_folder_path_=satellite_folder, ephemerides_path_=input_folder)
