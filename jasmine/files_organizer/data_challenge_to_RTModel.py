import os
import jasmine.files_organizer.lightcurve_cls as lc
from astropy.coordinates import SkyCoord
from astropy import units


def creating_directories(*, data_challenge_lc_number_,
                         folder_path_='/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events'):
    """
    Create directories for the data challenge event following RTModel structure
    """
    event_folder_ = f'{folder_path_}/event_{data_challenge_lc_number_:03}'
    data_folder_ = f'{folder_path_}/event_{data_challenge_lc_number_:03}/Data'
    try:
        os.mkdir(event_folder_)
        print(f'Folder {event_folder_} created!')
    except FileExistsError:
        print(f'Directory {event_folder_} already exists!')
    try:
        os.mkdir(data_folder_)
        print(f'Folder {data_folder_} created!')
    except FileExistsError:
        print(f'Directory {data_folder_} already exists!')


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
    for filter_ in filters:
        lightcurve_data_df = the_lightcurve.lightcurve_data(filter_=filter_, folder_path_=data_input_folder_path_)
        lightcurve_data_df_out = lightcurve_data_df[['Magnitude', 'Error', 'days']]
        with open(f'{data_folder_}/Roman{filter_}.dat', 'w') as f:
            f.write('# Mag err HJD-2450000\n')
        lightcurve_data_df_out.to_csv(f'{data_folder_}/Roman{filter_}.dat', sep=' ',
                                      index=False, header=False, mode='a')
        print(f'File Roman{filter_}.dat created in {data_folder_}.')

    # Create the file with the event information
    event_ra = the_lightcurve.event_info.event_right_ascension__deg
    event_dec = the_lightcurve.event_info.event_declination__deg
    event_coordinates = SkyCoord(event_ra, event_dec, unit=(units.deg, units.deg), obstime='J2000')
    rtmodel_event_coordinates = (event_coordinates.to_string('hmsdms').replace('h', ':').replace('m', ':')
                                 .replace('d', ':').replace('s', ''))
    with open(f'{data_folder_}/event.coordinates', 'w') as f:
        f.write(f'{rtmodel_event_coordinates}')
    print(f'File event.coordinates created in {data_folder_}.')


if __name__ == '__main__':
    data_challenge_lc_number = 4
    folder_to_be_created = '/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events'
    input_folder = '/Users/sishitan/Documents/Scripts/jasmine/data'
    creating_directories(data_challenge_lc_number_=data_challenge_lc_number,
                         folder_path_=folder_to_be_created)
    creating_files(data_challenge_lc_number_=data_challenge_lc_number,
                   output_folder_path_=folder_to_be_created,
                   master_input_folder_path_=input_folder,
                   data_input_folder_path_=input_folder)



