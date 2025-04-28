"""
This code creates the directories and files for the MOA events rereduced by Prof. Ian Bond in the RTModel structure.
"""
import os
import pandas as pd
import merida.moa_rereduced_lightcurve_cls as merida_func

from jasmine.files_organizer.ra_and_dec_conversions import convert_degree_to_hms_dms


def creating_rtmodel_event_directory_with_files(*,
                                                name_prefix_='si01',
                                                folder_to_be_created_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                                                 'merida/RTModel_runs/ian',
                                                master_input_folder_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                                                'merida/data/lightcurves_from_ian',
                                                data_input_folder_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                                              'merida/data/lightcurves_from_ian',
                                                limb_darkening_=0.5633,
                                                alternative_lightcurve_version_=None):
    """
    Create ONE directory and files for the moa event following RTModel structure
    :param name_prefix_:
    :param folder_to_be_created_path_:
    :param master_input_folder_path_:
    :param data_input_folder_path_:
    :param limb_darkening_:
    :param alternative_lightcurve_version_: some lightcurves like 12 has an alternative version 'a'
    :return:
    """
    light_curve_master_file = f'{master_input_folder_path_}/stela_list.txt'
    creating_subdirectories(name_prefix_=name_prefix_,
                            folder_where_data_will_be_created_=folder_to_be_created_path_,
                            alternative_lightcurve_version_=alternative_lightcurve_version_)
    creating_data_files(name_prefix_=name_prefix_,
                        master_file_path_=light_curve_master_file,
                        data_input_folder_path_=data_input_folder_path_,
                        limb_darkening_=limb_darkening_,
                        folder_to_be_created_path_=folder_to_be_created_path_,
                        alternative_lightcurve_version_=alternative_lightcurve_version_)
    print(f'{name_prefix_} ALT:{alternative_lightcurve_version_} DONE')
    return None


def creating_subdirectories(*, name_prefix_,
                            folder_where_data_will_be_created_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                                           'merida/RTModel_runs/ian',
                            alternative_lightcurve_version_=None):
    """
    Create directories for the moa event following RTModel structure
    :param name_prefix_:
    :param folder_where_data_will_be_created_:
    :param alternative_lightcurve_version_: some lightcurves like 12 has an alternative version 'a'
    """
    if alternative_lightcurve_version_ is not None:
        the_name_prefix = name_prefix_ + alternative_lightcurve_version_
    else:
        the_name_prefix = name_prefix_
    event_folder = f'{folder_where_data_will_be_created_}/{the_name_prefix}'
    data_folder = f'{event_folder}/Data'

    if os.path.isdir(event_folder):
        raise FileExistsError(f'Directory {event_folder} already exists!')
    else:
        os.mkdir(event_folder)
        print(f'Folder {event_folder} created!')

    if os.path.isdir(data_folder):
        raise FileExistsError(f'Directory {data_folder} already exists!')
    else:
        os.mkdir(data_folder)
        print(f'Folder {data_folder} created!')


def creating_data_files(*, name_prefix_,
                        master_file_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                     'merida/data/lightcurves_from_ian/stela_list.txt',
                        data_input_folder_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                           'merida/RTModel_runs/ian',
                        folder_to_be_created_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                               'merida/RTModel_runs/ian_planet',
                        limb_darkening_=0.5633,
                        alternative_lightcurve_version_=None):
    """
    Create files for the moa event following RTModel structure
    :param name_prefix_:
    :param master_file_path_:
    :param data_input_folder_path_:
    """
    if alternative_lightcurve_version_ is not None:
        the_name_prefix = name_prefix_ + alternative_lightcurve_version_
    else:
        the_name_prefix = name_prefix_
    lc_number = int(name_prefix_[2:])
    event_folder = f'{folder_to_be_created_path_}/{the_name_prefix}'
    data_folder = f'{event_folder}/Data'
    if os.path.exists(f'{data_folder}/MOA.dat'):
        raise FileExistsError(
            f'File MOA.dat already exists in directory {data_folder}')
    else:
        moa_file = open(f'{data_folder}/MOA.dat', 'w')
        moa_file.write('# Mag err HJD-2450000\n')
        the_lightcurve = merida_func.MOAReReducedLightcurve(lightcurve_number_=lc_number,
                                                            data_folder_=data_input_folder_path_,
                                                            master_file_path_=master_file_path_,
                                                            alternative_lightcurve_version=alternative_lightcurve_version_)

        days, magnitudes, magnitudes_errors = the_lightcurve.get_days_magnitudes_errors()
        wanted_columns = pd.concat([magnitudes, magnitudes_errors, days], axis=1)
        wanted_columns.to_csv(moa_file, index=False, mode='a', header=False, sep=' ')
        print(f'File MOA.dat created in {data_folder}.')

        # Write LimbDarkening.txt file
        limb_file = open(f'{data_folder}/LimbDarkening.txt', 'w')
        limb_file.write(f'{limb_darkening_}')

        # Create the file with the event information
        event_ra_deg = the_lightcurve.master_row['RA'].values[0]
        event_dec_deg = the_lightcurve.master_row['DEC'].values[0]
        event_coordinates = convert_degree_to_hms_dms(event_ra_deg, event_dec_deg)
        if os.path.exists(f'{data_folder}/event.coordinates'):
            raise FileExistsError(f'File event.coordinates already exists in directory {data_folder}')
        else:
            with open(f'{data_folder}/event.coordinates', 'w') as f:
                f.write(str(event_coordinates)[1:-1].replace(',', '').replace("'", ''))
            print(f'File event.coordinates created in {data_folder}.')


if __name__ == '__main__':
    # EXAMPLE USAGE
    general_path = '/Users/stela/Documents/Scripts/ai_microlensing'
    # WHERE THE FOLDER WITH THE RTModel STRUCTURE WILL BE CREATED:
    folder_to_be_created = f'{general_path}/merida/RTModel_runs/ian'
    # WHERE THE MASTER FILE IS:
    master_input_folder = f'{general_path}/merida/data/lightcurves_from_ian'
    # WHERE THE DATA FILES ARE:
    data_input_folder = f'{general_path}/merida/data/lightcurves_from_ian'

    # # CREATE ONE
    # creating_one_directory_with_files(name_prefix_=1,
    #                                   folder_to_be_created_path_=folder_to_be_created,
    #                                   master_input_folder_path_=master_input_folder,
    #                                   data_input_folder_path_=data_input_folder,
    #                                   limb_darkening_=0.5633)
    for lc_number in range(11, 32):
        # if lc_number == 4 or lc_number == 8 or lc_number == 10:
        #     continue
        # CREATE ONE
        lc_name_prefix = f'si{lc_number:02}'
        creating_rtmodel_event_directory_with_files(name_prefix_=lc_name_prefix,
                                                    folder_to_be_created_path_=folder_to_be_created,
                                                    master_input_folder_path_=master_input_folder,
                                                    data_input_folder_path_=data_input_folder,
                                                    limb_darkening_=0.5633)
    creating_rtmodel_event_directory_with_files(name_prefix_='si12',
                                                folder_to_be_created_path_=folder_to_be_created,
                                                master_input_folder_path_=master_input_folder,
                                                data_input_folder_path_=data_input_folder,
                                                limb_darkening_=0.5633,
                                                alternative_lightcurve_version_='a')