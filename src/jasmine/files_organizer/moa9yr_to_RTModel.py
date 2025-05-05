"""
This code creates the directories and files for the MOA9yr events released Dec2023 in the RTModel structure.
Note the fudge factor was set to 1.0, but should be adjusted accordingly
"""
import os
import pandas as pd
from merida.moa9yr_lightcurve_cls import MOA9yearLightcurve
from merida.metadata_cls import MOA_Lightcurve_Metadata

from jasmine.files_organizer.ra_and_dec_conversions import convert_degree_to_hms_dms


def creating_all_directories_with_files(*,
                                        folder_to_be_created_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                                                   'merida/RTModel_runs',
                                        master_input_folder_path_='/Users/stela/Documents/Scripts/ai_microlensing/merida/data',
                                        data_input_folder_path_='/Users/stela/Documents/Scripts/ai_microlensing/qusi_microlensing/data/'
                                                                'microlensing_2M',
                                        limb_darkening_=0.5633,
                                        offset_alternative_=False):
    """
    Create a list of directories and files for the moa9yr event following RTModel structure
    :param folder_to_be_created_path_:
    :param master_input_folder_path_:
    :param data_input_folder_path_:
    :param limb_darkening_:
    :param offset_alternative_:
    :return:
    """
    if os.path.isdir(folder_to_be_created_path_):
        print(f' Found directory {folder_to_be_created_path_}! Continuing.')
    else:
        print(f' Error: {folder_to_be_created_path_} not found.')
    ##########
    # get lc numbers from selected_metadata.csv files created by SIS with the NN outputs that are interesting
    ##########
    # pass dataframe index, subrun, event_id to CreatingDirectories/Files
    light_curve_master_file = f'{master_input_folder_path_}/selected_metadata.csv'
    light_curve_master = pd.read_csv(light_curve_master_file)
    for _, row in light_curve_master.iterrows():
        lc_filename = row['lightcurve_name']
        creating_directories(lightcurve_name_=lc_filename,
                             folder_where_data_will_be_created=folder_to_be_created_path_)
        creating_files(lightcurve_name_=lc_filename,
                       master_file_path_=light_curve_master_file,
                       data_input_folder_path_=data_input_folder_path_,
                       limb_darkening_=limb_darkening_,
                       offset_alternative_=offset_alternative_)
        print(f'{lc_filename} DONE')

    return None


def creating_one_directory_with_files(*,
                                      lc_filename_='gb9-R-8-5-27219',
                                      folder_to_be_created_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                                                 'merida/RTModel_runs',
                                      master_input_folder_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                                                'merida/data',
                                      data_input_folder_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                                              'qusi_microlensing/data/microlensing_2M',
                                      limb_darkening_=0.5633,
                                      offset_alternative_=False):
    """
    Create ONE directory and files for the moa9yr event following RTModel structure
    :param lc_filename_:
    :param folder_to_be_created_path_:
    :param master_input_folder_path_:
    :param data_input_folder_path_:
    :param limb_darkening_:
    :param offset_alternative_:
    :return:
    """
    light_curve_master_file = f'{master_input_folder_path_}/selected_metadata.csv'
    creating_directories(lightcurve_name_=lc_filename_,
                         folder_where_data_will_be_created=folder_to_be_created_path_)
    creating_files(lightcurve_name_=lc_filename_,
                   master_file_path_=light_curve_master_file,
                   data_input_folder_path_=data_input_folder_path_,
                   limb_darkening_=limb_darkening_,
                   offset_alternative_=offset_alternative_)
    print(f'{lc_filename_} DONE')

    return None


def creating_directories(*, lightcurve_name_,
                         folder_where_data_will_be_created='/Users/stela/Documents/Scripts/ai_microlensing/'
                                                           'merida/RTModel_runs'):
    """
    Create directories for the moa9yr event following RTModel structure
    """
    event_folder_flux = f'{folder_where_data_will_be_created}/{lightcurve_name_}_flux'
    event_folder_cor_flux = f'{folder_where_data_will_be_created}/{lightcurve_name_}_cor_flux'
    data_folder_flux_ = f'{event_folder_flux}/Data'
    data_folder_cor_flux_ = f'{event_folder_cor_flux}/Data'

    for event_folder in [event_folder_flux, event_folder_cor_flux]:
        if os.path.isdir(event_folder):
            raise FileExistsError(f'Directory {event_folder} already exists!')
        else:
            os.mkdir(event_folder)
            print(f'Folder {event_folder} created!')

    for data_folder in [data_folder_flux_, data_folder_cor_flux_]:
        if os.path.isdir(data_folder):
            raise FileExistsError(f'Directory {data_folder} already exists!')
        else:
            os.mkdir(data_folder)
            print(f'Folder {data_folder} created!')


def creating_files(*, lightcurve_name_,
                   general_folder_path_='/Users/stela/Documents/Scripts/ai_microlensing/'
                                        'merida/RTModel_runs',
                   master_file_path_='/Users/stela/Documents/Scripts/ai_microlensing/merida/data/selected_metadata.csv',
                   data_input_folder_path_='/Users/stela/Documents/Scripts/ai_microlensing/qusi_microlensing/data/'
                                           'microlensing_2M',
                   limb_darkening_=0.5633,
                   offset_alternative_=False):
    """
    Create files for the moa9yr event following RTModel structure
    """
    event_folder_flux = f'{general_folder_path_}/{lightcurve_name_}_flux'
    event_folder_cor_flux = f'{general_folder_path_}/{lightcurve_name_}_cor_flux'
    data_folder_flux_ = f'{event_folder_flux}/Data'
    data_folder_cor_flux_ = f'{event_folder_cor_flux}/Data'
    for data_to_be_created_folder_ in [data_folder_flux_, data_folder_cor_flux_]:
        if data_to_be_created_folder_.split('_')[-2] == 'cor':
            corrected = True
        else:
            corrected = False
        if os.path.exists(f'{data_to_be_created_folder_}/MOA.dat'):
            raise FileExistsError(
                f'File MOA.dat already exists in directory {data_to_be_created_folder_}')
        else:
            moa_file = open(f'{data_to_be_created_folder_}/MOA.dat', 'w')
            moa_file.write('# Mag err HJD-2450000\n')
            field = lightcurve_name_.split('-')[0]
            the_lightcurve = MOA9yearLightcurve(lightcurve_name_,
                                                f"{data_input_folder_path_}/{field}/")
            the_lightcurve.get_days_magnitudes_errors(fudge_factor=1.0, offset_alternative=offset_alternative_)
            if corrected:
                wanted_columns = the_lightcurve.lightcurve_dataframe[['cor_magnitudes', 'cor_magnitudes_err', 'HJD']].copy()
                wanted_columns.dropna(inplace=True)
            else:
                wanted_columns = the_lightcurve.lightcurve_dataframe[['magnitudes', 'magnitudes_err', 'HJD']]
            wanted_columns.to_csv(moa_file, index=False, mode='a', header=False, sep=' ')
            print(f'File MOA.dat created in {data_to_be_created_folder_}.')

        # Write LimbDarkening.txt file
        limb_file = open(f'{data_to_be_created_folder_}/LimbDarkening.txt', 'w')
        limb_file.write(f'{limb_darkening_}')

        # Create the file with the event information
        event_metadata = MOA_Lightcurve_Metadata(lightcurve_name_, path_to_metadata=master_file_path_)
        event_ra_deg = event_metadata.ra_j2000
        event_dec_deg = event_metadata.dec_j2000
        event_coordinates = convert_degree_to_hms_dms(event_ra_deg, event_dec_deg)
        if os.path.exists(f'{data_to_be_created_folder_}/event.coordinates'):
            raise FileExistsError(f'File event.coordinates already exists in directory {data_to_be_created_folder_}')
        else:
            with open(f'{data_to_be_created_folder_}/event.coordinates', 'w') as f:
                f.write(str(event_coordinates)[1:-1].replace(',', '').replace("'", ''))
            print(f'File event.coordinates created in {data_to_be_created_folder_}.')


if __name__ == '__main__':
    general_path = '/Users/stela/Documents/Scripts/ai_microlensing'
    folder_to_be_created = f'{general_path}/merida/RTModel_runs'
    master_input_folder = f'{general_path}/merida/data'
    data_input_folder = f'{general_path}/qusi_microlensing/data/microlensing_2M'
    # # CREATE ALL
    # creating_all_directories(folder_to_be_created_path_=folder_to_be_created,
    #                          master_input_folder_path_=master_input_folder,
    #                          data_input_folder_path_=data_input_folder)
    # CREATE ONE
    creating_one_directory_with_files(lc_filename_='gb9-R-8-5-27219',
                                      folder_to_be_created_path_=folder_to_be_created,
                                      master_input_folder_path_=master_input_folder,
                                      data_input_folder_path_=data_input_folder,
                                      limb_darkening_=0.5633,
                                      offset_alternative_=True)