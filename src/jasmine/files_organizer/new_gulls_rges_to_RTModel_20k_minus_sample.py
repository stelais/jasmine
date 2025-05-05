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


def getting_all_subruns_and_fields_folder_name(data_folder_path_):
    """
    Creates a .csv file to control the subruns and fields pairs
    :param data_folder_path_:
    :return:
    """
    subrun_field_pairs = []
    for directory in os.listdir(data_folder_path_):
        if directory.startswith("OMPLDG_croin_cassan_") and directory != "OMPLDG_croin_cassan_sample":
            parts = directory.split("_")
            subrun = parts[-3]
            field = parts[-2]
            subrun_field_pairs.append({"SubRun": subrun, "Field": field})
            print(f'SubRun is: {subrun} | Field is: {field}')

    df = pd.DataFrame(subrun_field_pairs)
    output_csv = f'{data_folder_path_}/subrun_field_pairs.csv'
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved as {output_csv}")


def does_this_lightcurve_exist(subrun_, field_, event_id_, data_folder_path_):
    directory = f"{data_folder_path_}/OMPLDG_croin_cassan_{subrun_}_{field_}_lc"
    lightcurve_path = f"{directory}/OMPLDG_croin_cassan_{subrun_}_{field_}_{event_id_}.det.lc"
    return os.path.exists(lightcurve_path)


def converting_mini_tables_in_one_general_csv(data_folder_path_):
    subrun_field_csv = f"{data_folder_path_}/subrun_field_pairs.csv"
    output_csv = f"{data_folder_path_}/10k_master.csv"

    subrun_field_df = pd.read_csv(subrun_field_csv)  # Loading all the SubRun-Field pairs
    combined_data = []  # df to be concatenated
    for _, row in subrun_field_df.iterrows():
        subrun = row["SubRun"]
        field = row["Field"]
        folder_name = f"OMPLDG_croin_cassan_{subrun}_{field}_lc"
        out_file_name = f"OMPLDG_croin_cassan_{subrun}_{field}.out"
        out_file_path = os.path.join(data_folder_path_, folder_name, out_file_name)
        if os.path.getsize(out_file_path) > 0:
            df = pd.read_csv(out_file_path, delim_whitespace=True)
            df['lcname'] = (f"OMPLDG_croin_cassan_{subrun}_{field}_lc/"
                            f"OMPLDG_croin_cassan_{subrun}_{field}_" + df['EventID'].astype(str) + ".det.lc")
            # EventID	SubRun	Field
            combined_data.append(df)  # Append the df to the combined list
        else:
            print(f"Skipped empty file: {out_file_path}")

    # Combine all dataframes into one
    combined_df = pd.concat(combined_data, ignore_index=True)
    df_filtrado = combined_df[combined_df.apply(lambda df_row: does_this_lightcurve_exist(df_row['SubRun'],
                                                                                          df_row['Field'],
                                                                                          df_row['EventID'],
                                                                                          data_folder_path_), axis=1)]
    df_filtrado.to_csv(output_csv, index=False)
    print(f"Master CSV file saved as {output_csv}")


def csv_file_without_sample(data_folder_path_):
    sample_df = pd.read_csv(f'{data_folder_path_}/OMPLDG_croin_cassan.sample.csv')
    master_df = pd.read_csv(f'{data_folder_path_}/20k_master.csv')
    columns_comparison = ['SubRun', 'Field', 'EventID']

    # Perform a merge with an indicator to identify differences
    merged = master_df.merge(sample_df, how='left', on=columns_comparison, indicator=True)

    # Filter rows that are only in the first DataFrame
    result_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Save or inspect the result
    result_df.to_csv(f"{data_folder_path_}/20k_master_minus_sample.csv", index=False)
    print(f"Filtered DataFrame has {len(result_df)} rows.")


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
    # pass dataframe index, subrun_, event_id_ to CreatingDirectories/Files
    light_curve_master_file = f'{master_input_folder_path_}/10k_master.csv'
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
    ld_par = [W149, Z087, K213]  # Order must match order in FilterToData.txt
    the_lightcurve = lc.LightcurveEventGULLSRGES_NameBased(lightcurve_name=lightcurve_name,
                                                           data_folder=data_input_folder_path_,
                                                           master_file_path=master_file_path_)
    lightcurve_data_df_list = the_lightcurve.lightcurve_data()
    for i in range(1, 4):
        filter_ = filters[i - 1]
        if os.path.exists(f'{data_to_be_created_folder_}/Roman{filter_}sat{i}.dat'):
            raise FileExistsError(
                f'File Roman{filter_}sat{i}.dat already exists in directory {data_to_be_created_folder_}')
        else:
            with open(f'{data_to_be_created_folder_}/Roman{filter_}sat{i}.dat', 'w') as f:
                f.write('# Mag err HJD-2450000\n')
            lightcurve_data_df_list[i - 1].to_csv(f'{data_to_be_created_folder_}/Roman{filter_}sat{i}.dat', sep=' ',
                                                  index=False, header=False, mode='a')
            print(f'File Roman{filter_}sat{i}.dat created in {data_to_be_created_folder_}.')
        i += 1
    # write LimbDarkening.txt file
    with open(data_to_be_created_folder_ + '/LimbDarkening.txt', 'w') as f:
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
    # NEW JAN 8
    folder_path = "/Users/stela/Documents/Scripts/orbital_task/data/gulls_orbital_motion_extracted"
    # getting_all_subruns_and_fields_folder_name(folder_path)
    # converting_mini_tables_in_one_general_csv(folder_path)
    csv_file_without_sample(folder_path)
    # TODO I AM HERE
    # general_path_outputs = '/discover/nobackup/sishitan/orbital_task'
    # # general_path_outputs = '/Users/stela/Documents/Scripts/orbital_task'
    # folder_to_be_created = f'{general_path_outputs}/RTModel_runs'
    # master_input_folder = f'{general_path_outputs}/data/gulls_orbital_motion_extracted'
    # data_input_folder = f'{general_path_outputs}/data/gulls_orbital_motion_extracted/OMPLDG_croin_cassan_sample'
    # creating_all_directories(folder_to_be_created_path_=folder_to_be_created,
    #                          master_input_folder_path_=master_input_folder,
    #                          data_input_folder_path_=data_input_folder)
