"""
Called by RTModel_results_reader.py - check that
"""
from jasmine import ModelResults
from jasmine.classes_and_files_reader.new_gullsrges_lightcurve_cls import LightcurveEventGULLSRGES_NameBased

import pandas as pd


def get_summary_of_parallax_per_event(folder_path_, are_there_binary_solutions_):
    """
    This function creates a summary of piEN and piEE for the top 1 models + trues values for the given event
    """
    general_path_ = folder_path_.split('/RTModel_runs')[0] + '/data/gulls_orbital_motion_extracted'
    master_file_path = f'{general_path_}/OMPLDG_croin_cassan.sample.csv'
    data_folder_path = f'{general_path_}/OMPLDG_croin_cassan_sample'
    event_identification = folder_path_.split('/')[-1].split('vent_')[-1]
    lightcurve_name = f'OMPLDG_croin_cassan/OMPLDG_croin_cassan_{event_identification}.det.lc'
    event_name = folder_path_.split('/')[-1]
    subrun = event_name.split('_')[1]
    field = event_name.split('_')[2]
    event_id = event_name.split('_')[3]

    the_lightcurve_event = LightcurveEventGULLSRGES_NameBased(lightcurve_name, data_folder_path, master_file_path)
    true_pie = the_lightcurve_event.microlensing.microlensing_parallax_effect_piE
    true_piee = the_lightcurve_event.microlensing.microlensing_parallax_effect_piE_East
    true_pien = the_lightcurve_event.microlensing.microlensing_parallax_effect_piE_North

    data_to_be_saved = {'event_name': [event_name,],
                        'subrun': [subrun,],
                        'field': [field,],
                        'event_id': [event_id,],
                        'true_piE': [true_pie, ],
                        'true_piEN': [true_pien, ],
                        'true_piEE': [true_piee, ]}
    if are_there_binary_solutions_:
        top_1_of_each = pd.read_csv(folder_path_ + '/Models/chi2_top1_of_each_binary_lens_model.csv')
        for model_name in top_1_of_each['model']:
            model_type = model_name[0:2]
            if model_type == 'LO' or model_type == 'LX':
                model_path = folder_path_ + '/Models/' + model_name + '.txt'
                model_parameters = ModelResults(model_path).model_parameters
                data_to_be_saved[f'{model_type}_piEN'] = [model_parameters.piEN, ]
                data_to_be_saved[f'{model_type}_piEN_error'] = [model_parameters.piEN_error, ]
                data_to_be_saved[f'{model_type}_piEE'] = [model_parameters.piEE, ]
                data_to_be_saved[f'{model_type}_piEE_error'] = [model_parameters.piEE_error, ]
                data_to_be_saved[f'{model_type}_chi2'] = [model_parameters.chi2,]
    event_summary = pd.DataFrame(data_to_be_saved)
    event_summary.to_csv(folder_path_ + '/Models/event_summary_parallax.csv', index=False)
    return event_summary


def event_summary_parallax_wrapper(root_path_, list_of_events_, type_of_event_):
    """
    This function wraps up all the results for the given type of events, and produce a .csv with all of them
    """
    df_general = []
    for event_ in list_of_events_:
        df_for_one_event = pd.read_csv(f'{root_path_}/{event_}/Models/event_summary_parallax.csv')
        df_general.append(df_for_one_event)
    # Concatenate all the dataframes
    all_parallax_wrapper_df = pd.concat(df_general)
    all_parallax_wrapper_df.to_csv(f'{root_path_}/all_{type_of_event_}_parallax.csv', index=False)
    return all_parallax_wrapper_df


if __name__ == '__main__':
    list_of_events = ['event_0_1000_1445',
                      'event_0_42_2848',
                      'event_0_762_407']
    type_of_event = 'top10_piE'
    general_path = f'/discover/nobackup/sishitan/orbital_task/RTModel_runs/{type_of_event}'
    # general_path_outputs = '/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/top10_piE'

    for event in list_of_events:
        folder_path = f'{general_path}/{event}'
        get_summary_of_parallax_per_event(folder_path, True)
    event_summary_parallax_wrapper(general_path, list_of_events, type_of_event)
