from jasmine import ModelResults
import pandas as pd


def get_summary_of_parallax_per_event(folder_path):
    """
    This function creates a summary of piN and piE for the top 1 models + trues values for the given event
    """
    top_1_of_each = pd.read_csv(folder_path + '/Models/chi2_top1_of_each_binary_lens_model.csv')
    data_challenge_lc_number = int(folder_path.split('_')[-1])
    data_to_be_saved = {'data_challenge_lc_number': [data_challenge_lc_number,]}
    for model_name in top_1_of_each['model']:
        model_type = model_name[0:2]
        if model_type == 'LO' or model_type == 'LX':
            model_path = folder_path + '/Models/' + model_name + '.txt'
            model_parameters = ModelResults(model_path,
                                            data_challenge_lc_number=folder_path.split('_')[-1]).model_parameters
            data_to_be_saved[f'{model_type}_piN'] = [model_parameters.piN,]
            data_to_be_saved[f'{model_type}_piN_error'] = [model_parameters.piN_error,]
            data_to_be_saved[f'{model_type}_piE'] = [model_parameters.piE,]
            data_to_be_saved[f'{model_type}_piE_error'] = [model_parameters.piE_error,]
            data_to_be_saved[f'{model_type}_chi2'] = [model_parameters.chi2,]
    event_summary = pd.DataFrame(data_to_be_saved)
    event_summary.to_csv(folder_path + '/Models/event_summary_parallax.csv', index=False)
    return event_summary


def event_summary_parallax_wrapper(root_path_, list_of_events, type_of_event):
    """
    This function wraps up all the results for the given type of events, and produce a .csv with all of them
    """
    df_general = []
    for event_number_ in list_of_events:
        df_for_one_event = pd.read_csv(f'{root_path_}/event_{event_number_:03}/Models/event_summary_parallax.csv')
        df_general.append(df_for_one_event)
    # Concatenate all the dataframes
    all_q_s_wrapper_df = pd.concat(df_general)
    all_q_s_wrapper_df.to_csv(f'{root_path_}/all_{type_of_event}_parallax.csv', index=False)
    return all_q_s_wrapper_df


if __name__ == '__main__':
    list_of_bound_planet_events = [4, 8, 12, 25, 32, 40, 47, 50, 53, 62, 66, 69, 74, 78, 81, 92, 95, 99, 100, 103,
                                   107, 124, 128, 131, 139, 152, 163, 186, 193, 194, 199, 208, 214, 217, 218, 223,
                                   226, 227, 250, 253, 258, 267, 289]

    root_path = '/local/data/emussd1/greg_shared/rtmodel_effort/datachallenge/datachallenge_events/'

    for event_number in list_of_bound_planet_events:
        folder_path_ = f'{root_path}event_{event_number:03}'
        get_summary_of_parallax_per_event(folder_path_, 'bound_planet')

    event_summary_parallax_wrapper(root_path, list_of_bound_planet_events, 'bound_planet')