import glob
import pandas as pd

from jasmine.classes_and_files_reader.datachallenge_lightcurve_cls import LightcurveEventDataChallenge
from jasmine.classes_and_files_reader.RTModel_results_cls import ModelResults


def chi2_getter(filepath):
    """
    This function reads the chi2 file and returns the chi2 value
    :param filepath:
    :return: chi2
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        chi2 = float(lines[0].split(' ')[-1])
    return chi2


def models_per_chi2_rank(folder_path):
    """
    This function rank the models per chi2 value
    """
    models = glob.glob(folder_path +'/Models/*txt')
    models_names = [model.replace(folder_path, '').replace('/Models/', '').replace('.txt', '') for model in models]
    chi2_values = []
    for model in models:
        chi2_values.append(chi2_getter(model))
    chi2_df = pd.DataFrame({'model': models_names, 'chi2': chi2_values})
    chi2_df = chi2_df.sort_values(by='chi2')
    chi2_df.to_csv(folder_path + '/Models/chi2_rank.csv', index=False)
    only_binary_lenses = chi2_df[chi2_df['model'].str.contains('L')]
    only_binary_lenses.to_csv(folder_path + '/Models/chi2_rank_binary_lenses.csv', index=False)
    only_LS = chi2_df[chi2_df['model'].str.contains('LS')]
    only_LX = chi2_df[chi2_df['model'].str.contains('LX')]
    only_LO = chi2_df[chi2_df['model'].str.contains('LO')]
    top_1_of_each = pd.concat([only_LS.iloc[[0]], only_LX.iloc[[0]], only_LO.iloc[[0]]])
    top_1_of_each.to_csv(folder_path + '/Models/chi2_top1_of_each_binary_lens_model.csv', index=False)


def get_summary_of_q_s_chi2_per_event(folder_path, type_of_event):
    """
    This function creates a summary of q and s for the top 1 models + trues values for the given event
    """
    top_1_of_each = pd.read_csv(folder_path + '/Models/chi2_top1_of_each_binary_lens_model.csv')
    data_challenge_lc_number = int(folder_path.split('_')[-1])
    master_path = folder_path.split('/datachallenge_events/')[0]
    true_values_path = f'{master_path}/data'
    the_lightcurve_event = LightcurveEventDataChallenge(data_challenge_lc_number, true_values_path)
    if type_of_event == 'bound_planet':
        true_q = the_lightcurve_event.planet.planet_mass_ratio
        true_s = the_lightcurve_event.planet.planet_separation
    elif type_of_event == 'binary_star':
        true_q = the_lightcurve_event.second_lens.mass_ratio
        true_s = the_lightcurve_event.second_lens.separation
    else:
        raise ValueError(f'type_of_event must be either bound_planet or binary_star. Given {type_of_event}')
    data_to_be_saved = {'data_challenge_lc_number': [data_challenge_lc_number,],
                        'true_q': [true_q,],
                        'true_s': [true_s,],}
    for model_name in top_1_of_each['model']:
        model_type = model_name[0:2]
        model_path = folder_path + '/Models/' + model_name + '.txt'
        model_parameters = ModelResults(model_path,
                                        data_challenge_lc_number=folder_path.split('_')[-1]).model_parameters
        data_to_be_saved[f'{model_type}_q'] = [model_parameters.mass_ratio,]
        data_to_be_saved[f'{model_type}_q_err'] = [model_parameters.mass_ratio_error,]
        data_to_be_saved[f'{model_type}_s'] = [model_parameters.separation,]
        data_to_be_saved[f'{model_type}_s_err'] = [model_parameters.separation_error,]
        data_to_be_saved[f'{model_type}_chi2'] = [model_parameters.chi2,]
    event_summary = pd.DataFrame(data_to_be_saved)
    event_summary.to_csv(folder_path + '/Models/event_summary_q_s.csv', index=False)
    return event_summary


def event_summary_q_s_wrapper(root_path_, list_of_events, type_of_event):
    """
    This function wraps up all the results for the given type of events, and produce a .csv with all of them
    """
    df_general = []
    for event_number_ in list_of_events:
        df_for_one_event = pd.read_csv(f'{root_path_}/event_{event_number_:03}/Models/event_summary_q_s.csv')
        df_general.append(df_for_one_event)
    # Concatenate all the dataframes
    all_q_s_wrapper_df = pd.concat(df_general)
    all_q_s_wrapper_df.to_csv(f'{root_path_}/all_{type_of_event}_q_s.csv', index=False)
    return all_q_s_wrapper_df


if __name__ == '__main__':
    # print(chi2_getter('/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events/event_004/Models/BS0004-1.txt'))
    list_of_bound_planet_events = [4, 8, 12, 25, 32, 40, 47, 50, 53, 62, 66, 69, 74, 78, 81, 92, 95, 99, 100, 103,
                                   107, 124, 128, 131, 139, 152, 163, 186, 193, 194, 199, 208, 214, 217, 218, 223,
                                   226, 227, 250, 253, 258, 267, 289]
    # event_number = list_of_bound_planet_events[0]
    # folder_path_ = (f'/Users/sishitan/Documents/Scripts/RTModel_project/'
    #                 f'RTModel/datachallenge_events/event_{event_number:03}')
    # models_per_chi2_rank(folder_path_)
    # get_summary_of_q_s_chi2_per_event(folder_path_)
    root_path = '/local/data/emussd1/greg_shared/rtmodel_effort/datachallenge/datachallenge_events/'
    # root_path = '/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events/'

    for event_number in list_of_bound_planet_events:
        folder_path_ = f'{root_path}event_{event_number:03}'
        models_per_chi2_rank(folder_path_)
        get_summary_of_q_s_chi2_per_event(folder_path_, 'bound_planet')

    event_summary_q_s_wrapper(root_path, list_of_bound_planet_events, 'bound_planet')
