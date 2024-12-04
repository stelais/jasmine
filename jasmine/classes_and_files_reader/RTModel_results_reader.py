import glob
import pandas as pd

import jasmine.classes_and_files_reader.mass_ratio_and_separation_getter as q_s_getter
import jasmine.classes_and_files_reader.parallax_getter as pie_getter


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
    models = glob.glob(folder_path + '/Models/*txt')
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


def main(general_path, list_of_events, parallax=True, mass_ratio_and_separation=True, type_of_event='top10_piE'):
    for event in list_of_events:
        folder_path_ = f'{general_path}/{event}'
        models_per_chi2_rank(folder_path_)

        if mass_ratio_and_separation:
            q_s_getter.get_summary_of_q_s_chi2_per_event(folder_path_)
        if parallax:
            pie_getter.get_summary_of_parallax_per_event(folder_path_)

    if mass_ratio_and_separation:
        q_s_getter.event_summary_q_s_wrapper(general_path, list_of_events, type_of_event)
    if parallax:
        pie_getter.event_summary_parallax_wrapper(general_path, list_of_events, type_of_event)



if __name__ == '__main__':
    # # print(chi2_getter('/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events/event_004/Models/BS0004-1.txt'))
    # list_of_bound_planet_events = [4, 8, 12, 25, 32, 40, 47, 50, 53, 62, 66, 69, 74, 78, 81, 92, 95, 99, 100, 103,
    #                                107, 124, 128, 131, 139, 152, 163, 186, 193, 194, 199, 208, 214, 217, 218, 223,
    #                                226, 227, 250, 253, 258, 267, 289]
    # # event_number = list_of_bound_planet_events[0]
    # # folder_to_be_created_path_ = (f'/Users/sishitan/Documents/Scripts/RTModel_project/'
    # #                 f'RTModel/datachallenge_events/event_{event_number:03}')
    # # models_per_chi2_rank(folder_to_be_created_path_)
    # # get_summary_of_q_s_chi2_per_event(folder_to_be_created_path_)
    # root_path = '/local/data/emussd1/greg_shared/rtmodel_effort/datachallenge/datachallenge_events/'
    # # root_path = '/Users/sishitan/Documents/Scripts/RTModel_project/RTModel/datachallenge_events/'

    general_path = '/discover/nobackup/sishitan/orbital_task/RTModel_runs/top10_piE'
    # general_path = '/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/top10_piE'
    list_of_events = ['event_0_1000_1445',
                      'event_0_42_2848',
                      'event_0_762_407',
                      'event_0_876_1031',
                      'event_0_992_224',
                      'event_1_793_3191',
                      'event_0_42_270',
                      'event_0_672_2455',
                      'event_0_798_371',
                      'event_0_922_1199',
                      'event_1_755_564']
