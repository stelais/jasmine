"""
Run this code if you want a quick summary of chi2, q, s, and piE values for the best models of each event.
"""

import glob
import pandas as pd

import jasmine.classes_and_files_reader.mass_ratio_and_separation_getter as q_s_getter
import jasmine.classes_and_files_reader.parallax_getter as pie_getter


def chi2_getter(filepath_):
    """
    This function reads the chi2 file and returns the chi2 value
    :param filepath_:
    :return: chi2
    """
    with open(filepath_, 'r') as f:
        lines = f.readlines()
        chi2 = float(lines[0].split(' ')[-1])
    return chi2


def models_per_chi2_rank(folder_path_):
    """
    This function rank the models per chi2 value and create a chi2_top1_of_each_binary_lens_model.csv file
    """
    models = glob.glob(folder_path_ + '/Models/*txt')
    models_names = [model.replace(folder_path_, '').replace('/Models/', '').replace('.txt', '') for model in models]
    chi2_values = []
    for model in models:
        chi2_values.append(chi2_getter(model))
    chi2_df = pd.DataFrame({'model': models_names, 'chi2': chi2_values})
    chi2_df = chi2_df.sort_values(by='chi2')
    chi2_df.to_csv(folder_path_ + '/Models/chi2_rank.csv', index=False)
    only_binary_lenses = chi2_df[chi2_df['model'].str.contains('L')]
    only_binary_lenses.to_csv(folder_path_ + '/Models/chi2_rank_binary_lenses.csv', index=False)
    only_ls = chi2_df[chi2_df['model'].str.contains('LS')]
    only_lx = chi2_df[chi2_df['model'].str.contains('LX')]
    only_lo = chi2_df[chi2_df['model'].str.contains('LO')]
    # top_1_of_each = pd.concat([only_ls.iloc[[0]], only_lx.iloc[[0]], only_lo.iloc[[0]]])

    # Collect the top 1 row from each type if available
    dfs_to_concat = [df.iloc[[0]] for df in [only_ls, only_lx, only_lo] if not df.empty]

    if dfs_to_concat:
        top_1_of_each = pd.concat(dfs_to_concat, ignore_index=True)
        # Save the concatenated result to CSV
        top_1_of_each.to_csv(folder_path_ + '/Models/chi2_top1_of_each_binary_lens_model.csv', index=False)
        return True
    else:
        event_name = folder_path_.split('/')[-1]
        print(f"No valid models to concatenate. "
              f"Skipping creation of 'chi2_top1_of_each_binary_lens_model.csv' for {event_name}.")
        return False


def main(general_path_, list_of_events_, parallax=True, mass_ratio_and_separation=True, type_of_event_='top10_piE'):
    """
    This function is the main function that calls the other functions to get the summary of chi2, q, s, and piE values
    :param general_path_:
    :param list_of_events_:
    :param parallax:
    :param mass_ratio_and_separation:
    :param type_of_event_:
    :return:
    """
    for event in list_of_events_:
        folder_path_ = f'{general_path_}/{event}'
        are_there_binary_solutions = models_per_chi2_rank(folder_path_)
        if mass_ratio_and_separation:
            q_s_getter.get_summary_of_q_s_chi2_per_event(folder_path_, are_there_binary_solutions)
        if parallax:
            pie_getter.get_summary_of_parallax_per_event(folder_path_, are_there_binary_solutions)

    if mass_ratio_and_separation:
        q_s_getter.event_summary_q_s_wrapper(general_path_, list_of_events_, type_of_event_)
    if parallax:
        pie_getter.event_summary_parallax_wrapper(general_path_, list_of_events_, type_of_event_)


def list_of_events_from_sample_df(runs_path_):
    """
    This function reads the sample file and returns a list of events
    :param runs_path_:
    :return:
    """
    general_path_ = runs_path_.split('/RTModel_runs')[0] + '/data/gulls_orbital_motion_extracted'
    master_file_path = f'{general_path_}/OMPLDG_croin_cassan.sample.csv'
    sample_df = pd.read_csv(master_file_path)
    list_of_events_ = []
    for _, row in sample_df.iterrows():
        subrun = row['SubRun']
        event_id = row['EventID']
        field = row['Field']
        event_name = f'event_{subrun}_{field}_{event_id}'
        list_of_events_.append(event_name)
    return list_of_events_


if __name__ == '__main__':
    # general_path_outputs = '/discover/nobackup/sishitan/orbital_task/RTModel_runs/top10_piE'
    # list_of_events = ['event_0_762_407',
    #                   'event_0_876_1031',
    #                   'event_1_793_3191',
    #                   'event_0_42_270',
    #                   'event_0_922_1199',
    #                   'event_0_672_2455',
    #                   'event_0_992_224',
    #                   'event_0_42_2848',
    #                   'event_1_755_564',
    #                   'event_0_798_371']

    #
    # general_path_outputs = '/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/top10_piE'
    # list_of_events = ['event_0_42_270',]
    # # 'event_0_1000_1445',

    # type_of_event = 'sample'
    # type_of_event = '6_events_for_testing'
    # type_of_event = 'icgs_levmar'
    type_of_event = 'sample_rtmodel_v2.4'
    # runs_path = f'/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/{type_of_event}'
    runs_path = f'/discover/nobackup/sishitan/orbital_task/RTModel_runs/{type_of_event}'
    list_of_events = list_of_events_from_sample_df(runs_path)
    # list_of_events = ['event_0_128_2350',
    #                   'event_0_167_1273',
    #                   'event_0_672_793',
    #                   'event_0_715_873',
    #                   'event_0_874_19',
    #                   'event_0_952_2841']
    main(runs_path, list_of_events, parallax=True, mass_ratio_and_separation=True, type_of_event_=type_of_event)
