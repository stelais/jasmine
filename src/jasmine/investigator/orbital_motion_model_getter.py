import pandas as pd
from jasmine import ModelResults


def get_LO_names(*, summary_path_folder_, run_type_, event_unique_id_):
    summary_df = pd.read_csv(f'{summary_path_folder_}/{run_type_}_LX_LO_summary.csv')
    LO_name = summary_df[summary_df['event_name'] == f'event_{event_unique_id_}']['best_LO'].values[0]
    return LO_name


def LO_model_retriever_per_event_and_runtype(*, general_path_, run_type_, event_unique_id_):
    summary_path_folder = f'{general_path_}/RTModel_runs'
    LO_name = get_LO_names(summary_path_folder_=summary_path_folder,
                           run_type_=run_type_,
                           event_unique_id_=event_unique_id_)
    LO_model = ModelResults(f'{summary_path_folder}/{run_type_}/event_{event_unique_id_}/Models/{LO_name}.txt')
    # Convert model parameters (dataclass) to dictionary
    lo_data = vars(LO_model.model_parameters)
    # Convert to DataFrame
    lo_df = pd.DataFrame(lo_data, index=pd.Index([f'event_{event_unique_id_}'], name='event_name'))
    return lo_df


def LO_model_orbital_motion_parameters_to_csv(*, run_type_, general_path_):
    all_dataframes = []

    all_representative_sample_df = pd.read_csv(f'{general_path_}/data/gulls_orbital_motion_extracted/'
                                               f'OMPLDG_croin_cassan.sample.csv')
    all_representative_sample_df['event_unique_id'] = (all_representative_sample_df['SubRun'].astype(str) + "_" +
                                                       all_representative_sample_df['Field'].astype(str) + "_" +
                                                       all_representative_sample_df['EventID'].astype(str))

    for event_unique_id_ in all_representative_sample_df['event_unique_id']:
        try:
            lo_df = LO_model_retriever_per_event_and_runtype(general_path_=general_path_,
                                                             run_type_=run_type_,
                                                             event_unique_id_=event_unique_id_)
            all_dataframes.append(lo_df)
        except Exception as e:
            print(f"Skipping run_type '{run_type_}' for event '{event_unique_id_}': {e}")

    # Save to CSV
    combined_df = pd.concat(all_dataframes)
    lo_csv_path = f'{general_path_}/RTModel_runs/{run_type_}_LO_model_parameters.csv'
    combined_df.to_csv(lo_csv_path)



if __name__ == '__main__':
    run_type = 'sample_rtmodel_v2.4'
    # general_path = '/Users/stela/Documents/Scripts/orbital_task'
    general_path = '/discover/nobackup/sishitan/orbital_task'

    LO_model_orbital_motion_parameters_to_csv(general_path_=general_path,
                                              run_type_=run_type)

