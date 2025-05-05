"""
This wrap up the RTModel runs
Note you have to run classes_and_files_reader/RTModel_results_reader.py first.
"""
import os
import pandas as pd

def collect_chi2_data(base_path, output_file):
    """
    Collect chi2 data from the best LX and LO from each event folder and save to a summary CSV file.
    :param base_path:
    :param output_file:
    :return:
    """
    # Dictionary to store results
    results = []

    # Loop through event_* folders
    for event_folder in sorted(os.listdir(base_path)):
        event_path = os.path.join(base_path, event_folder)
        models_path = os.path.join(event_path, "Models", "chi2_top1_of_each_binary_lens_model.csv")

        # Check if the expected file exists
        if os.path.isdir(event_path) and os.path.exists(models_path):
            df = pd.read_csv(models_path)

            # Extract best LX model and chi2
            best_lx_row = df[df['model'].str.startswith('LX')]
            if not best_lx_row.empty:
                best_lx, best_lx_chi2 = best_lx_row.iloc[0]['model'], best_lx_row.iloc[0]['chi2']
            else:
                best_lx, best_lx_chi2 = None, None  # Handle missing LX case

            # Extract best LO model and chi2
            best_lo_row = df[df['model'].str.startswith('LO')]
            if not best_lo_row.empty:
                best_lo, best_lo_chi2 = best_lo_row.iloc[0]['model'], best_lo_row.iloc[0]['chi2']
            else:
                best_lo, best_lo_chi2 = None, None  # Handle missing LO case

            results.append([event_folder, best_lx, best_lx_chi2, best_lo, best_lo_chi2])

        # Create DataFrame and save
        output_df = pd.DataFrame(results, columns=["event_name", "best_LX", "best_LX_chi2", "best_LO", "best_LO_chi2"])
        output_df.to_csv(output_file, index=False)


def finding_best_rtmodel_fit_name(*, event_name_='event_0_90_1748',
                                  model_type_= 'LX',
                                  best_rtmodel_fit_table_path_='/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/sample_rtmodel_v2.4_LX_LO_summary.csv'):
    """
    Find the best fit model name from the table of best models. Basically reads the table above
    :param best_rtmodel_fit_table_path_:
    :param model_type_:
    :param event_name_:
    :return:
    """
    # Read the table
    best_rtmodel_fit_table = pd.read_csv(best_rtmodel_fit_table_path_)
    # Filter the table for the given event name and model type
    event_row = best_rtmodel_fit_table[(best_rtmodel_fit_table['event_name'] == event_name_)]
    # Get the best model name
    best_rtmodel_fit_name = event_row[f'best_{model_type_}'].values[0]
    return best_rtmodel_fit_name

if __name__ == '__main__':
    # Example usage

    ### TO SAVE
    # computer_path = "/discover/nobackup/sishitan/orbital_task"
    computer_path = "/"
    run_type = 'sample_rtmodel_v2.4'

    base_path = f"{computer_path}/RTModel_runs/{run_type}"
    output_file = f"{computer_path}/RTModel_runs/{run_type}_LX_LO_summary.csv"
    collect_chi2_data(base_path, output_file)
    print(f"Summary file saved at {output_file}")

    ### TO READ
    event_name = 'event_0_90_1748'
    best_rtmodel_fit_table_path = f'{computer_path}/RTModel_runs/{run_type}_LX_LO_summary.csv'
    model_name = finding_best_rtmodel_fit_name(event_name_=event_name,
                                    model_type_='LX',
                                    best_rtmodel_fit_table_path_=best_rtmodel_fit_table_path)
    print(f"Best LX model for {event_name}: {model_name}")