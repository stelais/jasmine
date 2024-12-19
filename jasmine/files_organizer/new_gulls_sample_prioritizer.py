import pandas as pd


def df_filter(df_, property_and_intervals):
    # Filter rows based on the intervals for tha property
    # e.g. ['Lens_Mass', 0.2, 0.5]
    filtered_df = df_[(df_[property_and_intervals[0]] >= property_and_intervals[1]) & (
            df_[property_and_intervals[0]] <= property_and_intervals[2])]
    return filtered_df


def mass_and_distance_filter(general_path_, file_path_, is_saving=True):
    df = pd.read_csv(general_path_ + file_path_)
    lens_mass = ['Lens_Mass', 0.2, 0.5]
    df_mass = df_filter(df, lens_mass)
    lens_distance = ['Lens_Dist', 1.0, 5.0]
    df_distance = df_filter(df_mass, lens_distance)
    if is_saving:
        df_distance.to_csv(general_path_ + 'most_common_by_lens_mass_distance.csv', index=False)
    else:
        return df_distance


def make_sbatch_commands(project_path_, general_path_, file_path_):
    filtered_df = mass_and_distance_filter(project_path_ + general_path_, file_path_, is_saving=False)
    bash_path = f'{project_path_}bash_scripts'

    # Open the file for writing
    with open(f'{bash_path}/submission_common.sh', 'w') as f:
        # Iterate over the rows of the DataFrame
        for index, row in filtered_df.iterrows():
            # Extract the values for SubRun, Field, and EventID
            subrun = row['SubRun']
            field = row['Field']
            event_id = row['EventID']

            # Write the sbatch command to the file
            f.write(f'sbatch bash_scripts/rtmodel_new.sh {subrun} {field} {event_id}\n')

    print(f"Submission script saved to {bash_path}/submission_common.sh")


if __name__ == "__main__":
    project_path = '/Users/stela/Documents/Scripts/orbital_task/'
    general_path = 'data/gulls_orbital_motion_extracted/'
    file_path = 'OMPLDG_croin_cassan.sample.csv'
    # mass_and_distance_filter(project_path+general_path, file_path)
    make_sbatch_commands(project_path, general_path, file_path)
