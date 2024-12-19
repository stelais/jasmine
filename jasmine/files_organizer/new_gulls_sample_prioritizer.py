import pandas as pd


def df_filter(df_, property_and_intervals):
    # Filter rows based on the intervals for tha property
    # e.g. ['Lens_Mass', 0.2, 0.5]
    filtered_df = df_[(df_[property_and_intervals[0]] >= property_and_intervals[1]) & (
                df_[property_and_intervals[0]] <= property_and_intervals[2])]
    return filtered_df


if __name__ == "__main__":
    general_path = '/Users/stela/Documents/Scripts/orbital_task/data/gulls_orbital_motion_extracted/'
    file_path = general_path + 'OMPLDG_croin_cassan.sample.csv'
    df = pd.read_csv(file_path)
    lens_mass = ['Lens_Mass', 0.2, 0.5]
    df_mass = df_filter(df, lens_mass)
    lens_distance = ['Lens_Dist', 1.0, 5.0]
    df_distance = df_filter(df_mass, lens_distance)
    df_distance.to_csv(general_path + 'most_common_by_lens_mass_distance.csv', index=False)
