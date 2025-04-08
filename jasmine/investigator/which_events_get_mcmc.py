import numpy as np
import pandas as pd

def event_filter_by_errorbar_predicted_value_ratio(working_data_df, model_type, piE_type, precision_threshold=0.3):
    approved_df = working_data_df[working_data_df[f'{model_type}_{piE_type}_absolute_predicted_errorbar_predicted_value_ratio'] < precision_threshold].copy()
    return approved_df

def event_filter_by_relative_accuracy(working_data_df, model_type, piE_type, accuracy_threshold=0.3):
    # Metrics
    approved_df = working_data_df[working_data_df[f'{model_type}_{piE_type}_relative_difference_from_true_value'] < accuracy_threshold].copy()
    return approved_df

def combined_filtering_creteria(approved_df1, approved_df2):
    # Combine the two filters
    combined_approved_df = approved_df1[approved_df1.index.isin(approved_df2.index)].copy()
    return combined_approved_df

def main(statistics_data_path_, general_path_outputs_):
    working_data = pd.read_csv(
    statistics_data_path_,
    index_col=['event_name'])
    # piEE is the main goal
    piE_type = 'piEE'

    # LX GOOD FOR piEE
    model_type = 'LX'
    lx_approved_df1 = event_filter_by_errorbar_predicted_value_ratio(working_data, model_type, piE_type)
    lx_approved_df2 = event_filter_by_relative_accuracy(working_data, model_type, piE_type)
    lx_approved_df2.to_csv(f'{general_path_outputs_}/'
                            'lx_accuracy_0.3_df.csv')
    lx_combined_approved_df = combined_filtering_creteria(lx_approved_df1, lx_approved_df2)
    lx_combined_approved_df.to_csv(f'{general_path_outputs_}/'
                                   'lx_combined_0.3_df.csv')

    # LO GOOD FOR piEE
    model_type = 'LO'
    lo_approved_df1 = event_filter_by_errorbar_predicted_value_ratio(working_data, model_type, piE_type)
    lo_approved_df2 = event_filter_by_relative_accuracy(working_data, model_type, piE_type)
    lo_approved_df2.to_csv(f'{general_path_outputs_}/'
                            'lo_accuracy_0.3_df.csv')
    lo_combined_approved_df = combined_filtering_creteria(lo_approved_df1, lo_approved_df2)
    lo_combined_approved_df.to_csv(f'{general_path_outputs_}/'
                                   'lo_combined_0.3_df.csv')

    # LX and LO GOOD FOR piEE
    lx_and_lo_combined_df = lx_combined_approved_df[lx_combined_approved_df.index.isin(lo_combined_approved_df.index)].copy()
    lx_and_lo_combined_df.to_csv(f'{general_path_outputs_}/'
                                 'lx_and_lo_combined_0.3_df.csv')
    print()

if __name__ == '__main__':
    general_path_outputs = '/Users/stela/Documents/Scripts/orbital_task/RTModel_plots/Notebook_statiscs_by_cuts'
    statistics_data_path = '/Users/stela/Documents/Scripts/orbital_task/jasmine/analysis/rtmodelv2.4_statistics.csv'
    main(statistics_data_path, general_path_outputs)
