import numpy as np
import pandas as pd


def event_filter_by_errorbar_predicted_value_ratio(working_data_df, model_type, piE_type, precision_threshold=0.3):
    """
    Filter events based on the ratio of predicted errorbar to predicted value.
    :param working_data_df:
    :param model_type:
    :param piE_type:
    :param precision_threshold:
    :return:
    """
    approved_df = working_data_df[working_data_df[
                                      f'{model_type}_{piE_type}_absolute_predicted_errorbar_predicted_value_ratio'] < precision_threshold].copy()
    return approved_df


def event_filter_by_relative_accuracy(working_data_df, model_type, piE_type, accuracy_threshold=0.3):
    """
    Filter events based on the relative difference from the true value.
    :param working_data_df:
    :param model_type:
    :param piE_type:
    :param accuracy_threshold:
    :return:
    """
    # Metrics
    approved_df = working_data_df[
        working_data_df[f'{model_type}_{piE_type}_relative_difference_from_true_value'] < accuracy_threshold].copy()
    return approved_df


def combined_filtering_creteria(approved_precision_df, approved_accuracy_df):
    """
    Combine the two filters to get the final approved events.
    :param approved_precision_df:
    :param approved_accuracy_df:
    :return:
    """
    # Combine the two filters
    combined_approved_df = approved_precision_df[approved_precision_df.index.isin(approved_accuracy_df.index)].copy()
    return combined_approved_df


def main(statistics_data_path_, general_path_outputs_, precision_threshold=0.3, accuracy_threshold=0.3):
    """
    Main function to filter events based on the statistical analysis.
    :param statistics_data_path_:
    :param general_path_outputs_:
    :return:
    """
    working_data = pd.read_csv(
        statistics_data_path_,
        index_col=['event_name'])
    # piEE is the main goal
    piE_type = 'piEE'

    # LX GOOD FOR piEE
    model_type = 'LX'
    lx_approved_precision_df = event_filter_by_errorbar_predicted_value_ratio(working_data, model_type, piE_type,
                                                                              precision_threshold)
    lx_approved_precision_df.to_csv(f'{general_path_outputs_}/'
                                    'lx_precision_0.3_df.csv')
    lx_approved_accuracy_df = event_filter_by_relative_accuracy(working_data, model_type, piE_type,
                                                                accuracy_threshold)
    lx_approved_accuracy_df.to_csv(f'{general_path_outputs_}/'
                                   'lx_accuracy_0.3_df.csv')
    lx_combined_approved_df = combined_filtering_creteria(lx_approved_precision_df, lx_approved_accuracy_df)
    lx_combined_approved_df.to_csv(f'{general_path_outputs_}/'
                                   'lx_combined_0.3_df.csv')

    # LO GOOD FOR piEE
    model_type = 'LO'
    lo_approved_precision_df = event_filter_by_errorbar_predicted_value_ratio(working_data, model_type, piE_type,
                                                                              precision_threshold)
    lo_approved_precision_df.to_csv(f'{general_path_outputs_}/'
                                    'lo_precision_0.3_df.csv')
    lo_approved_accuracy_df = event_filter_by_relative_accuracy(working_data, model_type, piE_type,
                                                                accuracy_threshold)
    lo_approved_accuracy_df.to_csv(f'{general_path_outputs_}/'
                                   'lo_accuracy_0.3_df.csv')
    lo_combined_approved_df = combined_filtering_creteria(lo_approved_precision_df, lo_approved_accuracy_df)
    lo_combined_approved_df.to_csv(f'{general_path_outputs_}/'
                                   'lo_combined_0.3_df.csv')

    # LX and LO GOOD FOR piEE
    lx_and_lo_precision_df = lx_approved_precision_df[
        lx_approved_precision_df.index.isin(lo_approved_precision_df.index)].copy()
    lx_and_lo_precision_df.to_csv(f'{general_path_outputs_}/'
                                  'lx_and_lo_precision_0.3_df.csv')
    lx_and_lo_accuracy_df = lx_approved_accuracy_df[
        lx_approved_accuracy_df.index.isin(lo_approved_accuracy_df.index)].copy()
    lx_and_lo_accuracy_df.to_csv(f'{general_path_outputs_}/'
                                 'lx_and_lo_accuracy_0.3_df.csv')
    lx_and_lo_combined_df = lx_combined_approved_df[
        lx_combined_approved_df.index.isin(lo_combined_approved_df.index)].copy()
    lx_and_lo_combined_df.to_csv(f'{general_path_outputs_}/'
                                 'lx_and_lo_combined_0.3_df.csv')
    print()


if __name__ == '__main__':
    general_path_outputs = '/RTModel_plots/Notebook_statiscs_by_cuts'
    statistics_data_path = '/analysis/rtmodelv2.4_statistics.csv'
    main(statistics_data_path, general_path_outputs)
