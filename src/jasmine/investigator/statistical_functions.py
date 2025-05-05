import numpy as np
from jasmine.investigator.astrophysics_formulas import absolute_pi_E_calculator

def relative_difference_from_true_value_percentage_calculator(true_value, approximated_value):
    percent_error = 100 * np.absolute((true_value - approximated_value) / true_value)
    return percent_error


def relative_difference_from_true_value_calculator(true_value, approximated_value):
    relative_difference = np.absolute((true_value - approximated_value) / true_value)
    return relative_difference

def standardized_residual_calculator(true_value, predicted_value, predicted_errorbar):
    standardized_residual = (true_value - predicted_value) / predicted_errorbar
    return standardized_residual

def absolute_standardized_residual_calculator(true_value, predicted_value, predicted_errorbar):
    absolute_standardized_residual = np.absolute(standardized_residual_calculator(true_value, predicted_value, predicted_errorbar))
    return absolute_standardized_residual

def predicted_value_predicted_errorbar_ratio_calculator(predicted_value, predicted_errorbar):
    ratio = predicted_value / predicted_errorbar
    return ratio

def creating_statistics_table(working_data_, parallax_24_summary_df_, *, is_save_=False,
                              output_path_='/Users/stela/Documents/Scripts/orbital_task/RTModel_plots'):
    """
    Create a table with the statistics of the parallax
    :return:
    """
    working_data_ = working_data_.set_index('event_name')
    for model_type in ['LX', 'LO']:
        # copying RTModel results for parallax
        # LX/LO

        # Absolute piE calculator:
        working_data_[f'{model_type}_model_piE'], working_data_[
            f'{model_type}_model_piE_error'] = absolute_pi_E_calculator(parallax_24_summary_df_[f'{model_type}_piEN'],
                                                                        parallax_24_summary_df_[f'{model_type}_piEE'],
                                                                        pi_E_N_error_=parallax_24_summary_df_[
                                                                            f'{model_type}_piEN_error'],
                                                                        pi_E_E_error_=parallax_24_summary_df_[
                                                                            f'{model_type}_piEE_error'],
                                                                        is_error_included_=True)
        # Saving to working dataframe piEE and piEN for the models LX/LO
        for piE_type in ['piEE', 'piEN']:
            working_data_[f'{model_type}_model_{piE_type}'] = parallax_24_summary_df_[f'{model_type}_{piE_type}']
            working_data_[f'{model_type}_model_{piE_type}_error'] = parallax_24_summary_df_[
                f'{model_type}_{piE_type}_error']

        # Calculating parallax residuals
        for piE_type in ['piE', 'piEE', 'piEN']:
            # Calculating parallax piE, piEE, piEN standardized residuals
            working_data_[f'{model_type}_{piE_type}_standardized_residual'] = standardized_residual_calculator(
                working_data_[f'{piE_type}'],
                working_data_[f'{model_type}_model_{piE_type}'],
                working_data_[f'{model_type}_model_{piE_type}_error'])

            # Calculating parallax piE, piEE, piEN absolute standardized residual
            working_data_[
                f'{model_type}_{piE_type}_absolute_standardized_residual'] = absolute_standardized_residual_calculator(
                working_data_[f'{piE_type}'], working_data_[f'{model_type}_model_{piE_type}'],
                working_data_[f'{model_type}_model_{piE_type}_error'])

            # Ratio of predicted value to predicted error
            working_data_[
                f'{model_type}_{piE_type}_predicted_value_predicted_errorbar_ratio'] = predicted_value_predicted_errorbar_ratio_calculator(
                working_data_[f'{model_type}_model_{piE_type}'], working_data_[f'{model_type}_model_{piE_type}_error'])

            # Absolute Ratio of predicted value to predicted error
            working_data_[f'{model_type}_{piE_type}_absolute_predicted_value_predicted_errorbar_ratio'] = np.abs(
                working_data_[f'{model_type}_{piE_type}_predicted_value_predicted_errorbar_ratio'])

            # Inverse. Ratio of predicted error to predicted value
            working_data_[f'{model_type}_{piE_type}_predicted_errorbar_predicted_value_ratio'] = working_data_[
                                                                                                 f'{model_type}_model_{piE_type}_error'] / \
                                                                                              working_data_[
                                                                                                 f'{model_type}_model_{piE_type}']

            # Inverse. Absolute Ratio of predicted error to predicted value
            working_data_[f'{model_type}_{piE_type}_absolute_predicted_errorbar_predicted_value_ratio'] = np.abs(
                working_data_[f'{model_type}_{piE_type}_predicted_errorbar_predicted_value_ratio'])

            # Calculating parallax piE, piEE, piEN relative error
            working_data_[f'{model_type}_{piE_type}_relative_difference_from_true_value'] = relative_difference_from_true_value_calculator(
                working_data_[f'{piE_type}'],
                working_data_[f'{model_type}_model_{piE_type}'])
    if is_save_:
        working_data_.to_csv(f'{output_path_}.csv')
    return working_data_


if __name__ == '__main__':
    # Example usage

    # LOAD YOUR DATA as a pandas DataFrame
    import pandas as pd

    # Load the sample master file
    representative_sample_df = pd.read_csv(
        '/Users/stela/Documents/Scripts/orbital_task/data/gulls_orbital_motion_extracted/OMPLDG_croin_cassan.sample.csv')
    representative_sample_df['event_name'] = 'event_' + representative_sample_df['SubRun'].astype(str) + '_' + \
                                             representative_sample_df['Field'].astype(str) + '_' + \
                                             representative_sample_df['EventID'].astype(str)
    # LOAD your parallax results
    parallax_24_summary_df =  pd.read_csv(f'/RTModel_runs/sample_rtmodel_v2.4/all_sample_rtmodel_v2.4_parallax.csv',
                                          index_col=['event_name'])

    # WORKING DATA
    # Preparing columns of interest
    columns_to_copy = ['event_name', 'piE', 'piEE', 'piEN', 'final_weight']
    working_data = representative_sample_df[columns_to_copy].copy()

    # IF SAVING SET AN OUTPUT
    output_for_table = ('/Users/stela/Documents/Scripts/orbital_task/RTModel_plots/'
                        'Notebook_statiscs_by_cuts/rtmodelv2.4_statistics')
    working_data = creating_statistics_table(working_data, parallax_24_summary_df,
                                             is_save_=True, output_path_=output_for_table)