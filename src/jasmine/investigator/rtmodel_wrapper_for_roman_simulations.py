from pathlib import Path
import pandas as pd
from jasmine import ModelResults

from jasmine.classes_and_files_reader import RTModel_results_cls as rtm_results


def collect_event_summary(event_result, is_ICGS=False):
    """ Collects all parameters from an event result object."""
    data = {"event_name": event_result.event_name}
    categories = ["PS", "PX", "BS", "BO", "LS", "LX", "LO"]
    if is_ICGS:
        categories = ["PS", "PX", "LS", "LX", "LO"]
    for category in categories:
        best_obj = getattr(event_result, f"{category.lower()}_best")
        best_category_model_name = best_obj.name
        data[f"{category.upper()}_name"] = best_category_model_name
        print(event_result.event_folder + '/Models/' + best_category_model_name + '.txt')
        model_results = ModelResults(event_result.event_folder + '/Models/' + best_category_model_name + '.txt')
        # Dynamically add all attributes from model_parameters
        for attr, value in vars(model_results.model_parameters).items():
            data[f"{category.upper()}_{attr}"] = value
        print()
    return data


def loop_through_all_the_folders(folder_with_runs_path_, is_ICGS=False):
    all_event_summaries_ = []
    for folder in Path(folder_with_runs_path_).glob("*"):
        print(folder.name)
        if not folder.is_dir():
            continue
        try:
            event_result = rtm_results.EventResults(folder_with_runs_path_ + folder.name)
            event_result.extract_information_from_nature_file()
            event_result.looking_for_the_names_of_best_model_of_each_category()

            event_data = collect_event_summary(event_result, is_ICGS)

            event_data['complete_classification'] = event_result.complete_classification
            event_data['number_of_final_models'] = event_result.number_of_final_models
            event_data['final_models'] = event_result.final_models
            event_data['found_a_planetary_solution'] = event_result.found_a_planet
            event_data['found_a_2L1S_solution'] = event_result.found_a_2L1S_solution

            all_event_summaries_.append(event_data)

        except Exception as e:
            print(f"Error processing {folder.name}: {e}")

    return all_event_summaries_


def combining_models_with_true_values(true_values_table_path_, models_table_path_):
    """
    This function combines the true values with the models.
    :param true_values_path: Path to the true values CSV file.
    :param models_path: Path to the models CSV file.
    :return: DataFrame with combined data.
    """
    true_values_df = pd.read_csv(true_values_table_path_)
    true_values_df['event_name'] = (
        true_values_df['lcname']
        .str.replace('OMPLDG_croin_cassan/OMPLDG_croin_cassan_', 'event_', regex=False)
        .str.replace('.det.lc', '', regex=False)
    )
    models_df = pd.read_csv(models_table_path_)
    columns_of_interest = [
        "event_name", "Planet_q", "Planet_s", "piEN", "piEE", "u0lens1", "t0lens1",
        "tE_ref", "rho", "alpha", "Planet_inclination", "Planet_orbphase",
        "Planet_period", 'ObsGroup_0_chi2', 'ObsGroup_0_flatchi2'
    ]
    using_true_values = true_values_df[columns_of_interest].copy()

    # Rename columns
    rename_dict = {
        "Planet_q": "mass_ratio",
        "Planet_s": "separation",
        "u0lens1": "u0",
        "t0lens1": "t0",
        "tE_ref": "tE",
        "Planet_inclination": "planet_inclination",
        "Planet_orbphase": "planet_orbphase",
        "Planet_period": "planet_period"
    }

    using_true_values = using_true_values.rename(columns=rename_dict)

    # Rename all columns (except event_name and lcname) with "true_" prefix
    using_true_values = using_true_values.rename(
        columns={col: f"true_{col}" for col in using_true_values.columns if col not in ["event_name"]})
    combined_df = pd.merge(using_true_values, models_df, on='event_name', how='outer')
    # Calculate differences and errors for mass ratio in categories LS, LX, LO:
    categories = ['LS', 'LX', 'LO']
    for category in categories:
        model_prefix = category.lower()
        combined_df[f'{category.upper()}_mass_ratio_diff'] = (combined_df[f'{category.upper()}_mass_ratio'] -
                                                              combined_df['true_mass_ratio']).abs() / combined_df[
                                                                 f'{category.upper()}_mass_ratio_error']
    return combined_df


if __name__ == "__main__":
    # # Define the paths
    # # folder_with_runs_path = '/Users/stela/Documents/teste/sample_rtmodel_v2.4/'
    # folder_with_runs_path = '/gpfsm/dnb34/sishitan/orbital_task/RTModel_runs/269_problematic_events/'
    # # model_output_table_path = '/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/sample_rtmodel_v2.4/all_models_summary.csv'
    # model_output_table_path = '/gpfsm/dnb34/sishitan/orbital_task/RTModel_runs/269_problematic_events/all_models_summary.csv'
    # # master_file = '/Users/stela/Documents/Scripts/orbital_task/data/gulls_orbital_motion_extracted/OMPLDG_croin_cassan.sample.csv'
    # master_file = '/gpfsm/dnb34/sishitan/orbital_task/data/gulls_orbital_motion_extracted/OMPLDG_croin_cassan.sample.csv'
    # # true_and_rtmodel_fits_path = '/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/sample_rtmodel_v2.4/true_and_rtmodel_fits.csv'
    # true_and_rtmodel_fits_path = '/gpfsm/dnb34/sishitan/orbital_task/RTModel_runs/269_problematic_events/true_and_rtmodel_fits.csv'
    # Define the paths
    # folder_with_runs_path = '/gpfsm/dnb34/sishitan/orbital_task/RTModel_runs/sample_rtmodel_v3.1/'
    is_this_a_ICGS_run = True
    root_path = '/Users/stela/Documents/Scripts/orbital_task'
    # root_path = '/gpfsm/dnb34/sishitan/orbital_task'
    folder_with_runs_path = f'{root_path}/RTModel_runs/154_failures_v24_v31/'
    model_output_table_path = f'{root_path}/RTModel_runs/154_failures_v24_v31/all_models_summary.csv'
    master_file = f'{root_path}/data/gulls_orbital_motion_extracted/OMPLDG_croin_cassan.sample.csv'
    true_and_rtmodel_fits_path = f'{root_path}/RTModel_runs/154_failures_v24_v31/ICGS_true_and_rtmodel_fits.csv'

    
    all_event_summaries = loop_through_all_the_folders(folder_with_runs_path, is_ICGS=is_this_a_ICGS_run)
    model_summary_df = pd.DataFrame(all_event_summaries)
    model_summary_df.to_csv(model_output_table_path, index=False)
    print(f"✅ Saved {len(model_summary_df)} events to {model_output_table_path}")

    # Combine with true values
    final_df = combining_models_with_true_values(master_file, model_output_table_path)
    order_of_columns_to_be_saved = [
        'event_name',
        'complete_classification', 'found_a_planetary_solution', 'found_a_2L1S_solution',
        'number_of_final_models', 'final_models',
        'true_mass_ratio', 'true_separation', 'true_piEN', 'true_piEE', 'true_u0', 'true_t0', 'true_tE',
        'true_rho', 'true_alpha', 'true_planet_inclination', 'true_planet_orbphase', 'true_planet_period',
        'true_ObsGroup_0_chi2', 'true_ObsGroup_0_flatchi2',
        'PS_name', 'PS_chi2', 'PS_number_of_parameters', 'PS_u0', 'PS_u0_error', 'PS_tE', 'PS_tE_error', 'PS_t0',
        'PS_t0_error',
        'PS_rho', 'PS_rho_error', 'PS_blends', 'PS_sources', 'PS_blendings', 'PS_blendings_error',
        'PS_baselines', 'PS_baselines_error',
        'PX_name', 'PX_chi2', 'PX_number_of_parameters', 'PX_u0', 'PX_u0_error', 'PX_tE', 'PX_tE_error', 'PX_t0',
        'PX_t0_error',
        'PX_rho', 'PX_rho_error', 'PX_piEN', 'PX_piEN_error', 'PX_piEE', 'PX_piEE_error',
        'PX_blends', 'PX_sources', 'PX_blendings', 'PX_blendings_error', 'PX_baselines', 'PX_baselines_error',
        'BS_name', 'BS_chi2', 'BS_number_of_parameters', 'BS_tE', 'BS_tE_error', 'BS_flux_ratio', 'BS_flux_ratio_error',
        'BS_u01', 'BS_u01_error', 'BS_u02', 'BS_u02_error', 'BS_t01', 'BS_t01_error', 'BS_t02', 'BS_t02_error',
        'BS_rho1', 'BS_rho1_error', 'BS_blends', 'BS_sources', 'BS_blendings', 'BS_blendings_error',
        'BS_baselines', 'BS_baselines_error',
        'BO_name', 'BO_chi2', 'BO_number_of_parameters', 'BO_u01', 'BO_u01_error', 'BO_t01', 'BO_t01_error', 'BO_tE',
        'BO_tE_error',
        'BO_rho1', 'BO_rho1_error', 'BO_xi1', 'BO_xi1_error', 'BO_xi2', 'BO_xi2_error', 'BO_omega', 'BO_omega_error',
        'BO_inc', 'BO_inc_error', 'BO_phi', 'BO_phi_error', 'BO_qs', 'BO_qs_error', 'BO_blends',
        'BO_sources',
        'BO_blendings', 'BO_blendings_error', 'BO_baselines', 'BO_baselines_error',
        'LS_name', 'LS_chi2', 'LS_number_of_parameters', 'LS_separation', 'LS_separation_error', 'LS_mass_ratio',
        'LS_mass_ratio_error', 'LS_mass_ratio_diff',
        'LS_u0', 'LS_u0_error', 'LS_alpha', 'LS_alpha_error', 'LS_rho', 'LS_rho_error', 'LS_tE', 'LS_tE_error',
        'LS_t0', 'LS_t0_error', 'LS_blends', 'LS_sources', 'LS_blendings', 'LS_blendings_error',
        'LS_baselines', 'LS_baselines_error',
        'LX_name', 'LX_chi2', 'LX_number_of_parameters', 'LX_separation', 'LX_separation_error', 'LX_mass_ratio',
        'LX_mass_ratio_error', 'LX_mass_ratio_diff',
        'LX_u0', 'LX_u0_error', 'LX_alpha', 'LX_alpha_error', 'LX_rho', 'LX_rho_error', 'LX_tE', 'LX_tE_error',
        'LX_t0', 'LX_t0_error', 'LX_piEN', 'LX_piEN_error', 'LX_piEE', 'LX_piEE_error', 'LX_blends',
        'LX_sources',
        'LX_blendings', 'LX_blendings_error', 'LX_baselines', 'LX_baselines_error',
        'LO_name', 'LO_chi2', 'LO_number_of_parameters', 'LO_separation', 'LO_separation_error', 'LO_mass_ratio',
        'LO_mass_ratio_error', 'LO_mass_ratio_diff',
        'LO_u0', 'LO_u0_error', 'LO_alpha', 'LO_alpha_error', 'LO_rho', 'LO_rho_error', 'LO_tE',
        'LO_tE_error',
        'LO_t0', 'LO_t0_error', 'LO_piEN', 'LO_piEN_error', 'LO_piEE', 'LO_piEE_error',
        'LO_gamma1', 'LO_gamma1_error', 'LO_gamma2', 'LO_gamma2_error', 'LO_gammaz', 'LO_gammaz_error',
        'LO_blends', 'LO_sources', 'LO_blendings', 'LO_blendings_error', 'LO_baselines',
        'LO_baselines_error']
    if is_this_a_ICGS_run:
        order_of_columns_to_be_saved = [
            'event_name',
            'complete_classification', 'found_a_planetary_solution', 'found_a_2L1S_solution',
            'number_of_final_models', 'final_models',
            'true_mass_ratio', 'true_separation', 'true_piEN', 'true_piEE', 'true_u0', 'true_t0', 'true_tE',
            'true_rho', 'true_alpha', 'true_planet_inclination', 'true_planet_orbphase', 'true_planet_period',
            'true_ObsGroup_0_chi2', 'true_ObsGroup_0_flatchi2',
            'PS_name', 'PS_chi2', 'PS_number_of_parameters', 'PS_u0', 'PS_u0_error', 'PS_tE', 'PS_tE_error', 'PS_t0',
            'PS_t0_error',
            'PS_rho', 'PS_rho_error', 'PS_blends', 'PS_sources', 'PS_blendings', 'PS_blendings_error',
            'PS_baselines', 'PS_baselines_error',
            'PX_name', 'PX_chi2', 'PX_number_of_parameters', 'PX_u0', 'PX_u0_error', 'PX_tE', 'PX_tE_error', 'PX_t0',
            'PX_t0_error',
            'PX_rho', 'PX_rho_error', 'PX_piEN', 'PX_piEN_error', 'PX_piEE', 'PX_piEE_error',
            'PX_blends', 'PX_sources', 'PX_blendings', 'PX_blendings_error', 'PX_baselines', 'PX_baselines_error',
            'LS_name', 'LS_chi2', 'LS_number_of_parameters', 'LS_separation', 'LS_separation_error', 'LS_mass_ratio',
            'LS_mass_ratio_error', 'LS_mass_ratio_diff',
            'LS_u0', 'LS_u0_error', 'LS_alpha', 'LS_alpha_error', 'LS_rho', 'LS_rho_error', 'LS_tE', 'LS_tE_error',
            'LS_t0', 'LS_t0_error', 'LS_blends', 'LS_sources', 'LS_blendings', 'LS_blendings_error',
            'LS_baselines', 'LS_baselines_error',
            'LX_name', 'LX_chi2', 'LX_number_of_parameters', 'LX_separation', 'LX_separation_error', 'LX_mass_ratio',
            'LX_mass_ratio_error', 'LX_mass_ratio_diff',
            'LX_u0', 'LX_u0_error', 'LX_alpha', 'LX_alpha_error', 'LX_rho', 'LX_rho_error', 'LX_tE', 'LX_tE_error',
            'LX_t0', 'LX_t0_error', 'LX_piEN', 'LX_piEN_error', 'LX_piEE', 'LX_piEE_error', 'LX_blends',
            'LX_sources',
            'LX_blendings', 'LX_blendings_error', 'LX_baselines', 'LX_baselines_error',
            'LO_name', 'LO_chi2', 'LO_number_of_parameters', 'LO_separation', 'LO_separation_error', 'LO_mass_ratio',
            'LO_mass_ratio_error', 'LO_mass_ratio_diff',
            'LO_u0', 'LO_u0_error', 'LO_alpha', 'LO_alpha_error', 'LO_rho', 'LO_rho_error', 'LO_tE',
            'LO_tE_error',
            'LO_t0', 'LO_t0_error', 'LO_piEN', 'LO_piEN_error', 'LO_piEE', 'LO_piEE_error',
            'LO_gamma1', 'LO_gamma1_error', 'LO_gamma2', 'LO_gamma2_error', 'LO_gammaz', 'LO_gammaz_error',
            'LO_blends', 'LO_sources', 'LO_blendings', 'LO_blendings_error', 'LO_baselines',
            'LO_baselines_error']
    # check if all columns are present
    missing_columns = [col for col in order_of_columns_to_be_saved if col not in final_df.columns]
    if missing_columns:
        print("❗ Warning: The following columns are missing from the final DataFrame:", missing_columns)
    else:
        print("✅ All dcolumns are present in the DataFrame.")
    final_df = final_df[order_of_columns_to_be_saved]
    # excluding some columns that are not needed
    categories = ['PS', 'PX', 'BS', 'BO', 'LS', 'LX', 'LO']
    if is_this_a_ICGS_run:
        categories = ['PS', 'PX', 'LS', 'LX', 'LO']
    for category in categories:
        final_df = final_df.drop(columns=[f'{category}_blends',
                                         f'{category}_sources', f'{category}_blendings',
                                            f'{category}_blendings_error', f'{category}_baselines',
                                            f'{category}_baselines_error'])
    final_df.to_csv(true_and_rtmodel_fits_path, index=False)
    print("✅ Combined true values with model results and saved to", true_and_rtmodel_fits_path)
