"""
You must run rtmodel_wrapper_for_roman_simulations.py first to generate the input CSV file.
"""
import pandas as pd
import numpy as np


def main(file_path_, run_version_, root_path_):
    df = pd.read_csv(file_path_)
    df.dropna(inplace=True)

    # best chi2 of Ls and LX (excluding LO for now - cause crazy orbits)
    df['best_chi2'] = df[['LS_chi2', 'LX_chi2']].min(axis=1)
    # name of the lowest chi2 models:
    df['best_chi2_name'] = df[['LS_chi2', 'LX_chi2']].idxmin(axis=1)
    # save the best mass ratio according to the lowest chi2 model:
    df['best_mass_ratio'] = df.apply(lambda row: row[f"{str(row['best_chi2_name']).replace('_chi2', '')}_mass_ratio"],
                                     axis=1)

    # Step 1 : Create a boolean mask for chi2 difference > 500 for any model .
    # e.g. if LS_chi2 is 500 (or more) smaller than PS_chi2, then boolean is true = passes the cut
    print('\nChi2 cut...')
    chi2_mask_LS = (df['PS_chi2'] - df[f'LS_chi2']) > 500
    chi2_mask_LX = (df['PS_chi2'] - df[f'LX_chi2']) > 500
    chi2_mask_best = (df['PS_chi2'] - df[f'best_chi2']) > 500
    combined_chi2_mask_or = chi2_mask_LS | chi2_mask_LX
    combined_chi2_mask_and = chi2_mask_LS & chi2_mask_LX
    print("chi2 > 500 counts:")
    print("  LS:", chi2_mask_LS.sum())
    print("  LX:", chi2_mask_LX.sum())
    print("  best model:", chi2_mask_best.sum())
    print('  combined OR: ', combined_chi2_mask_or.sum())
    print('  combined AND: ', combined_chi2_mask_and.sum())
    print()

    # --- Step 2: mass ratio masks ---
    # e.g. if LS_mass_ratio is off by more than a factor of 2 (0.3 in log10) from true_mass_ratio, then boolean is true = passes the cut
    print('\nMass ratio cut...')
    ratio_LS = df['LS_mass_ratio'] / df['true_mass_ratio']
    mass_mask_LS = np.abs(np.log10(ratio_LS.replace(0, np.nan))) > 0.3
    ratio_LX = df['LX_mass_ratio'] / df['true_mass_ratio']
    mass_mask_LX = np.abs(np.log10(ratio_LX.replace(0, np.nan))) > 0.3
    ratio_best = df['best_mass_ratio'] / df['true_mass_ratio']
    mass_mask_best = np.abs(np.log10(ratio_best.replace(0, np.nan))) > 0.3
    print("mass ratio counts (|log10(q_model/q_true)| > 0.3):")
    print("  LS:", mass_mask_LS.sum())
    print("  LX:", mass_mask_LX.sum())
    print("  best model:", mass_mask_best.sum())
    print()

    # --- Step 3: combined per-model masks ---
    print('Combining chi2 and mass ratio cuts per model...')
    mask_LS = chi2_mask_LS & mass_mask_LS
    mask_LX = chi2_mask_LX & mass_mask_LX
    mask_best = chi2_mask_best & mass_mask_best
    print("Events passing BOTH cuts per model type only:")
    print("  LS:", mask_LS.sum())
    print("  LX:", mask_LX.sum())
    print("  best model:", mask_best.sum())
    print("Events passing combined chi2 cut OR, but per model mass ratio:")
    print("  LS:", (combined_chi2_mask_or & mass_mask_LS).sum())
    print("  LX:", (combined_chi2_mask_or & mass_mask_LX).sum())
    master_combined_or = combined_chi2_mask_or & (mass_mask_LS & mass_mask_LX)
    print("  Combined chi2 cut OR and all mass ratio cuts:", master_combined_or.sum())
    print("Events passing combined chi2 cut AND, but per model mass ratio:")
    print("  LS:", (combined_chi2_mask_and & mass_mask_LS).sum())
    print("  LX:", (combined_chi2_mask_and & mass_mask_LX).sum())
    master_combined_and = combined_chi2_mask_and & (mass_mask_LS & mass_mask_LX)
    print("  Combined chi2 cut AND and all mass ratio cuts:", master_combined_and.sum())

    # # --- Step 4: Combined mask versions, define criteria ---
    print("\nTotal events passing ALL model:", (mask_LS & mask_LX).sum())
    print("Total unique events passing any model:", (mask_LS | mask_LX).sum())
    print("Total unique events passing best model:", mask_best.sum())

    print('\nCriterion 1: ')
    print('Criterion 1: If any PS_chi2 - L*_chi2 > 500 AND  All abs(log10(L*_mass_ratio / true_mass_ratio)) > 0.3')
    print('v24: 237 events and v31: 227 events')
    print("Combined chi2 cut OR and all mass ratio cuts:", master_combined_or.sum())
    selected_c1 = df[master_combined_or].copy()

    print('\nCriterion 2: ')
    print('Criterion 2: If all PS_chi2 - L*_chi2 > 500 AND All abs(log10(L*_mass_ratio / true_mass_ratio)) > 0.3')
    print('v24: 191 events and v31: 184 events')
    print("chi2_mask_LS & mass_mask_LS & chi2_mask_LX & mass_mask_LX")
    print("Total events passing ALL model:", (mask_LS & mask_LX).sum())
    selected_c2 = df[(mask_LS & mask_LX)].copy()

    print('\nCriterion 3: ')
    print('Criterion 3: If PS_chi2 - best_L*_chi2 > 500  AND  abs(log10(best_L*_mass_ratio / true_mass_ratio)) > 0.3')
    print('v24: 257 events and v31: 267 events')
    print("Total unique events passing best model:", mask_best.sum())
    selected_c3 = df[mask_best].copy()
    print()

    # --- Step 5: select rows ---
    selected_c1 = selected_c1[['event_name', 'true_mass_ratio',
                               'LS_mass_ratio', 'LX_mass_ratio',
                               'PS_chi2', 'LS_chi2', 'LX_chi2']]
    selected_c2 = selected_c2[['event_name', 'true_mass_ratio',
                               'LS_mass_ratio', 'LX_mass_ratio',
                               'PS_chi2', 'LS_chi2', 'LX_chi2']]
    selected_c3 = selected_c3[['event_name', 'true_mass_ratio',
                               'LS_mass_ratio', 'LX_mass_ratio',
                               'PS_chi2', 'LS_chi2', 'LX_chi2']]

    # --- Step 6: save files ---
    # Save the filtered DataFrame to a new CSV file
    selected_c1.to_csv(f'{root_path_}/c1_{run_version_}_events_failure_in_q.csv', index=False)
    selected_c2.to_csv(f'{root_path_}/c2_{run_version_}_events_failure_in_q.csv', index=False)
    selected_c3.to_csv(f'{root_path_}/c3_{run_version_}_events_failure_in_q.csv', index=False)


if __name__ == "__main__":
    # Load the CSV file generated by `rtmodel_wrapper_for_roman_simulations.py`
    # run_version = 'v31'
    # run_version = 'v24'
    run_version = 'ICGS'
    print(run_version)
    # ADAPT THIS
    if run_version == 'v31':
        root_path = '/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/sample_rtmodel_v3.1'
        file_path = f'{root_path}/v3.1_true_and_rtmodel_fits.csv'
    elif run_version == 'v24':
        root_path = '/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/sample_rtmodel_v2.4'
        file_path = f'{root_path}/true_and_rtmodel_fits.csv'
    elif run_version == 'ICGS':
        root_path = '/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/154_failures_v24_v31'
        file_path = f'{root_path}/ICGS_true_and_rtmodel_fits_done.csv'
    else:
        raise ValueError("Invalid run_version. Use 'v31' or 'v24' or 'ICGS'.")
    main(file_path, run_version, root_path)
