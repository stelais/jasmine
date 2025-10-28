import pandas as pd
import numpy as np
import re

def hms_to_seconds(hms):
    h, m, s = map(int, hms.split(':'))
    return h * 3600 + m * 60 + s



def main(list_of_events_path_, log_names_path_, event_run_path_, logfile_base_path_):
    list_of_events_ = pd.read_table(list_of_events_path_, header=None)
    list_of_logfiles = pd.read_table(log_names_path_, header=None)
    stats = []
    for run_index in range(len(list_of_events_)):
        event_name = list_of_events_.iloc[run_index, 0]
        print(f"Event: {event_name}")
        grid_fit_path = f'{event_run_path_}/{event_name}/Data/grid_fit.txt'
        grid_df = pd.read_csv(grid_fit_path, delim_whitespace=True, names=['grid_index', 'chi2', 'calculation_time'])
        min_time = grid_df['calculation_time'].min()
        print(f"min_time: {min_time}")
        max_time = grid_df['calculation_time'].max()
        print(f"max_time: {max_time}")
        median_time = grid_df['calculation_time'].median()
        print(f"median_time: {median_time}")
        mean_time = grid_df['calculation_time'].mean()
        logfile_name = list_of_logfiles.iloc[run_index, 0]
        logfile_path = f'{logfile_base_path_}/{logfile_name}'
        with open(logfile_path, 'r') as f:
            for line in f:
                icgs_match = re.search(r'ICGS time:\s*([\d\.]+)', line)
                rtmodel_match = re.search(r'RTModel time:\s*([\d\.]+)', line)
                walltime_used_match = re.search(r'Walltime Used                   :\s*([\d:]+)', line)
                if icgs_match:
                    print(icgs_match)
                    icgs_times = float(icgs_match.group(1)) * 10
                    icgs_in_hours = icgs_times / 3600
                if rtmodel_match:
                    print(rtmodel_match)
                    rtmodel_times = float(rtmodel_match.group(1)) * 10
                    rtmodel_in_hours = rtmodel_times / 3600
                if walltime_used_match:
                    print(walltime_used_match)
                    walltime_used = walltime_used_match.group(1)
                    walltime_used_in_seconds = hms_to_seconds(walltime_used) * 10
                    walltime_used_in_hours = walltime_used_in_seconds / 3600
        stats.append({
            'event_name': event_name,
            'VBM_min_time_seconds': min_time,
            'VBM_max_time_seconds': max_time,
            'VBM_median_time_seconds': median_time,
            'VBM_mean_time_seconds': mean_time,
            'ICGS_time_total_hours': icgs_in_hours,
            'RTModel_time_total_hours': rtmodel_in_hours,
            'CPU_hours': walltime_used_in_hours,
            # 'ICGS_time_seconds': icgs_times,
            # 'RTModel_time': rtmodel_times,
            # 'Walltime': walltime_used_in_seconds,
        })
        print(grid_df.head())

    time_statistic = pd.DataFrame(stats)
    print(time_statistic)

    time_statistic.to_csv('time_statistics_slow_runs.csv', index=False)

if __name__ == "__main__":

    root_path = f'/gpfsm/dnb34/sishitan/orbital_task'
    event_run_path = f'{root_path}/RTModel_runs/154_failures_v24_v31/second_run'
    logfile_base_path = root_path
    list_of_events_path = f'{root_path}/RTModel_runs/154_failures_v24_v31/list_rerun_26.txt'
    log_names_path = f'{root_path}/RTModel_runs/154_failures_v24_v31/second_run/slurm_files_names.txt'
    main(list_of_events_path, log_names_path, event_run_path, logfile_base_path)