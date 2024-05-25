from jasmine.classes_and_files_reader.datachallenge_lightcurve_cls import LightcurveEventDataChallenge
import pandas as pd
import matplotlib.pyplot as plt


list_of_bound_planet_events = [4, 8, 12, 25, 32, 40, 47, 50, 53, 62, 66, 69, 74, 78, 81, 92, 95, 99, 100, 103,
                               107, 124, 128, 131, 139, 152, 163, 186, 193, 194, 199, 208, 214, 217, 218, 223,
                               226, 227, 250, 253, 258, 267, 289]
# list_of_bound_planet_events = [4,]
master_path = '/local/data/emussd1/greg_shared/rtmodel_effort/datachallenge'
# master_path = '/Users/sishitan/Documents/Scripts/jasmine'
df_general = []
for event_number in list_of_bound_planet_events:
    true_values_path = f'{master_path}/data'
    the_lightcurve_event = LightcurveEventDataChallenge(event_number, true_values_path)
    true_q = the_lightcurve_event.planet.planet_mass_ratio
    true_s = the_lightcurve_event.planet.planet_separation
    data_to_be_saved = {'data_challenge_lc_number': [event_number,],
                        'true_q': [true_q,],
                        'true_s': [true_s,],}
    df_for_one_event = pd.DataFrame(data_to_be_saved)
    df_general.append(df_for_one_event)
# Concatenate all the dataframes
wrapper_df = pd.concat(df_general)
wrapper_df.plot(x='true_s', y='true_q', kind='scatter', title='q VS s: 43 planet events', alpha=0.5)
plt.savefig('q_vs_s_43_planet_events.png', dpi=300)

wrapper_df.plot(x='true_s', y='true_q', kind='scatter', title='q VS s: 43 planet events', alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.savefig('log_scale_q_vs_s_43_planet_events.png', dpi=300)
