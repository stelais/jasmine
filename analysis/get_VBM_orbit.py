import pandas as pd
import numpy as np
import sys



sample_path = '/Users/jmbrashe/VBBOrbital/NEWGULLS/OMPLDG_croin_cassan.sample.csv'
df = pd.read_csv(sample_path)

event_name = sys.argv[1]
event_name = event_name[5:]
lc_name = f'OMPLDG_croin_cassan/OMPLDG_croin_cassan{event_name}.det.lc'
event = df[df['lcname']==lc_name].iloc[0,:]

phi = event['Planet_orbphase']*np.pi/180 #orbital phase
T = event['Planet_period']*365.25 # period
n = 2*np.pi/(T)
i = event['Planet_inclination']*np.pi/180
s = event['Planet_s']
a_au = event['Planet_semimajoraxis']
theta_e = event['thetaE']
DL = event['Lens_Dist']*206264806.24538 # convert to AU

a_rad = a_au/DL
a_mas = a_rad*(180/np.pi)*3600*1000
a = a_mas/theta_e

gamma = n*a/s
gamma1 = -(n/np.tan(phi))*((a/s)**2-1)
gamma2 = n*((a/s)**2)*np.cos(i)
gamma3 = -n*np.sqrt((a/s)**2-1)/np.tan(phi)

print(f'v_para, v_perp, v_radial (day^-1), period (days) = {gamma1},{gamma2},{gamma3},{T}')