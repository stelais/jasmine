import pandas as pd
import numpy as np
import os
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body
from astropy.time import Time


def creating_ephemerides_from_lc(*,
                                 satellite_folder_path_='/Users/jmbrashe/VBBOrbital/satellitedir',
                                 eph_list):
    if os.path.isdir(satellite_folder_path_):
        print(f'Directory {satellite_folder_path_} already exists! Continuing to file creation.')
    else:
        os.mkdir(satellite_folder_path_)
        print(f'Folder {satellite_folder_path_} created!')
    solar_system_ephemeris.set('de432s')   # The planetary ephemeris DE432 is
    # based on the ephemeris DE430 (Folkner et al. 2014; default if 'jpl').
    # The main difference between DE432 and DE430 is an update of the estimated orbit of
    # the Pluto system barycenter to support the New Horizons project.
    filters = ['W146', 'Z087', 'K213']
    deldot_str = '{:>11}'.format('0.0000000')  # string of zeros for the delta r column
    names = ['BJD', 'X_EQ', 'Y_EQ', 'Z_EQ']  # names of data challenge ephemerides columns
    au_per_km = 1 / 149_597_870.700  # for conversion
    for i in range(len(filters)):
        ephname_ = f'satellite{i + 1}.txt'
        if os.path.exists(f'{satellite_folder_path_}/{ephname_}'):
            raise FileExistsError(f'Ephemerides file {ephname_} already exists in directory {satellite_folder_path_}')
        else:
            eph_df = eph_list[i]
            eph_df = eph_df.reset_index(drop=True)
            # ecliptic_coords = SkyCoord(x=eph_df.iloc[:, 1], y=eph_df.iloc[:, 2], z=eph_df.iloc[:, 3],
            #                            unit='au', frame='barycentrictrueecliptic', obstime='J2000',
            #                            representation_type='cartesian')
            # eq_coords = ecliptic_coords.transform_to('icrs')
            # eq_coords.representation_type = 'cartesian'
            # SIS edits:
            ecliptic_coords = SkyCoord(x=eph_df['X_EQ'], y=eph_df['Y_EQ'], z=eph_df['Z_EQ'],
                                       unit='au', frame='geocentrictrueecliptic', obstime='J2000',
                                       representation_type='cartesian')

            # Transform to ICRS (barycentric coordinates)
            icrs_coords = ecliptic_coords.transform_to('icrs')
            # Extract the Cartesian coordinates in AU
            icrs_cartesian = icrs_coords.cartesian
            roman_loc = pd.DataFrame({
                'x': icrs_cartesian.x.value,
                'y': icrs_cartesian.y.value,
                'z': icrs_cartesian.z.value,
                'total': np.sqrt(icrs_cartesian.x.value**2 + icrs_cartesian.y.value**2 + icrs_cartesian.z.value**2)})

            # eq_coords = ecliptic_coords.transform_to('icrs')
            # # eq_coords = ecliptic_coords.transform_to('icrs') # SIS commented out
            print("Ecliptic coords (AU):", ecliptic_coords)
            print("ICRS coords (AU):", roman_loc)

            ###
            # Get earth location in cartesian ICRS to calculate actual distance ...
            t = Time(eph_df['BJD'], format='jd')
            print(eph_df['BJD'])
            # print(t)
            # eloc = get_body_barycentric('earth', t)
            # earth_loc = SkyCoord(eloc, unit='km', frame='icrs', obstime='J2000', representation_type='cartesian')
            # earth_loc = earth_loc.to_table().to_pandas()
            # earth_loc = earth_loc * au_per_km
            # roman_loc = eq_coords.to_table().to_pandas()

            # SIS EARTH
            earth_barycentric = get_body_barycentric('earth', t)  # Already returns a SkyCoord object in ICRS
            # Convert to Cartesian representation and ensure units are in AU
            earth_barycentric_au = earth_barycentric.xyz.to('AU')
            # Convert to a pandas DataFrame
            data = {'x': earth_barycentric_au[0].value,
                    'y': earth_barycentric_au[1].value,
                    'z': earth_barycentric_au[2].value,
                    'total': np.sqrt(earth_barycentric_au[0].value**2 +
                                     earth_barycentric_au[1].value**2 +
                                     earth_barycentric_au[2].value**2)}  # Extract values from Quantity
            earth_loc = pd.DataFrame(data) # Transform to DataFrame if needed

            roman_earth_distance = (roman_loc - earth_loc) ** 2
            roman_earth_distance = roman_earth_distance.sum(axis=1)
            roman_earth_distance = np.sqrt(roman_earth_distance)



            roman_loc.representation_type = 'spherical'
            eq_coords = roman_loc.to_table().to_pandas()
            eq_coords['distance'] = roman_earth_distance
            eq_coords['delta_dot'] = np.zeros(eq_coords.shape[0])
            eq_coords['BJD'] = eph_df['BJD']
            print(eq_coords.shape, eph_df.shape)
            print(eq_coords['BJD'])
            eq_coords = eq_coords[['BJD', 'ra', 'dec', 'distance', 'delta_dot']]
            with open(f'{satellite_folder_path_}/{ephname_}', 'w') as f:
                f.write('$$SOE\n')
                for j in range(eq_coords.shape[0]):
                    line = makeline(eq_coords.iloc[j, 0:4], deldot_str)

                    f.write(line)
                f.write('$$EOE')
            print(f'File {ephname_} created in {satellite_folder_path_}.')


def ephemeris_data_reader(SubRun, Field, ID, *, folder_path_='data'):
    """
    This function reads the lightcurve data file and returns a pandas dataframe with the Ephemeris XYZ +BJD Data
    :param folder_path_:
    :param data_challenge_lc_number_:
    :param filter_:
    :return: lightcurve_data_df
    """
    fname = f'OMPLDG_croin_cassan_{SubRun}_{Field}_{ID}.det.lc'
    columns = ['Simulation_time', 'measured_relative_flux', 'measured_relative_flux_error', 'true_relative_flux',
               'true_relative_flux_error', 'observatory_code', 'saturation_flag', 'best_single_lens_fit',
               'parallax_shift_t', 'parallax_shift_u', 'BJD', 'source_x', 'source_y', 'lens1_x', 'lens1_y', 'lens2_x',
               'lens2_y', 'X_EQ', 'Y_EQ', 'Z_EQ']
    lightcurve_data_df = pd.read_csv(f'{folder_path_}/{fname}', names=columns, comment='#', sep='\s+')
    eph_list_ = []
    for observatory_code in range(3):
        obs_data = lightcurve_data_df[lightcurve_data_df['observatory_code'] == observatory_code]
        eph_list_.append(obs_data[['BJD', 'X_EQ', 'Y_EQ', 'Z_EQ']])
    return eph_list_


# makes one line of the ephemerides file.
def makeline(dfline, deldot):
    bjd_str = '{:0<17}'.format(dfline['BJD'])
    # print(dfline['BJD'])
    # print(bjd_str)
    ra_str = f"{round(dfline['ra'], 5):5f}"
    ra_split = ra_str.split('.')
    ra_split[1] = '{:0<5}'.format(ra_split[1])
    ra_str = ra_split[0] + '.' + ra_split[1]
    ra_str = '{:>13}'.format(ra_str)
    dec_str = f"{round(dfline['dec'], 5):5f}"
    dec_split = dec_str.split('.')
    # print(dec_split)
    dec_split[1] = '{:0<5}'.format(dec_split[1])
    dec_str = dec_split[0] + '.' + dec_split[1]
    dec_str = '{:>9}'.format(dec_str)
    dist_str = roundton(dfline['distance'], 16)
    line = bjd_str + ' ' + ra_str + ' ' + dec_str + ' ' + dist_str + ' ' + deldot + '\n'
    return line


def roundton(sval, n):
    l_ = list(str(sval))
    if len(l_) <= n:
        s_ = "".join(l_)
        return s_.ljust(n, ' ')
    else:
        if int(l_[n]) >= 5:
            if int(l_[n - 1]) < 9:
                l_[n - 1] = str(int(l_[n - 1]) + 1)
            elif int(l_[n - 1]) == 9:
                j = 1
                while int(l_[n - j]) == 9:
                    l_[n - j] = '0'
                    j += 1
                l_[n - j] = str(int(l_[n - j]) + 1)
            s_ = "".join(l_[0:n])
            return s_
        else:
            s_ = "".join(l_[0:n])
            return s_


if __name__ == "__main__":
    general_path = '/Users/stela/Documents/Scripts/orbital_task/data'
    eph_list = ephemeris_data_reader(0, 163, 1174,
                                     folder_path_=f'{general_path}/gulls_orbital_motion_extracted/OMPLDG_croin_cassan_sample')
    creating_ephemerides_from_lc(satellite_folder_path_=f'{general_path}/satellitedir',
                                 eph_list=eph_list)
