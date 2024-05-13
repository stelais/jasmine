import pandas as pd
import numpy as np
import os
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric
from astropy.time import Time


def creating_ephemerides(*,
                         satellite_folder_path_='/Users/jmbrashe/VBBOrbital/satellitedir',
                         ephemerides_path_='/Users/jmbrashe/VBBOrbital/data-challenge-1'):
    if os.path.isdir(satellite_folder_path_):
        print(f'Directory {satellite_folder_path_} already exists! Continuing to file creation.')
    else:
        os.mkdir(satellite_folder_path_)
        print(f'Folder {satellite_folder_path_} created!')
    solar_system_ephemeris.set('de432s')
    filters = ['W149', 'Z087']
    deldot_str = '{:>11}'.format('0.0000000')  # string of zeros for the delta r column
    names = ['BJD', 'X_EQ', 'Y_EQ', 'Z_EQ', 'X_ECL', 'Y_ECL', 'Z_ECL']  # names of data challenge ephemerides columns
    au_per_km = 1 / 149_597_870.700  # for conversion
    for i in range(len(filters)):
        ephname_ = f'satellite{i + 1}.txt'
        if os.path.exists(f'{satellite_folder_path_}/{ephname_}'):
            raise FileExistsError(f'Ephemerides file {ephname_} already exists in directory {satellite_folder_path_}')
        else:
            eph_df = pd.read_csv(ephemerides_path_ + f'/wfirst_ephemeris_{filters[i]}' + '.txt', sep='\s+', names=names)
            eq_coords = SkyCoord(x=eph_df.iloc[:, 1], y=eph_df.iloc[:, 2], z=eph_df.iloc[:, 3],
                                 unit='au', frame='icrs', obstime='J2000', representation_type='cartesian')
            # Get earth location in cartesian ICRS to calculate actual distance ...
            t = Time(eph_df['BJD'], format='jd')
            eloc = get_body_barycentric('earth', t)
            earth_loc = SkyCoord(eloc, unit='km', frame='icrs', obstime='J2000', representation_type='cartesian')
            earth_loc = earth_loc.to_table().to_pandas()
            earth_loc = earth_loc * au_per_km
            roman_loc = eq_coords.to_table().to_pandas()
            roman_earth_distance = (roman_loc - earth_loc) ** 2
            roman_earth_distance = roman_earth_distance.sum(axis=1)
            roman_earth_distance = np.sqrt(roman_earth_distance)
            eq_coords.representation_type = 'spherical'
            eq_coords = eq_coords.to_table().to_pandas()
            eq_coords['distance'] = roman_earth_distance
            eq_coords['delta_dot'] = np.zeros(eq_coords.shape[0])
            eq_coords['BJD'] = eph_df['BJD']
            eq_coords = eq_coords[['BJD', 'ra', 'dec', 'distance', 'delta_dot']]
            with open(f'{satellite_folder_path_}/{ephname_}', 'w') as f:
                f.write('$$SOE\n')
                for j in range(eq_coords.shape[0]):
                    line = makeline(eq_coords.iloc[j, 0:4], deldot_str)
                    f.write(line)
                f.write('$$EOE')
            print(f'File {ephname_} created in {satellite_folder_path_}.')


# function for rounding a string so that is n characters long.
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


# makes one line of the ephemerides file.
def makeline(dfline, deldot):
    bjd_str = '{:0<17}'.format(dfline['BJD'])
    ra_str = str(round(dfline['ra'], 5))
    ra_split = ra_str.split('.')
    ra_split[1] = '{:0<5}'.format(ra_split[1])
    ra_str = ra_split[0] + '.' + ra_split[1]
    ra_str = '{:>13}'.format(ra_str)

    dec_str = str(round(dfline['dec'], 5))
    dec_split = dec_str.split('.')
    dec_split[1] = '{:0<5}'.format(dec_split[1])
    dec_str = dec_split[0] + '.' + dec_split[1]
    dec_str = '{:>9}'.format(dec_str)
    dist_str = roundton(dfline['distance'], 16)
    line = bjd_str + ' ' + ra_str + ' ' + dec_str + ' ' + dist_str + ' ' + deldot + '\n'
    return line


if __name__ == '__main__':
    satellite_folder = '/Users/jmbrashe/VBBOrbital/satellitedir/'
    ephemerides_location = '/Users/jmbrashe/VBBOrbital/data-challenge-1/'
    creating_ephemerides(satellite_folder_path_=satellite_folder, ephemerides_path_=ephemerides_location)
