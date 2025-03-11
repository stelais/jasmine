from astropy.coordinates import SkyCoord
import astropy.units as u

# Function to convert RA from 'HH:MM:SS' to decimal degrees
def ra_hms_to_deg(ra_hms):
    h, m, s = [float(i) for i in ra_hms.split(':')]
    ra_deg = 15 * (h + m / 60 + s / 3600)
    return ra_deg


# Function to convert Dec from 'DD:MM:SS' to decimal degrees
def dec_dms_to_deg(dec_dms):
    sign = -1 if dec_dms[0] == '-' else 1
    d, m, s = [float(i) for i in dec_dms.split(':')]
    dec_deg = sign * (abs(d) + m / 60 + s / 3600)
    return dec_deg


def convert_to_decimal_degrees(ra_hms, dec_dms):
    coord = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg))
    return coord.ra.deg, coord.dec.deg

def convert_degree_to_hms_dms(ra_deg, dec_deg):
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    ra_hms = coord.ra.to_string(unit=u.hourangle, sep=':', precision=2)
    dec_dms = coord.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True)
    return ra_hms, dec_dms


if __name__ == '__main__':
    # Answer for the example below
    # RA,Dec
    ra, dec = '17:46:48.87','-33:42:06.9'
    # RA_deg,Dec_deg
    ra_deg, dec_deg = 266.70362499999993,-33.70191666666667

    print('HMS DMS TO DECIMAL DEGREE')
    print(f'TRUE VALUES ra_deg: {ra_deg}, dec_deg: {dec_deg}')
    print(convert_to_decimal_degrees(ra, dec))

    print('DECIMAL DEGREE TO HMS DMS')
    print(f'TRUE VALUES ra: {ra}, dec: {dec}')
    print(convert_degree_to_hms_dms(ra_deg, dec_deg))
