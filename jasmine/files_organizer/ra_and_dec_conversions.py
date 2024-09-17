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
