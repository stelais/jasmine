from jasmine.files_organizer.ra_and_dec_conversions import convert_to_decimal_degrees, convert_degree_to_hms_dms

def test_ra_conversion():
    # Answer for the example below
    # RA,Dec
    ra = '17:46:48.87'
    # RA_deg,Dec_deg
    ra_deg, dec_deg = 266.70362499999993,-33.70191666666667

    print('DECIMAL DEGREE TO HMS DMS')
    print(f'TRUE VALUES ra: {ra}')
    convert_ra, convert_dec = convert_degree_to_hms_dms(ra_deg, dec_deg)

    assert convert_ra == ra