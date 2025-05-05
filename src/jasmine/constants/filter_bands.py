"""
Wide-Field Instrument https://roman.gsfc.nasa.gov/science/WFI_technical.html
"""
from dataclasses import dataclass


@dataclass
class WFI_filter_specifications:
    filter_name: str
    min_micrometer: float
    max_micrometer: float
    center_micrometer: float
    width_micrometer: float
    corner_PSF_FWHM_arcsec: float
    corner_n_eff_pixel: float
    corner_peak_flux: float
    center_n_eff_pixel: float
    center_peak_flux: float

    def __init__(self, filter_name_):
        self.filter_name = filter_name_
        self.filter_properties = filter_properties_getter(self.filter_name)
        self.min_micrometer = self.filter_properties[0]
        self.max_micrometer = self.filter_properties[1]
        self.center_micrometer = self.filter_properties[2]
        self.width_micrometer = self.filter_properties[3]
        self.corner_PSF_FWHM_arcsec = self.filter_properties[4]
        self.corner_n_eff_pixel = self.filter_properties[5]
        self.corner_peak_flux = self.filter_properties[6]
        self.center_n_eff_pixel = self.filter_properties[7]
        self.center_peak_flux = self.filter_properties[8]


def filter_properties_getter(filter_name_):
    if filter_name_ == 'F062':
        filter_properties = [0.48, 0.76, 0.620, 0.280, 0.058, 7.35, 0.20341, 3.80, 0.49536]
    elif filter_name_ == 'F087':
        filter_properties = [0.76, 0.977, 0.869, 0.217, 0.073, 9.35, 0.16517, 4.04, 0.4838]
    elif filter_name_ == 'F106':
        filter_properties = [0.927, 1.192, 1.060, 0.265, 0.087, 10.96, 0.15060, 4.83, 0.44004]
    elif filter_name_ == 'F129':
        filter_properties = [1.131, 1.454, 1.293, 0.323, 0.106, 11.79, 0.14863, 6.63, 0.36874]
    elif filter_name_ == 'F158':
        filter_properties = [1.380, 1.774, 1.577, 0.394, 0.128, 12.63, 0.14343, 9.65, 0.29081]
    elif filter_name_ == 'F184':
        filter_properties = [1.683, 2.000, 1.842, 0.317, 0.146, 17.15, 0.11953, 15.52, 0.21361]
    elif filter_name_ == 'F213':
        filter_properties = [1.95, 2.30, 2.125, 0.35, 0.169, 20.38, 0.10831, 20.14, 0.17052]
    elif filter_name_ == 'F146':
        filter_properties = [0.927, 2.000, 1.464, 1.030, 0.105, 12.18, 0.14521, 7.37, 0.34546]
    else:
        raise ValueError(f'filter_name should be F062, F087, F106, F129, F158, F184, F213, F146')
    return filter_properties
