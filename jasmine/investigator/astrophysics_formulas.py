import numpy as np
from jasmine.constants import general_physics_constants as astroconstants


def absolute_pi_E_calculator(pi_E_N_, pi_E_E_,
                             pi_E_N_error_=None, pi_E_E_error_=None,
                             is_error_included_=False):
    abs_piE = np.sqrt(pi_E_N_ ** 2 + pi_E_E_ ** 2)
    if is_error_included_:
        abs_piE_error = np.sqrt(((pi_E_N_ * pi_E_N_error_) ** 2 + (pi_E_E_ * pi_E_E_error_) ** 2) / (abs_piE ** 2))
        return abs_piE, abs_piE_error
    else:
        return abs_piE


def lens_mass__solar_mass_calculator(theta_E_, pi_E_,
                                     theta_E_error_=None, pi_E_error_=None,
                                     is_error_included_=False):
    kappa = astroconstants.kappa_mas_over_solar_mass
    lens_mass = theta_E_ / (kappa * pi_E_)
    if is_error_included_:
        lens_mass_error = np.sqrt((theta_E_error_ ** 2 / ((kappa ** 2) * (pi_E_ ** 2))) +
                                  (((theta_E_ ** 2) * (pi_E_error_ ** 2)) / ((kappa ** 2) * (pi_E_ ** 4))))
        return lens_mass, lens_mass_error
    else:
        return lens_mass


def planet_mass__earth_mass_calculator(mass_ratio_, lens_mass_solar_mass_,
                                       mass_ratio_error_=None, lens_mass_solar_mass_error_=None,
                                       is_error_included_=False):
    planet_mass__solar_mass = (lens_mass_solar_mass_ * mass_ratio_) / (1 + mass_ratio_)
    planet_mass__earth_mass = planet_mass__solar_mass * astroconstants.solar_mass__earth_mass
    if is_error_included_:
        partial_ML = mass_ratio_ / (1 + mass_ratio_)
        partial_q = lens_mass_solar_mass_ / (1 + mass_ratio_) ** 2

        planet_mass_error = np.sqrt(
            (partial_ML * lens_mass_solar_mass_error_) ** 2 +
            (partial_q * mass_ratio_error_) ** 2
        )
        planet_mass_error__earth_mass = planet_mass_error * astroconstants.solar_mass__earth_mass
        return planet_mass__earth_mass, planet_mass_error__earth_mass

    else:
        return planet_mass__earth_mass
