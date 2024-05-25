from jasmine import LightcurveEventDataChallenge
import numpy as np
import VBBinaryLensing


def datachallenge_bound_planet_magnification_using_vbb(lightcurve_, *,
                                                       time_interval=None,
                                                       parallax=False,
                                                       orbital_motion=False):
    """
    Note that parallax is not added yet
    TODO: calculate the parallax and OM
    :param lightcurve_:
    :param time_interval:
    :param parallax:
    :param orbital_motion:
    :return:
    """

    vbb = VBBinaryLensing.VBBinaryLensing()

    separation_s = lightcurve_.planet.planet_separation
    mass_ratio_q = lightcurve_.planet.planet_mass_ratio
    alpha = lightcurve_.microlensing.microlensing_event_angle_alpha__deg
    impact_parameter_u0 = lightcurve_.microlensing.microlensing_impact_parameter_u0
    einstein_time_tE = lightcurve_.microlensing.microlensing_einstein_crossing_time_tE__days
    peak_time_t0 = lightcurve_.microlensing.microlensing_time_of_peak_t0__days
    rho = lightcurve_.microlensing.microlensing_source_radius_in_einstein_radii_rho
    if time_interval is None:
        time_interval = np.linspace(peak_time_t0 - 2 * einstein_time_tE,
                                    peak_time_t0 + 2 * einstein_time_tE,
                                    100000)
    if parallax:
        print('Edit here to a collection of parameters with parallax')
    if orbital_motion:
        print('Edit here to a collection of parameters with orbital motion')
    pr = [np.log(separation_s), np.log(mass_ratio_q), impact_parameter_u0,
          alpha, np.log(rho), np.log(einstein_time_tE), peak_time_t0]
    results = vbb.BinaryLightCurve(pr, time_interval)

    return results[0], time_interval


def easy_plot_lightcurve(lightcurve_):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    magnification, time_interval = datachallenge_bound_planet_magnification_using_vbb(lightcurve_)
    ax.scatter(time_interval, magnification, s=0.8, alpha=0.3)
    plt.show(dpi=600)


if __name__ == '__main__':
    template_path = '/Users/stela/Documents/Scripts/RTModel_project/RTModel/RTModel/data/TemplateLibrary.txt'
    lightcurve = LightcurveEventDataChallenge(data_challenge_lc_number=267,
                                              data_folder='/Users/stela/Documents/Scripts/RTModel_project/datachallenge')

    datachallenge_bound_planet_magnification_using_vbb(lightcurve)
    easy_plot_lightcurve(lightcurve)

