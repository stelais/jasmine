from jasmine import RTModelTemplateBinaryLightCurve
import numpy as np
import VBBinaryLensing


def rtmodel_magnification_using_vbb(lightcurve_, *,
                                    time_interval=None,
                                    parallax=False,
                                    orbital_motion=False):

    vbb = VBBinaryLensing.VBBinaryLensing()

    separation_s = lightcurve_.separation_s
    mass_ratio_q = lightcurve_.mass_ratio_q
    alpha = lightcurve_.angle_alpha
    impact_parameter_u0 = lightcurve_.impact_parameter_u0
    einstein_time_tE = lightcurve_.einstein_time_tE
    peak_time_t0 = lightcurve_.peak_time_t0
    rho = lightcurve_.source_radius_rho
    if time_interval is None:
        time_interval = np.linspace(peak_time_t0 - 2 * einstein_time_tE,
                                    peak_time_t0 + 2 * einstein_time_tE,
                                    10000)
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
    magnification, time_interval = rtmodel_magnification_using_vbb(lightcurve_)
    ax.scatter(time_interval, magnification, s=0.8, alpha=0.3)
    plt.show(dpi=600)


if __name__ == '__main__':
    template_path = '/Users/stela/Documents/Scripts/RTModel_project/RTModel/RTModel/data/TemplateLibrary.txt'
    lightcurve = RTModelTemplateBinaryLightCurve(template_line=2,
                                                 path_to_template=template_path,
                                                 input_peak_t1=300,
                                                 input_peak_t2=302)
    easy_plot_lightcurve(lightcurve)

