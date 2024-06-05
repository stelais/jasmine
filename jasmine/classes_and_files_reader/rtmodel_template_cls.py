import pandas as pd
import numpy as np
from dataclasses import dataclass
import VBBinaryLensing
from moana import lens as mlens


@dataclass
class RTModelTemplateForBinaryLightCurve:
    """
    Generate a light curve based on a line of RTModel template
    """
    template_line: int
    separation_s: float
    mass_ratio_q: float
    impact_parameter_u0: float
    angle_alpha: float
    source_radius_rho: float
    pre_calculated_peak_tp1: float
    pre_calculated_peak_tp2: float
    einstein_time_tE: float
    peak_time_t0: float
    input_peak_t1: float
    input_peak_t2: float

    def __init__(self, *, template_line, path_to_template,
                 input_peak_t1, input_peak_t2):
        self.template_line = template_line
        self.path_to_template = path_to_template
        line_information = read_template(path_to_template).loc[template_line - 2]
        self.separation_s = line_information['s']
        self.mass_ratio_q = line_information['q']
        self.impact_parameter_u0 = line_information['u0']
        self.angle_alpha = line_information['alpha']
        self.source_radius_rho = line_information['rho']
        self.pre_calculated_peak_tp1 = line_information['tp1']
        self.pre_calculated_peak_tp2 = line_information['tp2']
        self.input_peak_t1 = input_peak_t1
        self.input_peak_t2 = input_peak_t2
        self.einstein_time_tE = tE_definer(t1=self.input_peak_t1,
                                           t2=self.input_peak_t2,
                                           tp1=self.pre_calculated_peak_tp1,
                                           tp2=self.pre_calculated_peak_tp2)
        self.peak_time_t0 = t0_definer(t1=self.input_peak_t1,
                                       tE=self.einstein_time_tE,
                                       tp1=self.pre_calculated_peak_tp1)
        self.results = None
        self.source_trajectory_y1 = None
        self.source_trajectory_y2 = None
        self.magnification = None
        self.times = None
        self.critical_curves = None
        self.caustics = None
        self.type_of_caustics = None

    def rtmodel_calculation_using_vbb(self, *, time_interval=None,
                                      parallax=False,
                                      orbital_motion=False,
                                      n_points=10000):
        """
        Calculate the light curve using the VBBinaryLensing package
        :param time_interval:
        :param parallax:
        :param orbital_motion:
        :param n_points:
        :return:
        """
        vbb = VBBinaryLensing.VBBinaryLensing()

        separation_s = self.separation_s
        mass_ratio_q = self.mass_ratio_q
        alpha = self.angle_alpha
        impact_parameter_u0 = self.impact_parameter_u0
        einstein_time_tE = self.einstein_time_tE
        peak_time_t0 = self.peak_time_t0
        rho = self.source_radius_rho
        if time_interval is None:
            time_interval = np.linspace(peak_time_t0 - 2 * einstein_time_tE,
                                        peak_time_t0 + 2 * einstein_time_tE,
                                        n_points)
        if parallax:
            print('Edit here to a collection of parameters with parallax')
        if orbital_motion:
            print('Edit here to a collection of parameters with orbital motion')
        pr = [np.log(separation_s), np.log(mass_ratio_q), impact_parameter_u0,
              alpha, np.log(rho), np.log(einstein_time_tE), peak_time_t0]
        self.results = vbb.BinaryLightCurve(pr, time_interval)
        self.times = time_interval
        return self.results

    def rtmodel_magnification_using_vbb(self, *,
                                        time_interval=None,
                                        parallax=False,
                                        orbital_motion=False,
                                        n_points=10000):
        """
        Calculate the magnification using the VBBinaryLensing package
        :param time_interval:
        :param parallax:
        :param orbital_motion:
        :param n_points:
        :return:
        """
        self.results = self.rtmodel_calculation_using_vbb(time_interval=time_interval,
                                                          parallax=parallax,
                                                          orbital_motion=orbital_motion,
                                                          n_points=n_points)

        self.magnification = self.results[0]
        return self.magnification, self.times

    def rtmodel_source_trajectory_using_vbb(self, *,
                                            time_interval=None,
                                            parallax=False,
                                            orbital_motion=False,
                                            n_points=10000):
        """
        Calculate the source trajectory using the VBBinaryLensing package
        :param time_interval:
        :param parallax:
        :param orbital_motion:
        :param n_points:
        :return:
        """
        if self.results is None:
            self.results = self.rtmodel_calculation_using_vbb(time_interval=time_interval,
                                                              parallax=parallax,
                                                              orbital_motion=orbital_motion,
                                                              n_points=n_points)
        self.source_trajectory_y1 = self.results[1]
        self.source_trajectory_y2 = self.results[2]
        return self.source_trajectory_y1, self.source_trajectory_y2

    def rtmodel_caustics_using_vbb(self):
        """
        Calculate the caustics using the VBBinaryLensing package
        :return:
        """
        vbb = VBBinaryLensing.VBBinaryLensing()
        caustics = vbb.Caustics(self.separation_s, self.mass_ratio_q)
        self.caustics = caustics
        return caustics

    def which_topology_from_moana(self):
        """
        Check the topology type of the binary lens system
        :return:
        """
        if self.separation_s < mlens.close_limit_2l(self.mass_ratio_q):
            self.type_of_caustics = 'close'
        elif self.separation_s <= mlens.wide_limit_2l(self.mass_ratio_q):
            self.type_of_caustics = 'ressonant'
        else:
            self.type_of_caustics = 'wide'
        return self.type_of_caustics

def tE_definer(*, t1, t2, tp1, tp2):
    """
    Equation 8 in Bozza 2024 https://arxiv.org/pdf/2405.04092
    :param t1: position of peak 1 in units of Einstein time
    :param t2: position of peak 2 in units of Einstein time
    :param tp1: pre-calculated position of peak 1 in units of Einstein time
    :param tp2: pre-calculated position of peak 2 in units of Einstein time
    :return:
    """
    tE = (t2 - t1) / (tp2 - tp1)
    return tE


def t0_definer(*, t1, tE, tp1):
    """
    Equation 9 in Bozza 2024 https://arxiv.org/pdf/2405.04092
    :param t1: position of peak 1 in units of Einstein time tE
    :param tE: Einstein time for that event, referred to function above
    :param tp1: pre-calculated position of peak 1 in units of Einstein time
    :return:
    """
    t0 = t1 - tE * tp1
    return t0


def read_template(path_to_template):
    template_df = pd.read_csv(path_to_template,
                              skiprows=1,
                              sep=r'\s+',
                              names=['s', 'q', 'u0', 'alpha', 'rho', 'tp1', 'tp2'],
                              )
    return template_df
