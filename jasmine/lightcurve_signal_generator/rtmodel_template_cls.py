import pandas as pd
from dataclasses import dataclass


@dataclass
class RTModelTemplateBinaryLightCurve:
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
        line_information = read_template(path_to_template).loc[template_line-2]
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
