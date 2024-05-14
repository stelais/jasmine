from dataclasses import dataclass
import pandas as pd


class ModelResults:
    def __init__(self, model_type, data_challenge_lc_number, file_to_be_read):
        self.model_type = model_type
        self.data_challenge_lc_number = data_challenge_lc_number
        self.chi2 = file_to_be_read[3]
        model_df = pd.read_csv(f"{model_type}*", sep=' ', header=None)
        if model_type == 'PS':
            self.model = SingleLensSingleSourcePS(model_df)
        elif model_type == 'PX':
            self.model = SingleLensSingleSourceWithParallaxPX(model_df)
        elif model_type == 'BS':
            self.model = SingleLensBinarySourceBS(model_df)
        elif model_type == 'BO':
            self.model = SingleLensBinarySourceWithXallarapBO(model_df)
        elif model_type == 'LS':
            self.model = BinaryLensSingleSourceLS(model_df)
        elif model_type == 'LX':
            self.model = BinaryLensSingleSourceWithParallaxLX(model_df)
        elif model_type == 'LO':
            self.model = BinaryLensSingleSourceWithOrbitalMotionLO(model_df)
        else:
            raise ValueError(f"Model type {model_type} not recognized.")



@dataclass
class SingleLensSingleSourcePS:
    """
    PS	Single_lens_single_source
    4 parameters
    TODO - not ready. file to be read not specified
    """
    u0: float  # Impact parameter normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    t0: float  # Closest approach time in HJD
    rho: float  # Source radius normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    number_of_parameters = 4

    def __init__(self, file_to_be_read):
        self.u0 = file_to_be_read[0]
        self.tE = file_to_be_read[1]
        self.t0 = file_to_be_read[2]
        self.rho = file_to_be_read[3]


@dataclass
class SingleLensSingleSourceWithParallaxPX:
    """
    PX	Single_lens_single_source with parallax
    6 parameters
    TODO - not ready. file to be read not specified
    """
    u0: float  # Impact parameter normalized to Einstein angle
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    t0: float  # Closest approach time in HJD
    rho: float  # Source radius normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    piN: float  # Parallax component along North
    piE: float  # Parallax component along East
    number_of_parameters = 6

    def __init__(self, file_to_be_read):
        self.u0 = file_to_be_read[0]
        self.tE = file_to_be_read[1]
        self.t0 = file_to_be_read[2]
        self.rho = file_to_be_read[3]
        self.piN = file_to_be_read[4]
        self.piE = file_to_be_read[5]


@dataclass
class SingleLensBinarySourceBS:
    """
    BS	Single_lens_binary_source
    7 parameters
    TODO - not ready. file to be read not specified
    """
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    flux_ratio: float  # Flux ratio of the secondary to the primary source (internally fit in logarithmic (ln) scale)
    u01: float  # Impact parameter of the primary source
    u02: float  # Impact parameter of the secondary source
    t01: float  # Closest approach time of the primary source
    t02: float  # Closest approach time of the secondary source
    rho1: float  # Source radius for the primary source (internally fit in logarithmic (ln) scale)
    number_of_parameters = 7

    def __init__(self, file_to_be_read):
        self.tE = file_to_be_read[0]
        self.flux_ratio = file_to_be_read[1]
        self.u01 = file_to_be_read[2]
        self.u02 = file_to_be_read[3]
        self.t01 = file_to_be_read[4]
        self.t02 = file_to_be_read[5]
        self.rho1 = file_to_be_read[6]


@dataclass
class SingleLensBinarySourceWithXallarapBO:
    """
    BO	Single_lens_binary_source with xallarap
    10 parameters
    TODO - not ready. file to be read not specified
    """
    u01: float  # Impact parameter of the primary source
    t01: float  # Closest approach time of the primary source
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    rho1: float  # Source radius for the primary source (internally fit in logarithmic (ln) scale)
    xi1: float  # Xallarap component parallel to the source velocity
    xi2: float  # Xallarap component orthogonal to the source velocity
    omega: float  # Orbital angular velocity in days^-1
    inc: float  # Inclination of the orbital plane in radians
    phi: float  # Phase of the orbit from the passage on the line of nodes
    qs: float  # Mass ratio of the secondary to the primary source
    number_of_parameters = 10

    def __init__(self, file_to_be_read):
        self.u01 = file_to_be_read[0]
        self.t01 = file_to_be_read[1]
        self.tE = file_to_be_read[2]
        self.rho1 = file_to_be_read[3]
        self.xi1 = file_to_be_read[4]
        self.xi2 = file_to_be_read[5]
        self.omega = file_to_be_read[6]
        self.inc = file_to_be_read[7]
        self.phi = file_to_be_read[8]
        self.qs = file_to_be_read[9]


@dataclass
class BinaryLensSingleSourceLS:
    """
    LS	Binary_lens_single_source
    7 parameters
    TODO - not ready. file to be read not specified
    """
    separation: float  # Separation between the lenses in Einstein radii (internally fit in logarithmic (ln) scale)
    mass_ratio: float # Mass ratio of the secondary to the primary lens (internally fit in logarithmic (ln) scale)
    u0: float  # Impact parameter normalized to Einstein angle
    alpha: float  # Angle between the source velocity and the vector pointing from the secondary to the primary lens
    rho: float  # Source radius normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    t0: float  # Closest approach time in HJD to the barycenter
    number_of_parameters = 7

    def __init__(self, file_to_be_read):
        self.separation = file_to_be_read[0]
        self.mass_ratio = file_to_be_read[1]
        self.u0 = file_to_be_read[2]
        self.alpha = file_to_be_read[3]
        self.rho = file_to_be_read[4]
        self.tE = file_to_be_read[5]
        self.t0 = file_to_be_read[6]

@dataclass
class BinaryLensSingleSourceWithParallaxLX:
    """
    LX	Binary_lens_single_source with parallax
    9 parameters
    TODO - not ready. file to be read not specified
    """
    separation: float  # Separation between the lenses in Einstein radii (internally fit in logarithmic (ln) scale)
    mass_ratio: float  # Mass ratio of the secondary to the primary lens (internally fit in logarithmic (ln) scale)
    u0: float  # Impact parameter normalized to Einstein angle
    alpha: float  # Angle between the source velocity and the vector pointing from the secondary to the primary lens
    rho: float  # Source radius normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    t0: float  # Closest approach time in HJD to the barycenter
    piN: float  # Parallax component along North
    piE: float  # Parallax component along East
    number_of_parameters = 9

    def __init__(self, file_to_be_read):
        self.separation = file_to_be_read[0]
        self.mass_ratio = file_to_be_read[1]
        self.u0 = file_to_be_read[2]
        self.alpha = file_to_be_read[3]
        self.rho = file_to_be_read[4]
        self.tE = file_to_be_read[5]
        self.t0 = file_to_be_read[6]
        self.piN = file_to_be_read[7]
        self.piE = file_to_be_read[8]


@dataclass
class BinaryLensSingleSourceWithOrbitalMotionLO:
    """
    LO	Binary_lens_single_source with orbital motion
    12 parameters
    TODO - not ready. file to be read not specified
    """
    separation: float   # Separation between the lenses in Einstein radii (internally fit in logarithmic (ln) scale)
    mass_ratio: float   # Mass ratio of the secondary to the primary lens (internally fit in logarithmic (ln) scale)
    u0: float  # Impact parameter normalized to Einstein angle
    alpha: float  # Angle between the source velocity and the vector pointing from the secondary to the primary lens
    rho: float  # Source radius normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    t0: float  # Closest approach time in HJD to the barycenter
    piN: float  # Parallax component along North
    piE: float  # Parallax component along East
    gamma1: float  # Angular velocity parallel to the lens axis
    gamma2: float  # Angular velocity perpendicular to the lens axis
    gammaz: float  # Angular velocity along the line of sight
    number_of_parameters = 12

    def __init__(self, file_to_be_read):
        self.separation = file_to_be_read[0]
        self.mass_ratio = file_to_be_read[1]
        self.u0 = file_to_be_read[2]
        self.alpha = file_to_be_read[3]
        self.rho = file_to_be_read[4]
        self.tE = file_to_be_read[5]
        self.t0 = file_to_be_read[6]
        self.piN = file_to_be_read[7]
        self.piE = file_to_be_read[8]
        self.gamma1 = file_to_be_read[9]
        self.gamma2 = file_to_be_read[10]
        self.gammaz = file_to_be_read[11]
