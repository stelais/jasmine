from dataclasses import dataclass
import numpy as np


class ModelResults:
    def __init__(self, file_to_be_read, *, data_challenge_lc_number=None):
        self.model_type = file_to_be_read.split('/')[-1][0:2]
        self.data_challenge_lc_number = data_challenge_lc_number
        if self.model_type == 'PS':
            self.model_extensive_name = 'Single Lens Single Source (1L1S)'
            self.model_parameters = SingleLensSingleSourcePS(file_to_be_read)
        elif self.model_type == 'PX':
            self.model_extensive_name = 'Single Lens Single Source with Parallax (1L1S+plx)'
            self.model_parameters = SingleLensSingleSourceWithParallaxPX(file_to_be_read)
        elif self.model_type == 'BS':
            self.model_extensive_name = 'Single Lens Binary Source (1L2S)'
            self.model_parameters = SingleLensBinarySourceBS(file_to_be_read)
        elif self.model_type == 'BO':
            self.model_extensive_name = 'Single Lens Binary Source with Xallarap (1L2S+xlp)'
            self.model_parameters = SingleLensBinarySourceWithXallarapBO(file_to_be_read)
        elif self.model_type == 'LS':
            self.model_extensive_name = 'Binary Lens Single Source (2L1S)'
            self.model_parameters = BinaryLensSingleSourceLS(file_to_be_read)
        elif self.model_type == 'LX':
            self.model_extensive_name = 'Binary Lens Single Source with Parallax (2L1S+plx)'
            self.model_parameters = BinaryLensSingleSourceWithParallaxLX(file_to_be_read)
        elif self.model_type == 'LO':
            self.model_extensive_name = 'Binary Lens Single Source with Parallax and Orbital Motion (2L1S+plx+OM)'
            self.model_parameters = BinaryLensSingleSourceWithOrbitalMotionLO(file_to_be_read)
        else:
            raise ValueError(f"Model type {self.model_type} not recognized.")


@dataclass
class SingleLensSingleSourcePS:
    """
    PS	Single_lens_single_source
    4 parameters
    """
    number_of_parameters: int
    u0: float  # Impact parameter normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    u0_error: float
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    tE_error: float
    t0: float  # Closest approach time in HJD
    t0_error: float
    rho: float  # Source radius normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    rho_error: float
    chi2: float  # Chi2 value for the fit
    blends: tuple
    sources: tuple
    blending: np.array
    baseline: np.array


    def __init__(self, file_to_be_read):
        self.number_of_parameters = 4
        with open(file_to_be_read, 'r') as f:
            lines = f.readlines()
            parameters = lines[0].split(' ')
            errors = lines[1].split(' ')
            self.u0 = float(parameters[0])
            self.u0_error = float(errors[0])
            self.tE = float(parameters[1])
            self.tE_error = float(errors[1])
            self.t0 = float(parameters[2])
            self.t0_error = float(errors[2])
            self.rho = float(parameters[3])
            self.rho_error = float(errors[3])
            self.chi2 = float(parameters[-1])
            self.blends = float(parameters[4]), float(parameters[6])
            self.sources = float(parameters[5]), float(parameters[7])
            self.blending = np.array(self.blends) / np.array(self.sources)
            self.baseline = -2.5 * np.log10(np.array(self.blends) + np.array(self.sources))


@dataclass
class SingleLensSingleSourceWithParallaxPX:
    """
    PX	Single_lens_single_source with parallax
    6 parameters
    """
    number_of_parameters: int
    u0: float  # Impact parameter normalized to Einstein angle
    u0_error: float
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    tE_error: float
    t0: float  # Closest approach time in HJD
    t0_error: float
    rho: float  # Source radius normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    rho_error: float
    piN: float  # Parallax component along North
    piN_error: float
    piE: float  # Parallax component along East
    piE_error: float
    chi2: float  # Chi2 value for the fit
    blends: tuple
    sources: tuple
    blending: np.array
    baseline: np.array


    def __init__(self, file_to_be_read):
        self.number_of_parameters = 6
        with open(file_to_be_read, 'r') as f:
            lines = f.readlines()
            parameters = lines[0].split(' ')
            errors = lines[1].split(' ')
            self.u0 = float(parameters[0])
            self.u0_error = float(errors[0])
            self.tE = float(parameters[1])
            self.tE_error = float(errors[1])
            self.t0 = float(parameters[2])
            self.t0_error = float(errors[2])
            self.rho = float(parameters[3])
            self.rho_error = float(errors[3])
            self.piN = float(parameters[4])
            self.piN_error = float(errors[4])
            self.piE = float(parameters[5])
            self.piE_error = float(errors[5])
            self.chi2 = float(parameters[-1])
            self.blends = float(parameters[6]), float(parameters[8])
            self.sources = float(parameters[7]), float(parameters[9])
            self.blending = np.array(self.blends) / np.array(self.sources)
            self.baseline = -2.5 * np.log10(np.array(self.blends) + np.array(self.sources))


@dataclass
class SingleLensBinarySourceBS:
    """
    BS	Single_lens_binary_source
    7 parameters
    """
    number_of_parameters: int
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    tE_error: float
    flux_ratio: float  # Flux ratio of the secondary to the primary source (internally fit in logarithmic (ln) scale)
    flux_ratio_error: float
    u01: float  # Impact parameter normalized to Einstein angle for primary source
    u01_error: float
    u02: float  # Impact parameter of the secondary source
    u02_error: float
    t01: float  # Closest approach time of the primary source
    t01_error: float
    t02: float  # Closest approach time of the secondary source
    t02_error: float
    rho1: float  # Source radius for the primary source (internally fit in logarithmic (ln) scale)
    rho1_error: float
    chi2: float  # Chi2 value for the fit
    blends: tuple
    sources: tuple
    blending: np.array
    baseline: np.array

    def __init__(self, file_to_be_read):
        self.number_of_parameters = 7
        with open(file_to_be_read, 'r') as f:
            lines = f.readlines()
            parameters = lines[0].split(' ')
            errors = lines[1].split(' ')
            self.tE = float(parameters[0])
            self.tE_error = float(errors[0])
            self.flux_ratio = float(parameters[1])
            self.flux_ratio_error = float(errors[1])
            self.u01 = float(parameters[2])
            self.u01_error = float(errors[2])
            self.u02 = float(parameters[3])
            self.u02_error = float(errors[3])
            self.t01 = float(parameters[4])
            self.t01_error = float(errors[4])
            self.t02 = float(parameters[5])
            self.t02_error = float(errors[5])
            self.rho1 = float(parameters[6])
            self.rho1_error = float(errors[6])
            self.chi2 = float(parameters[-1])
            self.blends = float(parameters[7]), float(parameters[9])
            self.sources = float(parameters[8]), float(parameters[10])
            self.blending = np.array(self.blends) / np.array(self.sources)
            self.baseline = -2.5 * np.log10(np.array(self.blends) + np.array(self.sources))


@dataclass
class SingleLensBinarySourceWithXallarapBO:
    """
    BO	Single_lens_binary_source with xallarap
    10 parameters
    """
    number_of_parameters: int
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
    chi2: float  # Chi2 value for the fit
    blends: tuple
    sources: tuple
    blending: np.array
    baseline: np.array


    def __init__(self, file_to_be_read):
        self.number_of_parameters = 10
        with open(file_to_be_read, 'r') as f:
            lines = f.readlines()
            parameters = lines[0].split(' ')
            errors = lines[1].split(' ')
            self.u01 = float(parameters[0])
            self.u01_error = float(errors[0])
            self.t01 = float(parameters[1])
            self.t01_error = float(errors[1])
            self.tE = float(parameters[2])
            self.tE_error = float(errors[2])
            self.rho1 = float(parameters[3])
            self.rho1_error = float(errors[3])
            self.xi1 = float(parameters[4])
            self.xi1_error = float(errors[4])
            self.xi2 = float(parameters[5])
            self.xi2_error = float(errors[5])
            self.omega = float(parameters[6])
            self.omega_error = float(errors[6])
            self.inc = float(parameters[7])
            self.inc_error = float(errors[7])
            self.phi = float(parameters[8])
            self.phi_error = float(errors[8])
            self.qs = float(parameters[9])
            self.qs_error = float(errors[9])
            self.chi2 = float(parameters[-1])
            self.blends = float(parameters[10]), float(parameters[12])
            self.sources = float(parameters[11]), float(parameters[13])
            self.blending = np.array(self.blends) / np.array(self.sources)
            self.baseline = -2.5 * np.log10(np.array(self.blends) + np.array(self.sources))


@dataclass
class BinaryLensSingleSourceLS:
    """
    LS	Binary_lens_single_source
    7 parameters
    """
    number_of_parameters: int
    separation: float  # Separation between the lenses in Einstein radii (internally fit in logarithmic (ln) scale)
    separation_error: float
    mass_ratio: float  # Mass ratio of the secondary to the primary lens (internally fit in logarithmic (ln) scale)
    mass_ratio_error: float
    u0: float  # Impact parameter normalized to Einstein angle
    u0_error: float
    alpha: float  # Angle between the source velocity and the vector pointing from the secondary to the primary lens
    alpha_error: float
    rho: float  # Source radius normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    rho_error: float
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    tE_error: float
    t0: float  # Closest approach time in HJD to the barycenter
    t0_error: float
    chi2: float  # Chi2 value for the fit
    blends: tuple
    sources: tuple
    blending: np.array
    baseline: np.array

    def __init__(self, file_to_be_read):
        self.number_of_parameters = 7
        with open(file_to_be_read, 'r') as f:
            lines = f.readlines()
            parameters = lines[0].split(' ')
            errors = lines[1].split(' ')
            self.separation = float(parameters[0])
            self.separation_error = float(errors[0])
            self.mass_ratio = float(parameters[1])
            self.mass_ratio_error = float(errors[1])
            self.u0 = float(parameters[2])
            self.u0_error = float(errors[2])
            self.alpha = float(parameters[3])
            self.alpha_error = float(errors[3])
            self.rho = float(parameters[4])
            self.rho_error = float(errors[4])
            self.tE = float(parameters[5])
            self.tE_error = float(errors[5])
            self.t0 = float(parameters[6])
            self.t0_error = float(errors[6])
            self.chi2 = float(parameters[-1])
            self.blends = float(parameters[7]), float(parameters[9])
            self.sources = float(parameters[8]), float(parameters[10])
            self.blending = np.array(self.blends) / np.array(self.sources)
            self.baseline = -2.5 * np.log10(np.array(self.blends) + np.array(self.sources))


@dataclass
class BinaryLensSingleSourceWithParallaxLX:
    """
    LX	Binary_lens_single_source with parallax
    9 parameters
    """
    number_of_parameters: int
    separation: float  # Separation between the lenses in Einstein radii (internally fit in logarithmic (ln) scale)
    separation_error: float
    mass_ratio: float  # Mass ratio of the secondary to the primary lens (internally fit in logarithmic (ln) scale)
    mass_ratio_error: float
    u0: float  # Impact parameter normalized to Einstein angle
    u0_error: float
    alpha: float  # Angle between the source velocity and the vector pointing from the secondary to the primary lens
    alpha_error: float
    rho: float  # Source radius normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    rho_error: float
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    tE_error: float
    t0: float  # Closest approach time in HJD to the barycenter
    t0_error: float
    piN: float  # Parallax component along North
    piN_error: float
    piE: float  # Parallax component along East
    piE_error: float
    chi2: float  # Chi2 value for the fit
    blends: tuple
    sources: tuple
    blending: np.array
    baseline: np.array

    def __init__(self, file_to_be_read):
        self.number_of_parameters = 9
        with open(file_to_be_read, 'r') as f:
            lines = f.readlines()
            parameters = lines[0].split(' ')
            errors = lines[1].split(' ')
            self.separation = float(parameters[0])
            self.separation_error = float(errors[0])
            self.mass_ratio = float(parameters[1])
            self.mass_ratio_error = float(errors[1])
            self.u0 = float(parameters[2])
            self.u0_error = float(errors[2])
            self.alpha = float(parameters[3])
            self.alpha_error = float(errors[3])
            self.rho = float(parameters[4])
            self.rho_error = float(errors[4])
            self.tE = float(parameters[5])
            self.tE_error = float(errors[5])
            self.t0 = float(parameters[6])
            self.t0_error = float(errors[6])
            self.piN = float(parameters[7])
            self.piN_error = float(errors[7])
            self.piE = float(parameters[8])
            self.piE_error = float(errors[8])
            self.chi2 = float(parameters[-1])
            self.blends = float(parameters[9]), float(parameters[11])
            self.sources = float(parameters[10]), float(parameters[12])
            self.blending = np.array(self.blends) / np.array(self.sources)
            self.baseline = -2.5 * np.log10(np.array(self.blends) + np.array(self.sources))


@dataclass
class BinaryLensSingleSourceWithOrbitalMotionLO:
    """
    LO	Binary_lens_single_source with orbital motion
    12 parameters
    """
    number_of_parameters: int
    separation: float  # Separation between the lenses in Einstein radii (internally fit in logarithmic (ln) scale)
    separation_error: float
    mass_ratio: float  # Mass ratio of the secondary to the primary lens (internally fit in logarithmic (ln) scale)
    mass_ratio_error: float
    u0: float  # Impact parameter normalized to Einstein angle
    u0_error: float
    alpha: float  # Angle between the source velocity and the vector pointing from the secondary to the primary lens
    alpha_error: float
    rho: float  # Source radius normalized to Einstein angle (internally fit in logarithmic (ln) scale)
    rho_error: float
    tE: float  # Einstein time in days (internally fit in logarithmic (ln) scale)
    tE_error: float
    t0: float  # Closest approach time in HJD to the barycenter
    t0_error: float
    piN: float  # Parallax component along North
    piN_error: float
    piE: float  # Parallax component along East
    piE_error: float
    gamma1: float  # Angular velocity parallel to the lens axis
    gamma1_error: float
    gamma2: float  # Angular velocity perpendicular to the lens axis
    gamma2_error: float
    gammaz: float  # Angular velocity along the line of sight
    gammaz_error: float
    chi2: float  # Chi2 value for the fit
    blends: tuple
    sources: tuple
    blending: np.array
    baseline: np.array


    def __init__(self, file_to_be_read):
        self.number_of_parameters = 12
        with open(file_to_be_read, 'r') as f:
            lines = f.readlines()
            parameters = lines[0].split(' ')
            errors = lines[1].split(' ')
            self.separation = float(parameters[0])
            self.separation_error = float(errors[0])
            self.mass_ratio = float(parameters[1])
            self.mass_ratio_error = float(errors[1])
            self.u0 = float(parameters[2])
            self.u0_error = float(errors[2])
            self.alpha = float(parameters[3])
            self.alpha_error = float(errors[3])
            self.rho = float(parameters[4])
            self.rho_error = float(errors[4])
            self.tE = float(parameters[5])
            self.tE_error = float(errors[5])
            self.t0 = float(parameters[6])
            self.t0_error = float(errors[6])
            self.piN = float(parameters[7])
            self.piN_error = float(errors[7])
            self.piE = float(parameters[8])
            self.piE_error = float(errors[8])
            self.gamma1 = float(parameters[9])
            self.gamma1_error = float(errors[9])
            self.gamma2 = float(parameters[10])
            self.gamma2_error = float(errors[10])
            self.gammaz = float(parameters[11])
            self.gammaz_error = float(errors[11])
            self.chi2 = float(parameters[-1])
            self.blends = float(parameters[12]), float(parameters[14])
            self.sources = float(parameters[13]), float(parameters[15])
            self.blending = np.array(self.blends) / np.array(self.sources)
            self.baseline = -2.5 * np.log10(np.array(self.blends) + np.array(self.sources))
