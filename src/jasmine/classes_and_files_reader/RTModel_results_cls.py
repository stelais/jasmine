from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
import re
import math
import warnings


class EventResults:
    """
    Class to store results (summary) of the RTModel fitting for a single event. It reads the Nature.txt file and also extracts
    the best fit names of each class of models (PS, PX, BS, BO, LS, LX, LO) and their chi2 values.
    """

    def __init__(self, event_folder_):
        """ Initialize the EventResults class with the event folder path."""
        self.event_folder = event_folder_
        self.event_name = event_folder_.split('/')[-1]

        # Initialize other attributes that need function below
        self.ps_best = BestModel()
        self.px_best = BestModel()
        self.bs_best = BestModel()
        self.bo_best = BestModel()
        self.ls_best = BestModel()
        self.lx_best = BestModel()
        self.lo_best = BestModel()
        self.complete_classification = None
        self.final_models = None
        self.number_of_final_models = None
        self.found_a_planet = None
        self.found_a_2L1S_solution = None

    def extract_information_from_nature_file(self):
        """Extracts information from the Nature.txt file in the event folder."""
        nature_file = os.path.join(self.event_folder, "Nature.txt")
        with open(nature_file, "r") as f:
            lines = f.readlines()
            content = ''.join(lines)
        # Extract the line starting with 'Successful:'
        self.complete_classification = ""
        match = re.search(r"Successful:\s*(.*)", content)
        if match:
            self.complete_classification = match.group(1).strip()
        self.found_a_planet = "Planetary lens" in self.complete_classification

        # Find the start of the 'chisquare model' section
        try:
            start = next(i for i, line in enumerate(lines) if "chisquare" in line)
        except StopIteration:
            start = None
        # Extract the best fit names
        self.final_models = []
        if start is not None:
            for line in lines[start + 1:]:
                parts = line.split()
                if len(parts) != 2 or not parts[1].endswith(".txt"):
                    break
                self.final_models.append(parts[1].replace(".txt", ""))
        self.number_of_final_models = len(self.final_models)

        # Check if any model starts with 'L'
        self.found_a_2L1S_solution = any(m.startswith("L") for m in self.final_models)

        # Extract chi2 values
        # Initialize a dictionary to hold chi2 values
        self.best_chi2 = {}
        # Pattern for the models you want
        chi2_pattern = re.compile(r"^(PS|PX|BS|BO|LS|LX|LO)\s*[:=]\s*([\d.eE+-]+)")
        for line in lines:
            match = chi2_pattern.search(line)
            if match:
                model, value = match.groups()
                self.best_chi2[model] = float(value)


    def looking_for_the_names_of_best_model_of_each_category(self):
        """Looks for the names of the best model of each category in the alternative models."""
        files = [f for f in os.listdir(self.event_folder + '/Models') if f.endswith(".txt")]

        for file_name in files:
            # Load model and get chi2
            model_name = file_name.split('.')[0]
            category = model_name[0:2]  # Get the first two characters to determine the category
            model_path = os.path.join(self.event_folder + '/Models', file_name)
            model = ModelResults(model_path)
            chi2 = model.model_parameters.chi2

            best_object = getattr(self, f"{category.lower()}_best")
            best_object.chi2 = self.best_chi2.get(category)

            if math.isclose(chi2, self.best_chi2[category], rel_tol=1e-5):
                best_object.chi2 = chi2
                best_object.name = model_name
            elif chi2 < self.best_chi2[category]:
                warnings.warn(
                    f"Chi2 for {file_name} ({chi2}) differs from existing {category} self.best_chi2[category] ({self.best_chi2[category]})\n"
                    f"It probably means that the chi2 value in the Nature.txt file is not the same as the one in the Model folder."
                )
                # If the chi2 is better than the current best, update the best object
                best_object.chi2 = chi2
                best_object.name = model_name


@dataclass
class BestModel:
    def __init__(self, name=None, chi2=None):
        self.name = name
        self.chi2 = chi2


class ModelResults:
    def __init__(self, file_to_be_read, *, data_challenge_lc_number=None):
        self.model_type = file_to_be_read.split('/')[-1][0:2]
        self.data_challenge_lc_number = data_challenge_lc_number
        if self.model_type == 'PS':
            self.model_type_extensive_name = 'Single Lens Single Source (1L1S)'
            self.model_parameters = SingleLensSingleSourcePS(file_to_be_read)
        elif self.model_type == 'PX':
            self.model_type_extensive_name = 'Single Lens Single Source with Parallax (1L1S+plx)'
            self.model_parameters = SingleLensSingleSourceWithParallaxPX(file_to_be_read)
        elif self.model_type == 'BS':
            self.model_type_extensive_name = 'Single Lens Binary Source (1L2S)'
            self.model_parameters = SingleLensBinarySourceBS(file_to_be_read)
        elif self.model_type == 'BO':
            self.model_type_extensive_name = 'Single Lens Binary Source with Xallarap (1L2S+xlp)'
            self.model_parameters = SingleLensBinarySourceWithXallarapBO(file_to_be_read)
        elif self.model_type == 'LS':
            self.model_type_extensive_name = 'Binary Lens Single Source (2L1S)'
            self.model_parameters = BinaryLensSingleSourceLS(file_to_be_read)
        elif self.model_type == 'LX':
            self.model_type_extensive_name = 'Binary Lens Single Source with Parallax (2L1S+plx)'
            self.model_parameters = BinaryLensSingleSourceWithParallaxLX(file_to_be_read)
        elif self.model_type == 'LO':
            self.model_type_extensive_name = 'Binary Lens Single Source with Parallax and Orbital Motion (2L1S+plx+OM)'
            self.model_parameters = BinaryLensSingleSourceWithOrbitalMotionLO(file_to_be_read)
        else:
            raise ValueError(f"Model type {self.model_type} not recognized.")
        self.covariance_matrix = pd.read_csv(file_to_be_read, skiprows=2, header=None, sep=r'\s+')


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
    blends: float
    sources: float
    blendings: float
    blendings_error: float
    baselines: float
    baselines_error: float

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

            self.blends = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters, len(parameters) - 1, 2)])
            blends_error = [float(errors[i]) for i in range(self.number_of_parameters, len(errors), 2)]
            self.sources = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters + 1, len(parameters) - 1, 2)])
            sources_error = [float(errors[i]) for i in range(self.number_of_parameters + 1, len(errors), 2)]

            # TODO CHECK THIS
            self.blendings = self.blends / (self.sources + 1.e-12 * self.blends)
            self.blendings_error = blends_error
            self.baselines = -2.5 * np.log10(self.blends + self.sources)
            self.baselines_error = sources_error


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
    piEN: float  # Parallax component along North
    piEN_error: float
    piEE: float  # Parallax component along East
    piEE_error: float
    chi2: float  # Chi2 value for the fit
    blends: float
    sources: float
    blendings: float
    blendings_error: float
    baselines: float
    baselines_error: float

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
            self.piEN = float(parameters[4])
            self.piEN_error = float(errors[4])
            self.piEE = float(parameters[5])
            self.piEE_error = float(errors[5])
            self.chi2 = float(parameters[-1])

            self.blends = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters, len(parameters) - 1, 2)])
            blends_error = [float(errors[i]) for i in range(self.number_of_parameters, len(errors), 2)]
            self.sources = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters + 1, len(parameters) - 1, 2)])
            sources_error = [float(errors[i]) for i in range(self.number_of_parameters + 1, len(errors), 2)]

            # TODO CHECK THIS
            self.blendings = self.blends / (self.sources + 1.e-12 * self.blends)
            self.blendings_error = blends_error
            self.baselines = -2.5 * np.log10(self.blends + self.sources)
            self.baselines_error = sources_error


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
    blends: float
    sources: float
    blendings: float
    blendings_error: float
    baselines: float
    baselines_error: float

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

            self.blends = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters, len(parameters) - 1, 2)])
            blends_error = [float(errors[i]) for i in range(self.number_of_parameters, len(errors), 2)]
            self.sources = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters + 1, len(parameters) - 1, 2)])
            sources_error = [float(errors[i]) for i in range(self.number_of_parameters + 1, len(errors), 2)]

            # TODO CHECK THIS
            self.blendings = self.blends / (self.sources + 1.e-12 * self.blends)
            self.blendings_error = blends_error
            self.baselines = -2.5 * np.log10(self.blends + self.sources)
            self.baselines_error = sources_error


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
    blends: float
    sources: float
    blendings: float
    blendings_error: float
    baselines: float
    baselines_error: float

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

            self.blends = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters, len(parameters) - 1, 2)])
            blends_error = [float(errors[i]) for i in range(self.number_of_parameters, len(errors), 2)]
            self.sources = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters + 1, len(parameters) - 1, 2)])
            sources_error = [float(errors[i]) for i in range(self.number_of_parameters + 1, len(errors), 2)]

            # TODO CHECK THIS
            self.blendings = self.blends / (self.sources + 1.e-12 * self.blends)
            self.blendings_error = blends_error
            self.baselines = -2.5 * np.log10(self.blends + self.sources)
            self.baselines_error = sources_error


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
    blends: float
    sources: float
    blendings: float
    blendings_error: float
    baselines: float
    baselines_error: float

    def __init__(self, file_to_be_read):
        self.number_of_parameters = 7
        with (open(file_to_be_read, 'r') as f):
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

            # blends are odd indices starting from 7
            self.blends = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters, len(parameters) - 1, 2)])
            blends_error = [float(errors[i]) for i in range(self.number_of_parameters, len(errors), 2)]
            # sources are even indices starting from 8
            self.sources = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters + 1, len(parameters) - 1, 2)])
            sources_error = [float(errors[i]) for i in range(self.number_of_parameters + 1, len(errors), 2)]

            # TODO CHECK THIS
            self.blendings = self.blends / (self.sources + 1.e-12 * self.blends)
            self.blendings_error = blends_error
            self.baselines = -2.5 * np.log10(self.blends + self.sources)
            self.baselines_error = sources_error


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
    piEN: float  # Parallax component along North
    piEN_error: float
    piEE: float  # Parallax component along East
    piEE_error: float
    chi2: float  # Chi2 value for the fit
    blends: float
    sources: float
    blendings: float
    blendings_error: float
    baselines: float
    baselines_error: float

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
            self.piEN = float(parameters[7])
            self.piEN_error = float(errors[7])
            self.piEE = float(parameters[8])
            self.piEE_error = float(errors[8])
            self.chi2 = float(parameters[-1])

            # blends are odd indices starting from 9
            self.blends = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters, len(parameters) - 1, 2)])
            blends_error = [float(errors[i]) for i in range(self.number_of_parameters, len(errors), 2)]
            # sources are even indices starting from 10
            self.sources = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters + 1, len(parameters) - 1, 2)])
            sources_error = [float(errors[i]) for i in range(self.number_of_parameters + 1, len(errors), 2)]

            # TODO CHECK THIS
            self.blendings = self.blends / (self.sources + 1.e-12 * self.blends)
            self.blendings_error = blends_error
            self.baselines = -2.5 * np.log10(self.blends + self.sources)
            self.baselines_error = sources_error


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
    piEN: float  # Parallax component along North
    piEN_error: float
    piEE: float  # Parallax component along East
    piEE_error: float
    gamma1: float  # Angular velocity parallel to the lens axis
    gamma1_error: float
    gamma2: float  # Angular velocity perpendicular to the lens axis
    gamma2_error: float
    gammaz: float  # Angular velocity along the line of sight
    gammaz_error: float
    chi2: float  # Chi2 value for the fit
    blends: float
    sources: float
    blendings: float
    blendings_error: float
    baselines: float
    baselines_error: float

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
            self.piEN = float(parameters[7])
            self.piEN_error = float(errors[7])
            self.piEE = float(parameters[8])
            self.piEE_error = float(errors[8])
            self.gamma1 = float(parameters[9])
            self.gamma1_error = float(errors[9])
            self.gamma2 = float(parameters[10])
            self.gamma2_error = float(errors[10])
            self.gammaz = float(parameters[11])
            self.gammaz_error = float(errors[11])
            self.chi2 = float(parameters[-1])

            self.blends = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters, len(parameters) - 1, 2)])
            blends_error = [float(errors[i]) for i in range(self.number_of_parameters, len(errors), 2)]
            self.sources = np.array(
                [float(parameters[i]) for i in range(self.number_of_parameters + 1, len(parameters) - 1, 2)])
            sources_error = [float(errors[i]) for i in range(self.number_of_parameters + 1, len(errors), 2)]

            # TODO CHECK THIS
            self.blendings = self.blends / (self.sources + 1.e-12 * self.blends)
            self.blendings_error = blends_error
            self.baselines = -2.5 * np.log10(self.blends + self.sources)
            self.baselines_error = sources_error


if __name__ == "__main__":
    # Example usage
    event_folder = "/Users/stela/Documents/Scripts/orbital_task/RTModel_runs/sample_rtmodel_v2.4/event_0_603_3135"
    event_results = EventResults(event_folder)
    event_results.extract_information_from_nature_file()
    event_results.looking_for_the_names_of_best_model_of_each_category()
    print(event_results)
