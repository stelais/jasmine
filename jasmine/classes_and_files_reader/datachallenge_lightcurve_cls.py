from dataclasses import dataclass

import jasmine.classes_and_files_reader.datachallenge_reader as dcr
import pandas as pd


class LightcurveEventDataChallenge:
    def __init__(self, data_challenge_lc_number, data_folder='../data'):
        self.lc_number = data_challenge_lc_number
        self.data_folder = data_folder
        self.lc_type = dcr.what_type_of_lightcurve(data_challenge_lc_number,
                                                   binary_star_path_=f'{data_folder}/binary_star.csv',
                                                   bound_planet_path_=f'{data_folder}/bound_planet.csv',
                                                   cataclysmic_variables_path_=f'{data_folder}/cataclysmic_variables.csv',
                                                   single_lens_path_=f'{data_folder}/single_lens.csv'
                                                   )
        master_path = f'{data_folder}/{self.lc_type}.csv'
        master_df = pd.read_csv(master_path)
        self.lightcurve_master = master_df[master_df['data_challenge_lc_number'] == self.lc_number]
        self.event_info = EventInformation(self.lightcurve_master)
        self.source = SourceStarProperties(self.lightcurve_master)
        self.weights_and_flags = WeightAndFlags(self.lightcurve_master, self.lc_type)
        if self.lc_type == 'bound_planet' or self.lc_type == 'binary_star' or self.lc_type == 'single_lens':
            self.lens = LensStarProperties(self.lightcurve_master)
            self.microlensing = MicrolensingParameters(self.lightcurve_master)
        if self.lc_type == 'bound_planet':
            self.planet = PlanetProperties(self.lightcurve_master)
        if self.lc_type == 'binary_star':
            self.second_lens = SecondLens(self.lightcurve_master)

    def lightcurve_data(self, *, filter_, folder_path_='../data'):
        """
        This function reads the lightcurve data and returns a pandas dataframe with the columns
        :param filter_:
        :param folder_path_:
        :return:
        """
        lightcurve_df = dcr.lightcurve_data_reader(self.lc_number, filter_=filter_, folder_path_=folder_path_)
        return lightcurve_df


@dataclass
class EventInformation:
    """
    Field and Coordinate Information: This includes the event's index, subrun number, field,
    galactic coordinates (l, b), and right ascension and declination (ra, dec).
    """
    event_index: int
    event_subrun: int
    event_field: int
    event_galactic_longitude__deg: float
    event_galactic_latitude__deg: float
    event_right_ascension__deg: float
    event_declination__deg: float
    event_type: str
    lc_root: str
    data_challenge_lc_number: int

    def __init__(self, lightcurve_master: pd.DataFrame):
        self.event_index = lightcurve_master['idx'].values[0]
        self.event_subrun = lightcurve_master['subrun'].values[0]
        self.event_field = lightcurve_master['field'].values[0]
        self.event_galactic_longitude__deg = lightcurve_master['l'].values[0]
        self.event_galactic_latitude__deg = lightcurve_master['b'].values[0]
        self.event_right_ascension__deg = lightcurve_master['ra'].values[0]
        self.event_declination__deg = lightcurve_master['dec'].values[0]
        self.event_type = lightcurve_master['event_type'].values[0]
        self.lc_root = lightcurve_master['lc_root'].values[0]
        self.data_challenge_lc_number = lightcurve_master['data_challenge_lc_number'].values[0]


@dataclass
class SourceStarProperties:
    """
    Source Star Properties: Details about the source star, such as its ID, distance (Ds), radius (Rs),
    proper motion (smu_l, smu_b), age, class (scl), and type (styp).
    Magnitudes: The magnitudes of the source in various filters (e.g., J, F087, H, W149, W169).
    """
    source_id: int
    source_distance__kpc: float
    source_radius__rsun: float
    source_age: int
    source_class: int
    source_type: float
    source_proper_motion_l_1masyr: float
    source_proper_motion_b_1masyr: float
    source_magnitude_J__mag: float
    source_magnitude_F087__mag: float
    source_magnitude_H__mag: float
    source_magnitude_W149__mag: float
    source_magnitude_W169__mag: float

    def __init__(self, lightcurve_master: pd.DataFrame):
        self.source_id = lightcurve_master['src_id'].values[0]
        self.source_distance__kpc = lightcurve_master['Ds'].values[0]
        self.source_radius__rsun = lightcurve_master['Rs'].values[0]
        self.source_age = lightcurve_master['sage'].values[0]
        self.source_class = lightcurve_master['scl'].values[0]
        self.source_type = lightcurve_master['styp'].values[0]

        self.source_proper_motion_l_1masyr = lightcurve_master['smu_l'].values[0]
        self.source_proper_motion_b_1masyr = lightcurve_master['smu_b'].values[0]

        self.source_magnitude_J__mag = lightcurve_master['Js'].values[0]
        self.source_magnitude_F087__mag = lightcurve_master['F087s'].values[0]
        self.source_magnitude_H__mag = lightcurve_master['Hs'].values[0]
        self.source_magnitude_W149__mag = lightcurve_master['W149s'].values[0]
        self.source_magnitude_W169__mag = lightcurve_master['W169s'].values[0]


@dataclass
class LensStarProperties:
    """
    Lens Star Properties: Information on the lens star, including its ID, distance (Dl), mass (Ml),
    proper motion (lmu_l, lmu_b), age, class (lcl), type (ltyp), bolometric magnitude (lMbol),
    effective temperature (lTeff), surface gravity (Llogg), and radius (Rl).
    """
    lens_system_mass__msun: float
    lens_distance__kpc: float
    lens_radius__rsun: float
    lens_proper_motion_l_1masyr: float
    lens_proper_motion_b_1masyr: float
    lens_id: int
    lens_age: int
    lens_class: int
    lens_type: float
    lens_surface_gravity: float
    lens_bolometric_magnitude: float
    lens_effective_temperature__k: float
    lens_magnitude_J__mag: float
    lens_magnitude_F087__mag: float
    lens_magnitude_H__mag: float
    lens_magnitude_W149__mag: float
    lens_magnitude_W169__mag: float

    def __init__(self, lightcurve_master: pd.DataFrame):
        # MASSES
        # Ml: mass of the lens
        self.lens_system_mass__msun = lightcurve_master['Ml'].values[0]

        # Dl: distance to the lens
        self.lens_distance__kpc = lightcurve_master['Dl'].values[0]
        #Radius
        self.lens_radius__rsun = lightcurve_master['Rl'].values[0]
        # Motion
        self.lens_proper_motion_l_1masyr = lightcurve_master['lmu_l'].values[0]
        self.lens_proper_motion_b_1masyr = lightcurve_master['lmu_b'].values[0]

        # Other lens parameters
        self.lens_id = lightcurve_master['lens_id'].values[0]
        self.lens_age = lightcurve_master['lage'].values[0]
        self.lens_class = lightcurve_master['lcl'].values[0]
        self.lens_type = lightcurve_master['ltyp'].values[0]

        self.lens_surface_gravity = lightcurve_master['Llogg'].values[0]
        self.lens_bolometric_magnitude = lightcurve_master['lMbol'].values[0]
        self.lens_effective_temperature__k = lightcurve_master['lTeff'].values[0]

        # Magnitudes: The magnitudes of the lens in various filters (e.g., J, F087, H, W149, W169).
        self.lens_magnitude_J__mag = lightcurve_master['Jl'].values[0]
        self.lens_magnitude_F087__mag = lightcurve_master['F087l'].values[0]
        self.lens_magnitude_H__mag = lightcurve_master['Hl'].values[0]
        self.lens_magnitude_W149__mag = lightcurve_master['W149l'].values[0]
        self.lens_magnitude_W169__mag = lightcurve_master['W169l'].values[0]


@dataclass
class MicrolensingParameters:
    """
    Microlensing Event Properties: Parameters such as the impact parameter (u0), event angle (alpha),
    time of maximum magnification (t0), Einstein crossing time (tE), Einstein radius (rE), theta_E (thE),
    parallax effect (piE), source radius in Einstein radii (rhos), relative proper motion (murel),
    transverse velocity (vt), and shear (gamma).

    Blending Flux (fs): The blending parameter, which is crucial for determining the combined
    flux from the source, lens, and any additional blended light.
    """
    microlensing_impact_parameter_u0: float
    microlensing_event_angle_alpha__deg: float
    microlensing_time_of_peak_t0__days: float
    microlensing_einstein_crossing_time_tE__days: float
    microlensing_einstein_radius_rE__AU: float
    microlensing_einstein_angular_radius_thetaE__mas: float
    microlensing_parallax_effect_piE: float
    microlensing_source_radius_in_einstein_radii_rho: float
    microlensing_relative_proper_motion_murel__1masyr: float
    microlensing_transverse_velocity_vt__km1s: float
    microlensing_shear_gamma: float
    microlensing_blending_flux_fs0: float
    microlensing_blending_flux_fs1: float

    def __init__(self, lightcurve_master: pd.DataFrame):
        self.microlensing_impact_parameter_u0 = lightcurve_master['u0'].values[0]
        self.microlensing_event_angle_alpha__deg = lightcurve_master['alpha'].values[0]
        self.microlensing_time_of_peak_t0__days = lightcurve_master['t0'].values[0]
        self.microlensing_einstein_crossing_time_tE__days = lightcurve_master['tE'].values[0]
        self.microlensing_einstein_radius_rE__AU = lightcurve_master['rE'].values[0]
        self.microlensing_einstein_angular_radius_thetaE__mas = lightcurve_master['thE'].values[0]
        self.microlensing_parallax_effect_piE = lightcurve_master['piE'].values[0]
        self.microlensing_source_radius_in_einstein_radii_rho = lightcurve_master['rhos'].values[0]
        self.microlensing_relative_proper_motion_murel__1masyr = lightcurve_master['murel'].values[0]
        self.microlensing_transverse_velocity_vt__km1s = lightcurve_master['vt'].values[0]
        self.microlensing_shear_gamma = lightcurve_master['gamma'].values[0]
        self.microlensing_blending_flux_fs0 = lightcurve_master['fs0'].values[0]
        self.microlensing_blending_flux_fs1 = lightcurve_master['fs1'].values[0]


@dataclass
class WeightAndFlags:
    """
    Weight and Flags: Includes the raw weight (raww), final weight (w), and flags indicating
    certain conditions of the simulation (e.g., FSflag, flatsatFlag).
    Analysis Flags and Chi-square Values: Indicators for the analysis status, including
    flat lightcurve flag (flatchi2), chi-square values for different fits (chi2_0, chi2_1, chi2),
    and the normalization weight (normw).
    """
    weight_and_flags_u0max: float
    weight_and_flags_raw_weight: float
    weight_and_flags_final_weight: float
    weight_and_flags_FS_flag: int
    weight_and_flags_flatsat_flag: int
    weight_and_flags_flat_lightcurve_flag: int
    weight_and_flags_chi2_0: float
    weight_and_flags_chi2_1: int
    weight_and_flags_chi2: float
    weight_and_flags_normalization_weight: float

    def __init__(self, lightcurve_master: pd.DataFrame, lc_type: str):
        self.weight_and_flags_u0max = lightcurve_master['u0max'].values[0]
        self.weight_and_flags_raw_weight = lightcurve_master['raww'].values[0]
        self.weight_and_flags_final_weight = lightcurve_master['w'].values[0]
        self.weight_and_flags_FS_flag = lightcurve_master['FSflag'].values[0]
        self.weight_and_flags_flatsat_flag = lightcurve_master['flatsatFlag'].values[0]

        self.weight_and_flags_flat_lightcurve_flag = lightcurve_master['flatchi2'].values[0]
        self.weight_and_flags_chi2_0 = lightcurve_master['chi2_0'].values[0]
        self.weight_and_flags_chi2_1 = lightcurve_master['chi2_1'].values[0]
        self.weight_and_flags_chi2 = lightcurve_master['chi2'].values[0]

        #if lc_type in ['bound_planet', 'binary_star', 'single_lens']:
        #    self.weight_and_flags_normalization_weight = lightcurve_master['normw'].values[0]


@dataclass
class PlanetProperties:
    """
    Planet Properties: For simulations involving planets,
    this section lists the planet's mass (Mp),
    semi-major axis (a), inclination (inc), orbital phase (phase), mass ratio to the host star (q),
    separation (s), and orbital period (period).
    """
    planet_mass_ratio: float
    planet_mass__msun: float
    lens_system_mass__msun: float
    lens_planet_mass_ver: float
    lens_host_mass__msun: float
    planet_semi_major_axis__au: float
    planet_separation: float
    planet_inclination__deg: float
    planet_orbital_phase: float
    planet_orbital_period__years: float

    def __init__(self, lightcurve_master: pd.DataFrame):
        # q: mass ratio
        self.planet_mass_ratio = lightcurve_master['q'].values[0]
        # Mp: mass of the planet
        self.planet_mass__msun = lightcurve_master['Mp'].values[0]
        # mass of the planet calculated from the mass ratio
        # Ml: mass of the lens
        self.lens_system_mass__msun = lightcurve_master['Ml'].values[0]

        self.lens_planet_mass_ver = (self.planet_mass_ratio / (
                    self.planet_mass_ratio + 1)) * self.lens_system_mass__msun
        # mass of the host star
        self.lens_host_mass__msun = self.lens_system_mass__msun / (1 + self.planet_mass_ratio)

        # Distances
        # a: semi-major axis
        self.planet_semi_major_axis__au = lightcurve_master['a'].values[0]
        # separation
        self.planet_separation = lightcurve_master['s'].values[0]
        self.planet_inclination__deg = lightcurve_master['inc'].values[0]
        self.planet_orbital_phase = lightcurve_master['phase'].values[0]
        # period
        self.planet_orbital_period__years = lightcurve_master['period'].values[0]


@dataclass
class SecondLens:
    """
    Second lens Properties: For simulations involving binary stars,
    this section lists the secondary mass (Mp),
    semi-major axis (a), inclination (inc), orbital phase (phase), mass ratio to the host star (q),
    separation (s), and orbital period (period).
    """
    second_lens_mass_ratio: float
    second_lens_mass__msun: float
    lens_system_mass__msun: float
    lens_planet_mass_ver: float
    lens_host_mass__msun: float
    second_lens_semi_major_axis__au: float
    second_lens_separation: float
    second_lens_inclination__deg: float
    second_lens_orbital_phase: float
    second_lens_orbital_period__years: float

    def __init__(self, lightcurve_master: pd.DataFrame):
        # q: mass ratio
        self.second_lens_mass_ratio = lightcurve_master['q'].values[0]
        # Mp: mass of the planet
        self.second_lens_mass__msun = lightcurve_master['Mp'].values[0]
        # mass of the planet calculated from the mass ratio
        # Ml: mass of the lens
        self.lens_system_mass__msun = lightcurve_master['Ml'].values[0]

        self.lens_planet_mass_ver = (self.second_lens_mass_ratio / (
                    self.second_lens_mass_ratio + 1)) * self.lens_system_mass__msun
        # mass of the host star
        self.lens_host_mass__msun = self.lens_system_mass__msun / (1 + self.second_lens_mass_ratio)

        # Distances
        # a: semi-major axis
        self.second_lens_semi_major_axis__au = lightcurve_master['a'].values[0]
        # separation
        self.second_lens_separation = lightcurve_master['s'].values[0]
        self.second_lens_inclination__deg = lightcurve_master['inc'].values[0]
        self.second_lens_orbital_phase = lightcurve_master['phase'].values[0]
        # period
        self.second_lens_orbital_period__years = lightcurve_master['period'].values[0]

