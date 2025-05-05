from dataclasses import dataclass

import src.jasmine.classes_and_files_reader.new_gullsrges_reader as grr
import pandas as pd


class LightcurveEventGULLSRGES_NameBased:
    def __init__(self, lightcurve_name, data_folder='../data/gulls_orbital_motion_extracted/',
                 master_file_path='../data/gulls_orbital_motion_extracted/OMPLDG_croin_cassan.sample.csv'):
        # This is for the new team-simulations release (from 2024Nov).
        master_df = pd.read_csv(master_file_path)
        self.lightcurve_name = lightcurve_name  #e.g. OMPLDG_croin_cassan_0_642_57.det.lc
        self.lightcurve_master = master_df[master_df.lcname == self.lightcurve_name]
        self.event_info = EventInformation(self.lightcurve_master)
        self.sample_folder = data_folder
        self.light_curve_path = f"{self.sample_folder}/{self.lightcurve_name}"
        self.source = SourceStarProperties(self.lightcurve_master)
        self.weights_and_flags = WeightAndFlags(self.lightcurve_master)
        self.lens = LensStarProperties(self.lightcurve_master)
        self.microlensing = MicrolensingParameters(self.lightcurve_master)
        self.planet = PlanetProperties(self.lightcurve_master)

    def lightcurve_data(self):
        """
        This function reads the lightcurve data and returns a tuple of pandas dataframes with the columns
        :param folder_to_be_created_path_:
        :return:
        """
        lightcurve_df_list = grr.lightcurve_data_reader(self.event_info.event_subrun,
                                                        self.event_info.event_field,
                                                        self.event_info.event_id,
                                                        folder_path_=self.sample_folder,
                                                        include_ephem=True)
        return lightcurve_df_list

    def microlensing_signal(self):
        """
        This function reads the microlensing signal data and returns a tuple of pandas dataframes with the columns
        :param folder_to_be_created_path_:
        :return:
        """
        microlensing_df_list = grr.microlensing_signal_reader(self.event_info.event_subrun,
                                                              self.event_info.event_field,
                                                              self.event_info.event_id,
                                                              folder_path_=self.sample_folder,
                                                              include_ephem=True)
        return microlensing_df_list

    def whole_lightcurve(self):
        """
        This function reads the whole lightcurve data and returns a pandas dataframe with the columns
        :param folder_to_be_created_path_:
        :return:
        """
        lightcurve_df = grr.whole_columns_lightcurve_reader(self.event_info.event_subrun,
                                                            self.event_info.event_field,
                                                            self.event_info.event_id,
                                                            folder_path_=self.sample_folder,
                                                            include_ephem=True)
        return lightcurve_df


@dataclass
class EventInformation:
    """
    Field and Coordinate Information: This includes the event's index, subrun number, field,
    galactic coordinates (l, b), and right ascension and declination (ra, dec).
    """
    event_id: int
    event_subrun: int
    event_field: int
    event_galactic_longitude__deg: float
    event_galactic_latitude__deg: float
    event_right_ascension__deg: float
    event_declination__deg: float

    # event_type: str
    # lc_root: str
    # data_challenge_lc_number: int

    def __init__(self, lightcurve_master: pd.DataFrame):
        self.event_id = lightcurve_master['EventID'].values[0]
        self.event_subrun = lightcurve_master['SubRun'].values[0]
        self.event_field = lightcurve_master['Field'].values[0]
        self.event_galactic_longitude__deg = lightcurve_master['galactic_l'].values[0]
        self.event_galactic_latitude__deg = lightcurve_master['galactic_b'].values[0]
        self.event_right_ascension__deg = lightcurve_master['ra_deg'].values[0]
        self.event_declination__deg = lightcurve_master['dec_deg'].values[0]


@dataclass
class SourceStarProperties:
    """
    Source Star Properties: Details about the source star, such as its ID, distance (Ds), radius (Rs),
    proper motion (smu_l, smu_b), age, class (scl).
    Magnitudes: The magnitudes of the source in various filters (e.g., J, F087, H, W149, W169).
    """
    source_id: int
    source_distance__kpc: float
    source_radius__rsun: float
    source_age: int
    source_class: int
    source_mass__msun: float
    source_effective_temperature__k: float
    source_surface_gravity: float
    source_metalicity: float
    source_alpha_Fe: float
    source_pop: int
    source_initial_mass__msun: float
    source_bolometric_magnitude: float
    source_Av: float

    source_l: float
    source_b: float
    source_ra: float
    source_dec: float
    source_x: float
    source_y: float
    source_z: float
    source_proper_motion_l_1masyr: float
    source_proper_motion_b_1masyr: float

    source_magnitude_R062__mag: float
    source_magnitude_Z087__mag: float
    source_magnitude_Y106__mag: float
    source_magnitude_J129__mag: float
    source_magnitude_W146__mag: float
    source_magnitude_H158__mag: float
    source_magnitude_F184__mag: float
    source_magnitude_K213__mag: float
    source_magnitude_2MASS_Ks__mag: float
    source_magnitude_2MASS_J__mag: float
    source_magnitude_2MASS_H__mag: float

    def __init__(self, lightcurve_master: pd.DataFrame):
        self.source_id = lightcurve_master['SourceID'].values[0]

        self.source_distance__kpc = lightcurve_master['Source_Dist'].values[0]
        self.source_radius__rsun = lightcurve_master['Source_Radius'].values[0]
        self.source_age = lightcurve_master['Source_age'].values[0]
        self.source_class = lightcurve_master['Source_CL'].values[0]
        # self.source_type = lightcurve_master['styp'].values[0]
        self.source_mass__msun = lightcurve_master['Source_Mass'].values[0]
        self.source_effective_temperature__k = lightcurve_master['Source_Teff'].values[0]
        self.source_surface_gravity = lightcurve_master['Source_logg'].values[0]
        self.source_metalicity = lightcurve_master['Source_Fe_H'].values[0]
        self.source_alpha_Fe = lightcurve_master['Source_alpha_Fe'].values[0]
        self.source_pop = lightcurve_master['Source_pop'].values[0]
        self.source_initial_mass__msun = lightcurve_master['Source_iMass'].values[0]
        self.source_bolometric_magnitude = lightcurve_master['Source_Mbol'].values[0]
        self.source_Av = lightcurve_master['Source_Av'].values[0]
        self.source_l = lightcurve_master['Source_l'].values[0]
        self.source_b = lightcurve_master['Source_l'].values[0]
        self.source_ra = lightcurve_master['Source_RA20000'].values[0]
        self.source_dec = lightcurve_master['Source_DEC20000'].values[0]
        self.source_x = lightcurve_master['Source_x'].values[0]
        self.source_y = lightcurve_master['Source_y'].values[0]
        self.source_z = lightcurve_master['Source_z'].values[0]
        self.source_proper_motion_l_1masyr = lightcurve_master['Source_mul'].values[0]
        self.source_proper_motion_b_1masyr = lightcurve_master['Source_mub'].values[0]

        self.source_magnitude_R062__mag = lightcurve_master['Source_R062'].values[0]
        self.source_magnitude_Z087__mag = lightcurve_master['Source_Z087'].values[0]
        self.source_magnitude_Y106__mag = lightcurve_master['Source_Y106'].values[0]
        self.source_magnitude_J129__mag = lightcurve_master['Source_J129'].values[0]
        self.source_magnitude_W146__mag = lightcurve_master['Source_W146'].values[0]
        self.source_magnitude_H158__mag = lightcurve_master['Source_H158'].values[0]
        self.source_magnitude_F184__mag = lightcurve_master['Source_F184'].values[0]
        self.source_magnitude_K213__mag = lightcurve_master['Source_K213'].values[0]
        self.source_magnitude_2MASS_Ks__mag = lightcurve_master['Source_2MASS_Ks'].values[0]
        self.source_magnitude_2MASS_J__mag = lightcurve_master['Source_2MASS_J'].values[0]
        self.source_magnitude_2MASS_H__mag = lightcurve_master['Source_2MASS_H'].values[0]


@dataclass
class LensStarProperties:
    """
    Lens Star Properties: Information on the lens star, including its ID, distance (Dl), mass (Ml),
    proper motion (lmu_l, lmu_b), age, class (lcl), type (ltyp), bolometric magnitude (lMbol),
    effective temperature (lTeff), surface gravity (Llogg), and radius (Rl).
    """
    lens_system_mass__msun: float
    lens_initial_mass__msun: float
    lens_distance__kpc: float
    lens_radius__rsun: float

    lens_l: float
    lens_b: float
    lens_ra: float
    lens_dec: float
    lens_x: float
    lens_y: float
    lens_z: float
    lens_proper_motion_l_1masyr: float
    lens_proper_motion_b_1masyr: float

    lens_id: int
    lens_age: int
    lens_class: int
    lens_metalicity: float
    lens_alpha_Fe: float
    lens_pop: int
    lens_Av: float

    # lens_type: float
    lens_surface_gravity: float
    lens_bolometric_magnitude: float
    lens_effective_temperature__k: float

    lens_magnitude_R062__mag: float
    lens_magnitude_Z087__mag: float
    lens_magnitude_Y106__mag: float
    lens_magnitude_J129__mag: float
    lens_magnitude_W146__mag: float
    lens_magnitude_H158__mag: float
    lens_magnitude_F184__mag: float
    lens_magnitude_K213__mag: float
    lens_magnitude_2MASS_Ks__mag: float
    lens_magnitude_2MASS_J__mag: float
    lens_magnitude_2MASS_H__mag: float

    def __init__(self, lightcurve_master: pd.DataFrame):
        # MASSES
        # Ml: mass of the lens
        self.lens_system_mass__msun = lightcurve_master['Lens_Mass'].values[0]
        self.lens_initial_mass__msun = lightcurve_master['Lens_iMass'].values[0]

        # Dl: distance to the lens
        self.lens_distance__kpc = lightcurve_master['Lens_Dist'].values[0]
        # Radius
        self.lens_radius__rsun = lightcurve_master['Lens_Radius'].values[0]

        # Position and Motion

        self.lens_l = lightcurve_master['Lens_l'].values[0]
        self.lens_b = lightcurve_master['Lens_b'].values[0]
        self.lens_ra = lightcurve_master['Lens_RA20000'].values[0]
        self.lens_dec = lightcurve_master['Lens_DEC20000'].values[0]
        self.lens_x = lightcurve_master['Lens_x'].values[0]
        self.lens_y = lightcurve_master['Lens_y'].values[0]
        self.lens_z = lightcurve_master['Lens_z'].values[0]
        self.lens_proper_motion_l_1masyr = lightcurve_master['Lens_mul'].values[0]
        self.lens_proper_motion_b_1masyr = lightcurve_master['Lens_mub'].values[0]

        # Other lens parameters
        self.lens_id = lightcurve_master['LensID'].values[0]
        self.lens_age = lightcurve_master['Lens_age'].values[0]
        self.lens_class = lightcurve_master['Lens_CL'].values[0]
        # self.lens_type = lightcurve_master['ltyp'].values[0]

        self.lens_surface_gravity = lightcurve_master['Lens_logg'].values[0]
        self.lens_bolometric_magnitude = lightcurve_master['Lens_Mbol'].values[0]
        self.lens_effective_temperature__k = lightcurve_master['Lens_Teff'].values[0]
        self.lens_metalicity = lightcurve_master['Lens_Fe_H'].values[0]
        self.lens_alpha_Fe = lightcurve_master['Lens_alpha_Fe'].values[0]
        self.lens_pop = lightcurve_master['Lens_pop'].values[0]
        self.lens_Av = lightcurve_master['Lens_Av'].values[0]

        # Magnitudes: The magnitudes of the lens in various filters (e.g., J, F087, H, W149, W169).
        self.lens_magnitude_R062__mag = lightcurve_master['Lens_R062'].values[0]
        self.lens_magnitude_Z087__mag = lightcurve_master['Lens_Z087'].values[0]
        self.lens_magnitude_Y106__mag = lightcurve_master['Lens_Y106'].values[0]
        self.lens_magnitude_J129__mag = lightcurve_master['Lens_J129'].values[0]
        self.lens_magnitude_W146__mag = lightcurve_master['Lens_W146'].values[0]
        self.lens_magnitude_H158__mag = lightcurve_master['Lens_H158'].values[0]
        self.lens_magnitude_F184__mag = lightcurve_master['Lens_F184'].values[0]
        self.lens_magnitude_K213__mag = lightcurve_master['Lens_K213'].values[0]
        self.lens_magnitude_2MASS_Ks__mag = lightcurve_master['Lens_2MASS_Ks'].values[0]
        self.lens_magnitude_2MASS_J__mag = lightcurve_master['Lens_2MASS_J'].values[0]
        self.lens_magnitude_2MASS_H__mag = lightcurve_master['Lens_2MASS_H'].values[0]


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
    # What are these
    microlensing_tcroin_days: float
    microlensing_ucroin: float
    microlensing_rcroin: float
    ### heliocentric and reference frame tE (in Sun and Roman Frames)
    microlensing_einstein_crossing_time_tE_ref__days: float
    microlensing_einstein_crossing_time_tE_helio__days: float
    microlensing_einstein_radius_rE__AU: float
    microlensing_einstein_angular_radius_thetaE__mas: float
    microlensing_source_radius_in_einstein_radii_rho: float
    # Parallax
    microlensing_parallax_effect_piE: float
    microlensing_parallax_effect_piE_East: float
    microlensing_parallax_effect_piE_North: float
    microlensing_parallax_effect_piE_par: float
    microlensing_parallax_effect_piE_rp: float
    # Heliocentric Relative Proper Motion
    microlensing_relative_proper_motion_murel_helio__1masyr: float
    microlensing_relative_proper_motion_murel_alpha_helio___1masyr: float
    microlensing_relative_proper_motion_murel_delta_helio___1masyr: float
    microlensing_relative_proper_motion_murel_l_helio___1masyr: float
    microlensing_relative_proper_motion_murel_b_helio___1masyr: float
    microlensing_relative_proper_motion_murel_lambda_helio___1masyr: float
    # Reference Frame Relative Proper Motion (Roman)
    microlensing_relative_proper_motion_murel_ref__1masyr: float
    microlensing_relative_proper_motion_murel_beta_ref___1masyr: float
    microlensing_relative_proper_motion_murel_ref__1masyr: float
    microlensing_relative_proper_motion_murel_alpha_ref___1masyr: float
    microlensing_relative_proper_motion_murel_delta_ref___1masyr: float
    microlensing_relative_proper_motion_murel_l_ref___1masyr: float
    microlensing_relative_proper_motion_murel_b_ref___1masyr: float
    microlensing_relative_proper_motion_murel_lambda_ref___1masyr: float
    microlensing_relative_proper_motion_murel_beta_ref___1masyr: float
    # Transverse Velocity
    microlensing_transverse_velocity_vt__km1s: float
    microlensing_gamma_ld: float
    microlensing_blending_flux_fs0: float
    microlensing_blending_flux_fs1: float
    microlensing_blending_flux_fs2: float

    def __init__(self, lightcurve_master: pd.DataFrame):
        self.microlensing_impact_parameter_u0 = lightcurve_master['u0lens1'].values[0]
        self.microlensing_event_angle_alpha__deg = lightcurve_master['alpha'].values[0]
        self.microlensing_time_of_peak_t0__days = lightcurve_master['t0lens1'].values[0]

        # Caustic Region of INfluence
        self.microlensing_tcroin_days = lightcurve_master['tcroin'].values[0]
        self.microlensing_ucroin = lightcurve_master['ucroin'].values[0]
        self.microlensing_rcroin = lightcurve_master['rcroin'].values[0]

        # Heliocentric and reference frame tE (in Sun and Roman Frames)
        self.microlensing_einstein_crossing_time_tE_ref__days = lightcurve_master['tE_ref'].values[0]
        self.microlensing_einstein_crossing_time_tE_helio__days = lightcurve_master['tE_helio'].values[0]
        self.microlensing_einstein_radius_rE__AU = lightcurve_master['rE'].values[0]
        self.microlensing_einstein_angular_radius_thetaE__mas = lightcurve_master['thetaE'].values[0]
        self.microlensing_source_radius_in_einstein_radii_rho = lightcurve_master['rho'].values[0]

        # Parallax
        self.microlensing_parallax_effect_piE = lightcurve_master['piE'].values[0]
        self.microlensing_parallax_effect_piE_East = lightcurve_master['piEE'].values[0]
        self.microlensing_parallax_effect_piE_North = lightcurve_master['piEN'].values[0]
        self.microlensing_parallax_effect_piE_par = lightcurve_master['piEll'].values[0]
        self.microlensing_parallax_effect_piE_rp = lightcurve_master['piErp'].values[0]

        # Heliocentric Relative Proper Motion
        self.microlensing_relative_proper_motion_murel_helio__1masyr = lightcurve_master['murel_helio'].values[0]
        self.microlensing_relative_proper_motion_murel_alpha_helio___1masyr = \
            lightcurve_master['murel_helio_alpha'].values[0]
        self.microlensing_relative_proper_motion_murel_delta_helio___1masyr = \
            lightcurve_master['murel_helio_delta'].values[0]
        self.microlensing_relative_proper_motion_murel_l_helio___1masyr = lightcurve_master['murel_helio_l'].values[0]
        self.microlensing_relative_proper_motion_murel_b_helio___1masyr = lightcurve_master['murel_helio_b'].values[0]
        self.microlensing_relative_proper_motion_murel_lambda_helio___1masyr = \
            lightcurve_master['murel_helio_lambda'].values[0]

        # Reference Frame Relative Proper Motion (Roman)
        self.microlensing_relative_proper_motion_murel_ref__1masyr = lightcurve_master['murel_ref'].values[0]
        self.microlensing_relative_proper_motion_murel_beta_ref___1masyr = lightcurve_master['murel_ref_beta'].values[0]
        self.microlensing_relative_proper_motion_murel_ref__1masyr = lightcurve_master['murel_ref'].values[0]
        self.microlensing_relative_proper_motion_murel_alpha_ref___1masyr = lightcurve_master['murel_ref_alpha'].values[
            0]
        self.microlensing_relative_proper_motion_murel_delta_ref___1masyr = lightcurve_master['murel_ref_delta'].values[
            0]
        self.microlensing_relative_proper_motion_murel_l_ref___1masyr = lightcurve_master['murel_ref_l'].values[0]
        self.microlensing_relative_proper_motion_murel_b_ref___1masyr = lightcurve_master['murel_ref_b'].values[0]
        self.microlensing_relative_proper_motion_murel_lambda_ref___1masyr = \
            lightcurve_master['murel_ref_lambda'].values[0]
        self.microlensing_relative_proper_motion_murel_beta_ref___1masyr = lightcurve_master['murel_ref_beta'].values[0]

        # Transverse Velocity
        self.microlensing_transverse_velocity_vt__km1s = lightcurve_master['vt'].values[0]
        '''To Do: add vtilde, V_N, V_E. I don't understand the difference between them right now. '''
        self.microlensing_gamma_ld = 0.36  # since its 0 in the csv ...
        self.microlensing_blending_flux_fs0 = lightcurve_master['Obs_0_fs'].values[0]
        self.microlensing_blending_flux_fs1 = lightcurve_master['Obs_1_fs'].values[0]
        self.microlensing_blending_flux_fs2 = lightcurve_master['Obs_2_fs'].values[0]


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
    weight_and_flags_Error_flag: int
    weight_and_flags_ObsGroup0_FS_flag: int
    weight_and_flags_ObsGroup1_FS_flag: int
    weight_and_flags_ObsGroup2_FS_flag: int
    weight_and_flags_ObsGroup0_flat_lightcurve_flag: int
    weight_and_flags_ObsGroup1_flat_lightcurve_flag: int
    weight_and_flags_ObsGroup2_flat_lightcurve_flag: int
    weight_and_flags_ObsGroup0_flat_chi2: float
    weight_and_flags_ObsGroup1_flat_chi2: float
    weight_and_flags_ObsGroup2_flat_chi2: float
    weight_and_flags_ObsGroup0_chi2: float
    weight_and_flags_ObsGroup1_chi2: float
    weight_and_flags_ObsGroup2_chi2: float
    weight_and_flags_LC_output_flag: int

    def __init__(self, lightcurve_master: pd.DataFrame):
        self.weight_and_flags_u0max = lightcurve_master['u0max'].values[0]
        self.weight_and_flags_raw_weight = lightcurve_master['raw_weight'].values[0]
        self.weight_and_flags_final_weight = lightcurve_master['weight'].values[0]

        self.weight_and_flags_Error_flag = lightcurve_master['ErrorFlag'].values[0]

        self.weight_and_flags_ObsGroup0_FS_flag = lightcurve_master['ObsGroup_0_FiniteSourceflag'].values[0]
        self.weight_and_flags_ObsGroup1_FS_flag = lightcurve_master['ObsGroup_1_FiniteSourceflag'].values[0]
        self.weight_and_flags_ObsGroup2_FS_flag = lightcurve_master['ObsGroup_2_FiniteSourceflag'].values[0]

        self.weight_and_flags_ObsGroup0_flat_lightcurve_flag = lightcurve_master['ObsGroup_0_flatlc'].values[0]
        self.weight_and_flags_ObsGroup1_flat_lightcurve_flag = lightcurve_master['ObsGroup_1_flatlc'].values[0]
        self.weight_and_flags_ObsGroup2_flat_lightcurve_flag = lightcurve_master['ObsGroup_2_flatlc'].values[0]

        self.weight_and_flags_ObsGroup0_flat_chi2 = lightcurve_master['ObsGroup_0_flatchi2'].values[0]
        self.weight_and_flags_ObsGroup1_flat_chi2 = lightcurve_master['ObsGroup_1_flatchi2'].values[0]
        self.weight_and_flags_ObsGroup2_flat_chi2 = lightcurve_master['ObsGroup_2_flatchi2'].values[0]

        self.weight_and_flags_ObsGroup0_chi2 = lightcurve_master['ObsGroup_0_chi2'].values[0]
        self.weight_and_flags_ObsGroup1_chi2 = lightcurve_master['ObsGroup_1_chi2'].values[0]
        self.weight_and_flags_ObsGroup2_chi2 = lightcurve_master['ObsGroup_2_chi2'].values[0]
        self.weight_and_flags_LC_output_flag = lightcurve_master['LCOutput'].values[0]


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
        self.planet_mass_ratio = lightcurve_master['Planet_q'].values[0]
        # Mp: mass of the planet
        self.planet_mass__msun = lightcurve_master['Planet_mass'].values[0]
        # mass of the planet calculated from the mass ratio
        # Ml: mass of the lens
        self.lens_system_mass__msun = lightcurve_master['Lens_Mass'].values[0]

        self.lens_planet_mass_ver = (self.planet_mass_ratio / (
                self.planet_mass_ratio + 1)) * self.lens_system_mass__msun
        # mass of the host star
        self.lens_host_mass__msun = self.lens_system_mass__msun / (1 + self.planet_mass_ratio)

        # Distances
        # a: semi-major axis
        self.planet_semi_major_axis__au = lightcurve_master['Planet_semimajoraxis'].values[0]
        # separation
        self.planet_separation = lightcurve_master['Planet_s'].values[0]
        self.planet_inclination__deg = lightcurve_master['Planet_inclination'].values[0]
        self.planet_orbital_phase = lightcurve_master['Planet_orbphase'].values[0]
        # period
        self.planet_orbital_period__years = lightcurve_master['Planet_period'].values[0]
