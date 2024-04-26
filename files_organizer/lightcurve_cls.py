import files_organizer.data_challenge_reader as dcr
import pandas as pd


class LightcurveEvent:
    def __init__(self, data_challenge_lc_number):
        self.lc_number = data_challenge_lc_number
        self.lc_type = dcr.what_type_of_lightcurve(data_challenge_lc_number,
                                                   binary_star_path_='../data/binary_star.csv',
                                                   bound_planet_path_='../data/bound_planet.csv',
                                                   cataclysmic_variables_path_='../data/cataclysmic_variables.csv',
                                                   single_lens_path_='../data/single_lens.csv'
                                                   )
        master_path = f'../data/{self.lc_type}.csv'
        self.master_df = pd.read_csv(master_path)
        self.lightcurve_master = self.master_df[self.master_df['data_challenge_lc_number'] == self.lc_number]

        self.event_dict = self.event_dict_creator(self.lightcurve_master)
        self.source_dict = self.source_dict_creator(self.lightcurve_master)
        self.weights_and_flags_dict = self.weights_and_flags_dict_creator(self.lightcurve_master, self.lc_type)
        self.all_dicts = {'Event dictionary: event_dict': self.event_dict,
                          'Source dictionary: source_dict': self.source_dict,
                          'Weights and flags dictionary: weights_and_flags_dict': self.weights_and_flags_dict}
        if self.lc_type == 'bound_planet' or self.lc_type == 'binary_star' or self.lc_type == 'single_lens':
            self.lens_dict = self.lens_dict_creator(self.lightcurve_master)
            self.microlensing_dict = self.microlensing_dict_creator(self.lightcurve_master)
            self.all_dicts['Lens dictionary: lens_dict'] = self.lens_dict
            self.all_dicts['Microlensing Parameter dictionary: microlensing_dict'] = self.microlensing_dict
        if self.lc_type == 'bound_planet':
            self.planet_dict = self.second_lens_dict_creator(self.lightcurve_master, self.lc_type)
            self.all_dicts['Planet dictionary: planet_dict'] = self.planet_dict
        elif self.lc_type == 'binary_star':
            self.second_lens_dict = self.second_lens_dict_creator(self.lightcurve_master, self.lc_type)
            self.all_dicts['Second lens dictionary: second_lens_dict'] = self.second_lens_dict

    def lightcurve_data(self, *, filter_, folder_path_='../data'):
        """
        This function reads the lightcurve data and returns a pandas dataframe with the columns
        :param filter_:
        :param folder_path_:
        :return:
        """
        lightcurve_df = dcr.lightcurve_data_reader(self.lc_number, filter_=filter_, folder_path_=folder_path_)
        return lightcurve_df

    def print_dictionaries(self):
        # Print info about available dictionaries
        print("Available Dictionaries:")
        for key in self.all_dicts.keys():
            print(key)

    @staticmethod
    def event_dict_creator(lightcurve_master):
        # Field and Coordinate Information: This includes the event's index, subrun number, field,
        # galactic coordinates (l, b), and right ascension and declination (ra, dec).
        event_index = lightcurve_master['idx'].values[0]
        event_subrun = lightcurve_master['subrun'].values[0]
        event_field = lightcurve_master['field'].values[0]
        event_galactic_longitude__deg = lightcurve_master['l'].values[0]
        event_galactic_latitude__deg = lightcurve_master['b'].values[0]
        event_right_ascension__deg = lightcurve_master['ra'].values[0]
        event_declination__deg = lightcurve_master['dec'].values[0]

        dict_event = {'event_index': event_index,
                      'event_subrun': event_subrun,
                      'event_field': event_field,
                      'event_galactic_longitude': event_galactic_longitude__deg,
                      'event_galactic_latitude': event_galactic_latitude__deg,
                      'event_right_ascension': event_right_ascension__deg,
                      'event_declination': event_declination__deg}
        return dict_event

    @staticmethod
    def source_dict_creator(lightcurve_master):
        # Source Star Properties: Details about the source star, such as its ID, distance (Ds), radius (Rs),
        # proper motion (smu_l, smu_b), age, class (scl), and type (styp).
        source_id = lightcurve_master['src_id'].values[0]
        source_distance__kpc = lightcurve_master['Ds'].values[0]
        source_radius__rsun = lightcurve_master['Rs'].values[0]
        source_age = lightcurve_master['sage'].values[0]
        source_class = lightcurve_master['scl'].values[0]
        source_type = lightcurve_master['styp'].values[0]

        source_proper_motion_l_1masyr = lightcurve_master['smu_l'].values[0]
        source_proper_motion_b_1masyr = lightcurve_master['smu_b'].values[0]

        # Magnitudes: The magnitudes of the source in various filters (e.g., J, F087, H, W149, W169).
        source_magnitude_J__mag = lightcurve_master['Js'].values[0]
        source_magnitude_F087__mag = lightcurve_master['F087s'].values[0]
        source_magnitude_H__mag = lightcurve_master['Hs'].values[0]
        source_magnitude_W149__mag = lightcurve_master['W149s'].values[0]
        source_magnitude_W169__mag = lightcurve_master['W169s'].values[0]

        source_dict = {'source_id': source_id,
                       'source_distance__kpc': source_distance__kpc,
                       'source_radius__rsun': source_radius__rsun,
                       'source_proper_motion_l_1masyr': source_proper_motion_l_1masyr,
                       'source_proper_motion_b_1masyr': source_proper_motion_b_1masyr,
                       'source_age': source_age,
                       'source_class': source_class,
                       'source_type': source_type,
                       'source_magnitude_J__mag': source_magnitude_J__mag,
                       'source_magnitude_F087__mag': source_magnitude_F087__mag,
                       'source_magnitude_H__mag': source_magnitude_H__mag,
                       'source_magnitude_W149__mag': source_magnitude_W149__mag,
                       'source_magnitude_W169__mag': source_magnitude_W169__mag}
        return source_dict

    @staticmethod
    def lens_dict_creator(lightcurve_master):
        # Lens Star Properties: Information on the lens star, including its ID, distance (Dl), mass (Ml),
        # proper motion (lmu_l, lmu_b), age, class (lcl), type (ltyp), bolometric magnitude (lMbol),
        # effective temperature (lTeff), surface gravity (Llogg), and radius (Rl).
        # MASSES
        # Ml: mass of the lens
        lens_system_mass__msun = lightcurve_master['Ml'].values[0]

        # Dl: distance to the lens
        lens_distance__kpc = lightcurve_master['Dl'].values[0]
        #Radius
        lens_radius__rsun = lightcurve_master['Rl'].values[0]
        # Motion
        lens_proper_motion_l_1masyr = lightcurve_master['lmu_l'].values[0]
        lens_proper_motion_b_1masyr = lightcurve_master['lmu_b'].values[0]

        # Other lens parameters
        lens_id = lightcurve_master['lens_id'].values[0]
        lens_age = lightcurve_master['lage'].values[0]
        lens_class = lightcurve_master['lcl'].values[0]
        lens_type = lightcurve_master['ltyp'].values[0]

        lens_surface_gravity = lightcurve_master['Llogg'].values[0]
        lens_bolometric_magnitude = lightcurve_master['lMbol'].values[0]
        lens_effective_temperature__k = lightcurve_master['lTeff'].values[0]

        # Magnitudes: The magnitudes of the lens in various filters (e.g., J, F087, H, W149, W169).
        lens_magnitude_J__mag = lightcurve_master['Jl'].values[0]
        lens_magnitude_F087__mag = lightcurve_master['F087l'].values[0]
        lens_magnitude_H__mag = lightcurve_master['Hl'].values[0]
        lens_magnitude_W149__mag = lightcurve_master['W149l'].values[0]
        lens_magnitude_W169__mag = lightcurve_master['W169l'].values[0]

        dict_lens = {'lens_id': lens_id,
                     'lens_distance__kpc': lens_distance__kpc,
                     'lens_system_mass__msun': lens_system_mass__msun,
                     'lens_proper_motion_l_1masyr': lens_proper_motion_l_1masyr,
                     'lens_proper_motion_b_1masyr': lens_proper_motion_b_1masyr,
                     'lens_age': lens_age,
                     'lens_class': lens_class,
                     'lens_type': lens_type,
                     'lens_surface_gravity': lens_surface_gravity,
                     'lens_bolometric_magnitude': lens_bolometric_magnitude,
                     'lens_effective_temperature__k': lens_effective_temperature__k,
                     'lens_radius__rsun': lens_radius__rsun,
                     'lens_magnitude_J__mag': lens_magnitude_J__mag,
                     'lens_magnitude_F087__mag': lens_magnitude_F087__mag,
                     'lens_magnitude_H__mag': lens_magnitude_H__mag,
                     'lens_magnitude_W149__mag': lens_magnitude_W149__mag,
                     'lens_magnitude_W169__mag': lens_magnitude_W169__mag}
        return dict_lens

    @staticmethod
    def microlensing_dict_creator(lightcurve_master):
        # Microlensing Event Properties: Parameters such as the impact parameter (u0), event angle (alpha),
        # time of maximum magnification (t0), Einstein crossing time (tE), Einstein radius (rE), theta_E (thE),
        # parallax effect (piE), source radius in Einstein radii (rhos), relative proper motion (murel),
        # transverse velocity (vt), and shear (gamma).
        microlensing_impact_parameter_u0 = lightcurve_master['u0'].values[0]
        microlensing_event_angle_alpha__deg = lightcurve_master['alpha'].values[0]
        microlensing_time_of_peak_t0__days = lightcurve_master['t0'].values[0]
        microlensing_einstein_crossing_time_tE__days = lightcurve_master['tE'].values[0]
        microlensing_einstein_radius_rE__AU = lightcurve_master['rE'].values[0]
        microlensing_einstein_angular_radius_thetaE__mas = lightcurve_master['thE'].values[0]
        microlensing_parallax_effect_piE = lightcurve_master['piE'].values[0]
        microlensing_source_radius_in_einstein_radii_rho = lightcurve_master['rhos'].values[0]
        microlensing_relative_proper_motion_murel__1masyr = lightcurve_master['murel'].values[0]
        microlensing_transverse_velocity_vt__km1s = lightcurve_master['vt'].values[0]
        microlensing_shear_gamma = lightcurve_master['gamma'].values[0]

        # Blending Flux (fs): The blending parameter, which is crucial for determining the combined
        # flux from the source, lens, and any additional blended light.
        microlensing_blending_flux_fs0 = lightcurve_master['fs0'].values[0]
        microlensing_blending_flux_fs1 = lightcurve_master['fs1'].values[0]

        microlensing_dict = {'microlensing_impact_parameter_u0': microlensing_impact_parameter_u0,
                             'microlensing_event_angle_alpha__deg': microlensing_event_angle_alpha__deg,
                             'microlensing_time_of_peak_t0__days': microlensing_time_of_peak_t0__days,
                             'microlensing_einstein_crossing_time_tE__days': microlensing_einstein_crossing_time_tE__days,
                             'microlensing_einstein_radius_rE__AU': microlensing_einstein_radius_rE__AU,
                             'microlensing_einstein_angular_radius_thetaE__mas': microlensing_einstein_angular_radius_thetaE__mas,
                             'microlensing_parallax_effect_piE': microlensing_parallax_effect_piE,
                             'microlensing_source_radius_in_einstein_radii_rho': microlensing_source_radius_in_einstein_radii_rho,
                             'microlensing_relative_proper_motion_murel__1masyr': microlensing_relative_proper_motion_murel__1masyr,
                             'microlensing_transverse_velocity_vt__km1s': microlensing_transverse_velocity_vt__km1s,
                             'microlensing_shear_gamma': microlensing_shear_gamma,
                             'microlensing_blending_flux_fs0': microlensing_blending_flux_fs0,
                             'microlensing_blending_flux_fs1': microlensing_blending_flux_fs1}
        return microlensing_dict

    @staticmethod
    def second_lens_dict_creator(lightcurve_master, lc_type):
        # Planet Properties: For simulations involving planets/binary stars, this section lists the planet's mass (Mp)
        # /secondary mass,
        # semi-major axis (a), inclination (inc), orbital phase (phase), mass ratio to the host star (q),
        # separation (s), and orbital period (period).
        # q: mass ratio
        planet_mass_ratio = lightcurve_master['q'].values[0]
        # Mp: mass of the planet
        planet_mass__msun = lightcurve_master['Mp'].values[0]
        # mass of the planet calculated from the mass ratio
        # Ml: mass of the lens
        lens_system_mass__msun = lightcurve_master['Ml'].values[0]

        lens_planet_mass_ver = (planet_mass_ratio / (planet_mass_ratio + 1)) * lens_system_mass__msun
        # mass of the host star
        lens_host_mass__msun = lens_system_mass__msun / (1 + planet_mass_ratio)

        # Distances
        # a: semi-major axis
        planet_semi_major_axis__au = lightcurve_master['a'].values[0]
        # separation
        planet_separation = lightcurve_master['s'].values[0]
        planet_inclination__deg = lightcurve_master['inc'].values[0]
        planet_orbital_phase = lightcurve_master['phase'].values[0]
        # period
        planet_orbital_period__years = lightcurve_master['period'].values[0]
        if lc_type == 'bound_planet':
            dict_name = 'planet'
        else:
            dict_name = 'second_lens'
        second_lens_dict = {f'{dict_name}_mass__msun': planet_mass__msun,
                            f'{dict_name}_semi_major_axis__au': planet_semi_major_axis__au,
                            f'{dict_name}_inclination__deg': planet_inclination__deg,
                            f'{dict_name}_orbital_phase': planet_orbital_phase,
                            f'{dict_name}_mass_ratio': planet_mass_ratio,
                            f'{dict_name}_separation': planet_separation,
                            f'{dict_name}_orbital_period__years': planet_orbital_period__years}
        return second_lens_dict

    @staticmethod
    def weights_and_flags_dict_creator(lightcurve_master, lc_type):
        # Weight and Flags: Includes the raw weight (raww), final weight (w), and flags indicating
        # certain conditions of the simulation (e.g., FSflag, flatsatFlag).
        weight_and_flags_u0max = lightcurve_master['u0max'].values[0]
        weight_and_flags_raw_weight = lightcurve_master['raww'].values[0]
        weight_and_flags_final_weight = lightcurve_master['w'].values[0]
        weight_and_flags_FS_flag = lightcurve_master['FSflag'].values[0]
        weight_and_flags_flatsat_flag = lightcurve_master['flatsatFlag'].values[0]

        # Analysis Flags and Chi-square Values: Indicators for the analysis status, including
        # flat lightcurve flag (flatchi2), chi-square values for different fits (chi2_0, chi2_1, chi2),
        # and the normalization weight (normw).
        weight_and_flags_flat_lightcurve_flag = lightcurve_master['flatchi2'].values[0]
        weight_and_flags_chi2_0 = lightcurve_master['chi2_0'].values[0]
        weight_and_flags_chi2_1 = lightcurve_master['chi2_1'].values[0]
        weight_and_flags_chi2 = lightcurve_master['chi2'].values[0]

        weight_and_flags_dict = {'weight_and_flags_u0max': weight_and_flags_u0max,
                                 'weight_and_flags_raw_weight': weight_and_flags_raw_weight,
                                 'weight_and_flags_final_weight': weight_and_flags_final_weight,
                                 'weight_and_flags_FS_flag': weight_and_flags_FS_flag,
                                 'weight_and_flags_flatsat_flag': weight_and_flags_flatsat_flag,
                                 'weight_and_flags_flat_lightcurve_flag': weight_and_flags_flat_lightcurve_flag,
                                 'weight_and_flags_chi2_0': weight_and_flags_chi2_0,
                                 'weight_and_flags_chi2_1': weight_and_flags_chi2_1,
                                 'weight_and_flags_chi2': weight_and_flags_chi2}

        if lc_type in ['bound_planet', 'binary_star', 'single_lens']:
            weight_and_flags_normalization_weight = lightcurve_master['normw'].values[0]
            weight_and_flags_dict['weight_and_flags_normalization_weight'] = weight_and_flags_normalization_weight

        return weight_and_flags_dict
