import pandas as pd
import re


def master_head_getter(path_to_wfirstcolumn_file_):
    """
    This function reads the wfirstcolumn file and returns a pandas dataframe with the columns
    :param path_to_wfirstcolumn_file_:
    :return:
    """
    wfirstcolumn_df = pd.read_csv(path_to_wfirstcolumn_file_, sep="\s+")
    return wfirstcolumn_df


def master_file_spliter(path_to_master_file_, path_to_wfirstcolumn_file_, path_to_output_folder_):
    """
    This function reads the master file and splits it into 4 different files based on the type of event.
    :param path_to_master_file_:
    :param path_to_wfirstcolumn_file_:
    :param path_to_output_folder_:
    :return:
    """
    single_lens_lines = []  # dcnormffp
    binary_star_lines = []  # ombin
    bound_planet_lines = []  # omcassan
    cataclysmic_variables_lines = []  # dccv
    header_line = str(master_head_getter(path_to_wfirstcolumn_file_)['name'].values)
    header_line = header_line.replace(' \'|\' ', ',').replace('\' \'', ',').replace('\'', '')
    header_line = header_line.replace(' ', '').replace('|', '').replace('\n', ',')
    header_line = header_line.replace('[','').replace(']','').replace(',,', ',')
    print()
    with open(path_to_master_file_, "r") as master_file:
        for line in master_file:
            line = line.replace(' | ', ' ')
            if "Sky position" in line:
                line = line.replace(' ', ' ')
                line = re.sub("\s+", ",", line.strip())
                master_header_line = line
            else:
                line = line.replace(' ', ',')
                if "dcnormffp" in line:
                    single_lens_lines.append(line)
                elif "ombin" in line:
                    binary_star_lines.append(line)
                elif "omcassan" in line:
                    bound_planet_lines.append(line)
                elif "dccv" in line:
                    cataclysmic_variables_lines.append(line)

                else:
                    raise ValueError("Unknown type of event\n", line)

    with open(f'{path_to_output_folder_}/single_lens.csv', 'w') as single_lens_file:
        single_lens_header_line = header_line+(',unimportant0,unimportant1,unimportant2,'
                                               'event_type,unimportant3,lc_root,data_challenge_lc_number\n')
        single_lens_file.write(single_lens_header_line)
        for line in single_lens_lines:
            single_lens_file.write(line)

    with open(f'{path_to_output_folder_}/binary_star.csv', 'w') as binary_star_file:
        binary_lens_header_line = header_line.replace(',normw,sigma_t0,sigma_tE,sigma_u0,sigma_alpha,sigma_s,'
                                                      'sigma_q,sigma_rs,sigma_F00,sigma_fs0,sigma_F01,sigma_fs1,'
                                                      'sigma_thetaE',
                                                      ',normw,event_type,unimportant0,lc_root,'
                                                      'data_challenge_lc_number\n')
        binary_star_file.write(binary_lens_header_line)
        for line in binary_star_lines:
            line = line.replace(',,,', ',')
            binary_star_file.write(line)

    with open(f'{path_to_output_folder_}/bound_planet.csv', 'w') as bound_planet_file:
        bound_planet_header_line = header_line.replace(',normw,sigma_t0,sigma_tE,sigma_u0,sigma_alpha,sigma_s,'
                                                       'sigma_q,sigma_rs,sigma_F00,sigma_fs0,sigma_F01,sigma_fs1,'
                                                       'sigma_thetaE',
                                                       ',normw,event_type,unimportant0,lc_root,'
                                                       'data_challenge_lc_number\n')
        bound_planet_file.write(bound_planet_header_line)
        for line in bound_planet_lines:
            line = line.replace(',,,', ',')
            bound_planet_file.write(line)

    with open(f'{path_to_output_folder_}/cataclysmic_variables.csv', 'w') as cataclysmic_variables_file:
        cataclysmic_variables_header_line = header_line.replace(',normw,sigma_t0,sigma_tE,sigma_u0,sigma_alpha,sigma_s,'
                                                                'sigma_q,sigma_rs,sigma_F00,sigma_fs0,sigma_F01,'
                                                                'sigma_fs1,'
                                                                'sigma_thetaE',
                                                                ',event_type,unimportant0,lc_root,'
                                                                'data_challenge_lc_number\n')
        cataclysmic_variables_file.write(cataclysmic_variables_header_line)
        for line in cataclysmic_variables_lines:
            cataclysmic_variables_file.write(line)


if __name__ == '__main__':
    # Write where you can find the master files
    root_path = '/Users/stela/Documents/Scripts/RTModel_project/datachallenge'
    path_to_master_file = f'{root_path}/master_file.txt'
    path_to_wfirstcolumn_file = f'{root_path}/wfirstColumnNumbers.txt'
    path_to_output_folder = root_path
    master_file_spliter(path_to_master_file, path_to_wfirstcolumn_file, path_to_output_folder)
    print("Files have been created successfully")


