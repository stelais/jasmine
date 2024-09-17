import pandas as pd


def rtmodel_data_to_pylima(data_folder_path='None', data_output_folder_path='None', event_number='None'):
    # Column names based on the file content
    column_names = ['Mag', 'err', 'HJD_minus_2450000']
    if data_folder_path == 'None':
        data_folder_path = ''
    else:
        data_folder_path = data_folder_path + '/'
    if data_output_folder_path == 'None':
        data_output_folder_path = ''
    else:
        data_output_folder_path = data_output_folder_path + '/'

    # Reading the data into a DataFrame, skipping the first line which is a header comment
    for filepath in ['RomanW149sat1.dat', 'RomanZ087sat2.dat']:
        data = pd.read_csv(f'{data_folder_path}{filepath}',
                           delim_whitespace=True,
                           skiprows=1,
                           names=column_names)

        # Adding 2450000 to the third column
        data['HJD'] = data['HJD_minus_2450000'] + 2450000
        data = data.drop(columns=['HJD_minus_2450000'])

        # Saving the modified data to a new CSV file
        output_txt_path = f'{data_output_folder_path}{event_number:03}_{filepath.split(".")[0]}_pylima.dat'
        data.to_csv(output_txt_path, index=False, sep=' ', header=False)


if __name__ == '__main__':
    rtmodel_data_to_pylima()
    print('Data files created successfully.')
