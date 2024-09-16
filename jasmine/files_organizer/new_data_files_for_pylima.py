import pandas as pd


def rtmodel_data_to_pylima(folder_path='None'):
    # Column names based on the file content
    column_names = ['Mag', 'err', 'HJD_minus_2450000']
    if folder_path == 'None':
        folder_path =''
    else:
        folder_path = folder_path + '/'
    # Reading the data into a DataFrame, skipping the first line which is a header comment
    for filepath in [f'{folder_path}RomanW149sat1.dat', f'{folder_path}RomanZ087sat2.dat']:
        data = pd.read_csv(filepath, delim_whitespace=True, skiprows=1, names=column_names)

        # Adding 2450000 to the third column
        data['HJD'] = data['HJD_minus_2450000'] + 2450000
        data = data.drop(columns=['HJD_minus_2450000'])

        # Saving the modified data to a new CSV file
        output_txt_path = f'{filepath.split(".")[0]}_pylima.dat'
        data.to_csv(output_txt_path, index=False, sep=' ', header=False)


if __name__ == '__main__':
    rtmodel_data_to_pylima()
    print('Data files created successfully.')
