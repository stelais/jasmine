import pandas as pd

# Column names based on the file content
column_names = ['Mag', 'err', 'HJD_minus_2450000']

# Reading the data into a DataFrame, skipping the first line which is a header comment
for filepath in ['RomanW149sat1.dat', 'RomanZ087sat2.dat']:
    data = pd.read_csv(filepath, delim_whitespace=True, skiprows=1, names=column_names)

    # Adding 2450000 to the third column
    data['HJD'] = data['HJD_minus_2450000'] + 2450000
    data = data.drop(columns=['HJD_minus_2450000'])

    # Saving the modified data to a new CSV file
    output_txt_path = f'{filepath.split(".")[0]}_pylima.dat'
    data.to_csv(output_txt_path, index=False, sep=' ', header=False)
