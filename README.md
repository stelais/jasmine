# jasmine
JASMINE: **J**oint **A**nalysis of **S**imulation for **M**icrolensing **IN**terested **E**vents

---
### Microlensing Data Challenge Simulations

#### For Splitting Master File
1. Make sure you downloaded the `master_file.txt` and `wfirstColumnNumbers.txt`  from the data challenge folder:  
   https://github.com/microlensing-data-challenge/data-challenge-1/tree/master/Answers
2. Run `python files_organizer/data_challenge_prep.py` , changing path as needed. It will split master file and create 4 new files:
   * binary_star.csv
   * bound_planet.csv
   * cataclysmic_variables.csv
   * single_lens.csv

#### For reading the 4 csv files
Call the function you need, and it will return a pandas dataframe:  
`import files_organizer.data_challenge_reader as dcr`
* `dataframe = dcr.binary_star_master_reader()`  
* `dataframe = dcr.bound_planet_master_reader()`  
* `dataframe = dcr.cataclysmic_variables_master_reader()`  
* `dataframe = dcr.single_lens_master_reader()`  

Obs: The column you are looking for is: `data_challenge_lc_number` . 