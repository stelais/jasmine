# jasmine
JASMINE: **J**oint **A**nalysis of **S**imulation for **M**icrolensing **IN**terested **E**vents

---
### 1. Microlensing Data Challenge Simulations

#### 1.1 Splitting master file
1. Make sure you downloaded the `master_file.txt` and `wfirstColumnNumbers.txt`  from the data challenge folder:  
   https://github.com/microlensing-data-challenge/data-challenge-1/tree/master/Answers
2. Run `python files_organizer/data_challenge_prep.py` , changing path as needed. It splits master file and create 4 new files:
   * binary_star.csv
   * bound_planet.csv
   * cataclysmic_variables.csv
   * single_lens.csv
---
See notebook `analysis/reading_the_data_challenge.ipynb` for more details.
#### 1.2. Reading the four master csv files
Call the function you need, and it returns a pandas dataframe:  
`import files_organizer.data_challenge_reader as dcr`
* `dataframe = dcr.binary_star_master_reader()`  
* `dataframe = dcr.bound_planet_master_reader()`  
* `dataframe = dcr.cataclysmic_variables_master_reader()`  
* `dataframe = dcr.single_lens_master_reader()`  

Obs: The column you are looking for is: `data_challenge_lc_number`.  

#### 1.3 Reading the light curve files
The function `lightcurve_data_reader` reads the light curve files and returns a pandas dataframe with `BJD`, `Magnitude`, `Error` and `days` (days = BJD - 2450000) ':  

```
import files_organizer.data_challenge_reader as dcr
lightcurve_df = dcr.lightcurve_data_reader(data_challenge_lc_number_=5, folder_path_='../data')
```

#### 1.4 Reading the microlensing parameters
How to use the `LightcurveEvent` class:  
```   
import files_organizer.lightcurve_cls as lc
# Call the lightcurve class
the_lightcurve = lc.LightcurveEvent(2) # Binary star
# See what are the available dictionaries
the_lightcurve.print_dictionaries()
# Get info from that dict
the_lightcurve.lens_dict['lens_system_mass__msun']
# Get the lightcurve datapoints
lightcurve_datapoints = the_lightcurve.lightcurve_data(filter_='W149', folder_path_='../data')
```

See notebook `analysis/getting_information_about_a_lightcurve.ipynb` for more details.
