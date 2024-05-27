# jasmine
JASMINE: **J**oint **A**nalysis of **S**imulation for **M**icrolensing **IN**terested **E**vents

---
pip installable as `jasmine-astro`
```
pip install jasmine-astro
```
---
## 1. Reading RTModel outputs
### `ModelResults` class
```
from jasmine import ModelResults
model = ModelResults(file_to_be_read='[your_path]/[Final]Models/LX0000-1.txt')
print(model.model_type, model.model_extensive_name)
print(model.model_parameters)
```
See notebook `analysis/reading_rtmodel_models.ipynb`

---
## 2. Generating a Binary lens signal based on one of the 113 RTModel templates
### `RTModelTemplateForBinaryLightCurve` class
```
from jasmine import RTModelTemplateForBinaryLightCurve
rtmodel_template_two_lenses = RTModelTemplateForBinaryLightCurve(template_line=2,
                                                                     path_to_template=template_path,
                                                                     input_peak_t1=300,
                                                                     input_peak_t2=302)
magnification, times = rtmodel_template_two_lenses_.rtmodel_magnification_using_vbb()
```


---
## 3. Microlensing Data Challenge Simulations

### Splitting master file
1. Make sure you downloaded the `master_file.txt` and `wfirstColumnNumbers.txt`  from the data challenge folder:  
   https://github.com/microlensing-data-challenge/data-challenge-1/tree/master/Answers
2. Run `python jasmine/files_organizer/data_challenge_prep.py` , changing path as needed. It splits master file and create 4 new files:
   * binary_star.csv
   * bound_planet.csv
   * cataclysmic_variables.csv
   * single_lens.csv
---
###  Using the `LightcurveEventDataChallenge` class:  
```
from jasmine import LightcurveEventDataChallenge
the_lightcurve = LightcurveEventDataChallenge(2) # Binary star # Call the lightcurve class
vars(the_lightcurve).keys() # See what are the available attributes and subclasses
the_lightcurve.lens # subclass
lightcurve_datapoints = the_lightcurve.lightcurve_data(filter_='W149', folder_path_='../data') # Get the lightcurve datapoints
```

See notebook `analysis/getting_information_about_a_lightcurve.ipynb` for details.

---
If you opt to not use a class. You can use the functions below:
See notebook `analysis/reading_the_data_challenge.ipynb` for more details.
#### 1. Reading the four master csv files
Call the function you need, and it returns a pandas dataframe:  
`import jasmine.files_organizer.data_challenge_reader as dcr`
* `dataframe = dcr.binary_star_master_reader()`  
* `dataframe = dcr.bound_planet_master_reader()`  
* `dataframe = dcr.cataclysmic_variables_master_reader()`  
* `dataframe = dcr.single_lens_master_reader()`  

Obs: The column you are looking for is: `data_challenge_lc_number`.  

#### 2. Reading the light curve data points files
The function `lightcurve_data_reader` reads the light curve files and returns a pandas dataframe with `BJD`, `Magnitude`, `Error` and `days` (days = BJD - 2450000) ':  

```
import jasmine.files_organizer.data_challenge_reader as dcr
lightcurve_df = dcr.lightcurve_data_reader(data_challenge_lc_number_=5, folder_path_='../data')
```
