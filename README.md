# DoS
Depth-of-Search

This Python package calculates the depth-of-search on a grid of semi-major axis and planetary radius for a selected target list. 

The ```Scripts``` folder contains examples of how to calculate depth-of-search with and without the ```DoSFuncs``` class object.
See the individual scripts for a description of their use.

The ```DoSFuncs``` class object requires the following packages:

- ```numpy```
- ```os```
- ```sympy```
- ```scipy```
- ```astropy```
- ```cPickle``` or ```pickle```
- ```ortools```
- [```EXOSIMS```] (https://github.com/dsavransky/EXOSIMS) 
- ```matplotlib```

Unless the EXOSIMS default contrast value is desired, a fits file or constant value must be supplied for ```'core_contrast'``` in the json script file used to generate the ```EXOSIMS.MissionSim``` object.

### ```DoSFuncs``` class object initialization and attributes

##### ```DoSFuncs``` class object arguments:

- ```path``` -> path to a json script file used to generate an ```EXOSIMS.MissionSim``` object
- ```abins``` -> number of semi-major axis bins for grid (optional-default is 100)
- ```Rbins``` -> number of planetary radius bins for grid (optional-default is 30)
- ```maxTime``` -> maximum total integration time in days (optional-default is 365)
- ```intCutoff``` -> maximum integration time for a single target in days (optional-default is 30)
- ```WA_targ``` -> target working angle for instrument contrast (astropy Quantity) if not specified, DoSFuncs finds the working angle for minimum contrast to use in integration time calculations

##### ```DoSFuncs``` class object attributes:

- ```result``` -> dictionary containing results of depth-of-search calculations with the following keys:
  - ```'aedges'``` -> 1D ```numpy.ndarray``` containing bin edges of logarithmically spaced grid for semi-major axis in AU
  - ```'Redges'``` -> 1D ```numpy.ndarray``` containing bin edges of logarithmically spaced grid for planetary radius in R_earth
  - ```'NumObs'``` -> dictionary containing number of stars observed for each stellar type (```DoSFuncs``` key is ```'all'```, ```DoSFuncsMulders``` keys include: ```'Mstars'```, ```'Kstars'```, ```'Gstars'```, ```'Fstars'```, and ```'all'```)
  - ```'DoS'``` -> dictionary containing 2D ```numpy.ndarray``` of depth-of-search values on grid corresponding to semi-major axis and planetary radius bins for each stellar type (```DoSFuncs``` key is ```'all'```, ```DoSFuncsMulders``` keys include: ```'Mstars'```, ```'Kstars'```, ```'Gstars'```, ```'Fstars'```, and ```'all'```)
  - ```'occ_rates'``` -> dictionary containing 2D ```numpy.ndarray``` of occurrence rates from EXOSIMS (or extrapolated from Mulders 2015 with ```DoSFuncsMulders```) on grid corresponding to semi-major axis and planetary radius bins for each stellar type (```DoSFuncs``` key is ```'all'```, ```DoSFuncsMulders``` keys include: ```'Mstars'```, ```'Kstars'```, ```'Gstars'```, ```'Fstars'```, and ```'all'```)
  - ```'DoS_occ'``` -> dictionary containing 2D ```numpy.ndarray``` of depth-of-search convolved with occurrence rates on grid corresponding to semi-major axis and planetary radius bins for each stellar type (```DoSFuncs``` key is ```'all'```, ```DoSFuncsMulders``` keys include: ```'Mstars'```, ```'Kstars'```, ```'Gstars'```, ```'Fstars'```, and ```'all'```)
- ```sim``` -> ```EXOSIMS.MissionSim``` object used to generate the target list and integration times
- ```outspec``` -> dictionary containing ```EXOSIMS.MissionSim``` output specifications

### ```DoSFuncs``` Methods

Methods for quickly displaying depth-of-search results and saving them to disk are also included.

##### ```plot_dos```
Plots the depth-of-search as a filled contour plot with contour lines (color in log scale)

Args:
- ```targ``` -> string indicating which key to access from depth-of-search result dictionary (e.g., 'all')
- ```name``` -> string indicating what to include in figure title (e.g., 'All Stars')
- ```path``` -> string for path to save figure as pdf to disk (optional) (e.g., '.../DoS.pdf')

##### ```plot_nplan```
Plots the depth-of-search convolved with occurrence rates as a filled contour plot with contour lines (color in log scale)

Args:
- ```targ``` -> string indicating which key to access from depth-of-search result dictionary (e.g., 'Mstars')
- ```name``` -> string indicating what to include in figure title (e.g., 'M Stars')
- ```path``` -> string for path to save figure as pdf to disk (optional) (e.g., '.../nplan.pdf')

##### ```save_results```
Saves the results and ```EXOSIMS.MissionSim``` outspec as a pickled dictionary to disk

Args:
- ```path``` -> string for path to save results

Results are saved in the specified path as a pickled dictionary with keys ```'Results'``` and ```'outspec'``` containing the ```result``` and ```outspec``` attributes respectively.

##### ```save_json```
Saves the output json script to disk

Args:
- ```path``` -> string for path to save json script

Results are saved in the specified path.

##### ```save_csvs```
Saves results as individual csv files to disk

Args:
- ```directory``` -> string for directory path to save results

Results are saved as:
- '.../aedges.csv'
- '.../Redges.csv'
- '.../NumObs.csv'
- '.../DoS_all.csv', etc
- '.../occ_rates_Mstars.csv', etc
- '.../DoS_occ_Gstars.csv', etc
