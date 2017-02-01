# DoS
Depth of Search

This Python package calculates the depth of search on a grid of semi-major axis and planetary radius for a selected target list. The primary function is ```DoS_calc.``` ```DoS_calc``` requires the following packages:

- ```numpy```
- ```os```
- ```sympy```
- ```scipy```
- ```astropy```
- ```cPickle``` or ```pickle```
- ```ortools```
- [```EXOSIMS```] (https://github.com/dsavransky/EXOSIMS) 

Unless the EXOSIMS default contrast value is desired, a fits file or constant value must be supplied for ```'core_contrast'``` in the json script file used to generate the ```EXOSIMS.MissionSim``` object.

### ```DoS_calc```

```DoS_calc``` takes the following arguments:

- ```path``` -> path to a json script file used to generate an ```EXOSIMS.MissionSim``` object
- ```abins``` -> number of semi-major axis bins for grid (optional)
- ```Rbins``` -> number of planetary radius bins for grid (optional)
- ```maxTime``` -> maximum total integration time in days (optional)

```DoS_calc``` returns:

- ```res``` -> dictionary containing results of depth of search calculations with the following keys:
  - ```'aedges'``` -> 1D ```numpy.ndarray``` containing bin edges of logarithmically spaced grid for semi-major axis in AU
  - ```'Redges'``` -> 1D ```numpy.ndarray``` containing bin edges of logarithmically spaced grid for planetary radius in R_earth
  - ```'NumObs'``` -> dictionary containing number of stars observed for each stellar type, keys include ```'Mstars'```, ```'Kstars'```, ```'Gstars'```, ```'Fstars'```, and ```'Entire'```
  - ```'DoS'``` -> dictionary containing 2D ```numpy.ndarray``` of depth of search values on grid corresponding to semi-major axis and planetary radius bins for each stellar type, keys include ```'Mstars'```, ```'Kstars'```, ```'Gstars'```, ```'Fstars'```, and ```'Entire'```
  - ```'occ_rates'``` -> dictionary containing 2D ```numpy.ndarray``` of occurrence rates extrapolated from Mulders 2015 on grid corresponding to semi-major axis and planetary radius bins for each stellar type, keys include ```'Mstars'```, ```'Kstars'```, ```'Gstars'```, and ```'Fstars'```
  - ```'DoS_occ'``` -> dictionary containing 2D ```numpy.ndarray``` of depth of search convolved with occurrence rates on grid corresponding to semi-major axis and planetary radius bins for each stellar type, keys include ```'Mstars'```, ```'Kstars'```, ```'Gstars'```, ```'Fstars'```, and ```'Entire'```
