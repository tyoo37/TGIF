# TGIF
Two d Gaussian in Fitting; use 2D gaussian fitting for photometry

This is a package dedicated to extract 2D gaussian source from ALMA images. 
This tool can be useful for the image where sources are surrounded by neighboring structure or sources sit on the artifical pattern.

One can measure the sizes along the major axis and the minor axis, and integrated fluxes.

## Installation

TGIF can be installed via pip.

``` pip install TGIF ```

## How to use
The main function is ```plot_and_save_fitting_results``` which makes plot and saves the fit result.
TGIF has two free parameters for the fitting--```fitting_size``` and ```maximum_size```.
```fitting_size``` controls the size of the cutout array used for the fitting in the units of the major axis size of the image beam. The default is 0.7.
```maximum_size``` is the maximum size of the 2D gaussian ellipse major axis in the units of the image beam size. The default is 4.
The test case shows ~90% of good fitting results with the default setting. If some of the sources show bad fitting result, one might need to tweak ```fitting_size```. 
This can be controlled by the parameter ```fitting_size_dict``` which has key for the index of sources and value for the ```fitting_size``` for the specific source.

One of the good way to use this function is just running the code with default setup and check the index of sources with bad fitting result and make a dictionary to put different ```fitting_size``` for specific sources.


Here's an example jupyter notebook.
```
import TGIF.TGIF as tgif 
import Paths.Paths as paths # This is for reading python file storing some file paths. you don't have to use this.
from radio_beam import Beam
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

import numpy as np
Path = paths.filepaths() # again, don't need to use this.

#loading image fits file
fitsdata_b3 = fits.open(Path.w51e_b3_cont_local)
image_b3 = fitsdata_b3[0].data
if len(image_b3.shape)>2:
    image_b3 = fitsdata_b3[0].data[0][0]
    
fitsdata_b6 = fits.open(Path.w51e_b6_cont_local)
image_b6 = fitsdata_b6[0].data

if len(image_b6.shape)>2:
    image_b6 = fitsdata_b6[0].data[0][0]

#header
hdrNB3 = fits.getheader(Path.w51e_b3_cont_local)  
hdrNB6 = fits.getheader(Path.w51e_b6_cont_local)

#wcs
wcsNB3 = WCS(hdrNB3,naxis=2)
wcsNB6 = WCS(hdrNB6,naxis=2)

my_beamNB6 = Beam.from_fits_header(hdrNB6)
my_beamNB3 = Beam.from_fits_header(hdrNB3)

scaleNB6 = wcsNB6.proj_plane_pixel_scales()[0]
scaleNB3 = wcsNB3.proj_plane_pixel_scales()[0]

# load the catalog file and assign the peak positions of each sources to start fitting
catalog = Table.read(Path.w51e_dendro_matched_catalog_local)
peakxy_b3 = np.vstack((catalog['b3_xpix'], catalog['b3_ypix'])).T
peakxy_b6 = np.vstack((catalog['b6_xpix'], catalog['b6_ypix'])).T

peakxy_sky_b3 = np.vstack((catalog['b3_xsky'], catalog['b3_ysky'])).T
peakxy_sky_b6 = np.vstack((catalog['b6_xsky'], catalog['b6_ysky'])).T

tgif.plot_and_save_fitting_results(image_b6, peakxy_b6, my_beamNB6, wcsNB6, scaleNB6, fitting_size=0.6, savedir='w51e_b6_test.fits',label='w51e_b6',
                                   vmin=None, vmax=None, maximum_size=4,
                                  fitting_size_dict={10: 1,
                                                    13: 1,
                                                    20: 1,
                                                    21: 1,
                                                    30: 1.5,
                                                    32: 0.5,
                                                    35: 0.5,
                                                    38: 1.5,
                                                    39: 2,
                                                    48: 1.5,
                                                    56: 0.6})
```


