# TGIF
Two d Gaussian in Fitting; use 2D gaussian fitting for photometry

This is a package dedicated to extract 2D gaussian source from ALMA images. The main fitting code uses LMFIT package (https://lmfit.github.io/lmfit-py/).
This tool can be useful for the image where sources are surrounded by neighboring structure or sources sitting on the artificial pattern.


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
```python
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

#header
hdrNB3 = fits.getheader(Path.w51e_b3_cont_local)  #header of the image fits

#wcs
wcsNB3 = WCS(hdrNB3,naxis=2)
#image beam
my_beamNB3 = Beam.from_fits_header(hdrNB3)
#pixel_scale
scaleNB3 = wcsNB3.proj_plane_pixel_scales()[0]

# load the catalog file and assign the peak positions of each sources to start fitting
catalog = Table.read(Path.w51e_dendro_matched_catalog_local)
peakxy_b3 = np.vstack((catalog['b3_xpix'], catalog['b3_ypix'])).T #pixel coordinates of sources

# main function for savning and plotting the fitting results
tgif.plot_and_save_fitting_results(image_b3, peakxy_b3, my_beamNB3, wcsNB3, scaleNB3, fitting_size_default=0.6, saveimgdir='image_new/',label_img='w51e_b3',
                                   vmin=None, vmax=None, maximum_size=4,savefitsdir='/home/t.yoo/w51/catalogue/photometry/flux_new/', label_fits='w51e_b3_test',
                                  fitting_size_dict={10: 1,
                                                    13: 1,
                                                    20: 1,
                                                    21: 1.2,
                                                    30: 1.5,
                                                    32: 0.5,
                                                    35: 0.5,
                                                    38: 1.5,
                                                    39: 2,
                                                    48:0.5,
                                                    56: 0.6,
                                                    66: 0.5,
                                                    72:1,
                                                    73:0.3,
                                                    74: 3.,
                                                    76: 3,
                                                    124: 1})
```

## arguments for the main function
```data```: 2d array of the image

```peakxy```: list of the pixel coordinates of the sources

```beam```: Beam object of the image beam

```wcsNB```: WCS object of the image

```pixel_scale```: pixel scale of the image

```fitting_size_default```: default size of the fitting box

```issqrt```: if True, the image is shown in sqrt scale (default: True)

```vmin```: minimum value for the color scale of the image (default: None)

```vmax```: maximum value for the color scale of the image (default: None)

```flux_unit```: unit of the flux (default: 'Jy/beam')

```do_subpixel_adjust```: if True, the subpixel adjustment is done (default: True)

```bkg_inner_width```: inner width of the background annulus in the unit of pixel (default: 4)

```bkg_annulus_width```: outer width of the background annulus in the unit of pixel (default: 2)

```bkg_inner_height```: inner height of the background annulus in the unit of pixel (default: 4)

```bkg_annulus_height```: outer height of the background annulus in the unit of pixel (default: 2)

```maximum_size```: maximum size of the fitting cutout in the unit of pixel (default: 4)

```saveimgdir```: directory to save the fitting result image (default: None)

```savefitsdir```: directory to save the fitting result table in the fits format (default: None)

```make_plot```: if True, the fitting results are plotted (default: True)

```show```: if True, the fitting results are shown (default: True)

```label_img```: label of the image file name. The directory of the image will be saveimgdir+label_img+'_%06d.png'%idx. (default: None)

```label_fits```: label of the fits file. The directory of the fits file will be savefitsdir+'%s.fits'%label_fits. (default: 'w51e_b6_test')

```fix_pos_idx```: list of the index of the source to fix the position. Useful when no significant peak is found at the original position of the source. (default: [])

```fitting_size_dict```: dictionary of the fitting size for each source. The key is the index of the source and the value is the size of the fitting box. (default: {})

```idx```: index of the source (default: 0)

```subpixel_adjust_limit```: the maximum limit of the subpixel adjustment in units of pixel.  (default: 4)

## outputs
```plot_and_save_fitting_results``` generates the image (when ```plot`` is ```True```) and the table containing the fitting results.

The tables contains following information:

```flux```: flux of sources in the same unit as the unit of flux in the image.

```flux_err``` : flux error based on LMFIT fitting results

```pa```: position angle of fitted 2D gaussian model

```pa_err```: error of ```pa``` based on LMFIT fitting results

```fitted_major```: FWHM of Gaussian width along the major axis in the same unit of the pixel scale

```fitted_minor```: FWHM of Gaussian width along the minor axis in the same unit of the pixel scale

```deconvolved_major```: FWHM of Gaussian width along the major axis deconvolved to the image beam. Usually used in estimating the physical size. same unit of the pixel scale.

```deconvolved_minor```: FWHM of Gaussian width along the minor axis deconvolved to the image beam. 

```peal_flux``` : the peak flux of the 2D Gaussian model




