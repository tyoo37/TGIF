# TGIF
Two d Gaussian in Fitting; use 2D gaussian fitting for photometry

This is a package dedicated to extract 2D gaussian source from ALMA images. The main fitting code uses LMFIT package (https://lmfit.github.io/lmfit-py/).
This tool can be useful for the image where sources are surrounded by neighboring structure or sources sitting on the artificial pattern.


One can measure the sizes along the major axis and the minor axis, and integrated fluxes.

## Installation

TGIF can be installed via pip.

``` pip install TGIF ```

## How to use
The main function is ```plot_and_save_fitting_results``` which makes plots and saves the fit results.
TGIF has two free parameters for the fitting--```fitting_size``` and ```maximum_size```.
```fitting_size``` controls the size of the cutout array used for the fitting in the units of the major axis size of the image beam. The default is 0.7.
```maximum_size``` is the maximum size of the 2D gaussian ellipse major axis in the units of the image beam size. The default is 4.
The test case shows ~90% of good fitting results with the default setting. If some of the sources show bad fitting result, one might need to tweak ```fitting_size```. 
This can be controlled by the parameter ```fitting_size_dict``` which has key for the index of sources and value for the ```fitting_size``` for the specific source.

One of the good way to use this function is just running the code with default setup and check the index of sources with bad fitting result and make a dictionary to put different ```fitting_size``` for specific sources.


Here's an example jupyter notebook.
```python
import TGIF.TGIF as tgif 
from radio_beam import Beam
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

import numpy as np

import TGIF.TGIF as tgif
import Paths.Paths as paths
from radio_beam import Beam
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

import numpy as np
Path = paths.filepaths()

def run_tgif(fitsfile, catalogfile, saveimgdir='', label_img='', savefitsdir='', fix_pos_idx=[], fitting_size_dict={}):
    """
    Run TGIF on the specified FITS file and catalog, saving results to PNG and FITS files.
    Parameters:
    fitsfile (str): Path to the FITS file containing the image data.
    catalogfile (str): Path to the catalog file containing peak positions of sources.
    saveimgdir (str): Path to the directory where the image showing the fitting result is saved.
    label_img (str): Label of the image file name. For example, the filename of the saved image will be (saveimgdir)/(label_img)_(six-digits index of the source).png
    savefitsdir (str) : Path to the directory where the fitting result table is saved.
    fix_pos_idx (list): Indices for sources which needs their peak positions to be fixed at the initial guess points. This is for sources with bad centering issue due to their faint peaks.
    fitting_size_dict (dict): Dictionary mapping indices to specific fitting sizes (default is empty dictionary)
    """

    fitsdata = fits.open(fitsfile)
    image = fitsdata[0].data
    if len(image.shape)>2:
        image= fitsdata[0].data[0][0]
   
    hdr = fits.getheader(fitsfile)  
    wcs = WCS(hdr,naxis=2)

    beam = Beam.from_fits_header(hdr)

    pixel_scale = wcs.proj_plane_pixel_scales()[0]

    catalog = Table.read(catalogfile)
    peakxy = np.vstack((catalog[f'{band}_xpix'], catalog[f'{band}_ypix'])).T

    peakxy_sky = np.vstack((catalog['b3_xsky'], catalog['b3_ysky'])).T


    tgif.plot_and_save_fitting_results(image, peakxy, beam, wcs, pixel_scale, fitting_size_default=0.6, saveimgdir=saveimgdir, label_img=label_img,
                                vmin=None, vmax=None, maximum_size=4, savefitsdir=savefitsdir,fix_pos_idx=fix_pos_idx, fitting_size_dict=fitting_size_dict)

```

## arguments for the main function ```plot_and_save_fitting_results```
```data```: 2d array of the image

```peakxy```: list of the pixel coordinates of the sources

```beam```: Beam object of the image beam

```wcsNB```: WCS object of the image

```pixel_scale```: pixel scale of the image in the unit of angle (astropy.Quantity)

```fitting_size_default```: default size of the fitting box in the unit of the beam major size

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

```deconvolved_angle```: the position angle of the deconvolved 2D gaussian model

```peak_flux``` : the peak flux of the 2D Gaussian model

```bad_centering```: flag for whether TGIF fails to find the exact peak

```least_chi_square``` : (for testing) chi-square values from the fitting


