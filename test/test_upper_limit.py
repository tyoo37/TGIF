import numpy as np
from regions import PixCoord, EllipsePixelRegion
from astropy.convolution import convolve, Gaussian2DKernel
import astropy.units as u
import TGIF.TGIF as tgif
import matplotlib.pyplot as plt

def generate_mock_image(image_beam, image_size, peak_height, bkg_value, bkg_mad, major_sig_pix, minor_sig_pix, pa, pixel_scale):
    """
    Generate a mock image with a single elliptical Gaussian source
    """
    #y,x = np.mgrid[:image_size[0],:image_size[1]]
   # print(major_sig_pix, minor_sig_pix)
    
    region_pix = EllipsePixelRegion(center=PixCoord(x=image_size[1]/2, y=image_size[0]/2), 
                       height = major_sig_pix.value, width = minor_sig_pix.value, angle=pa)
    mask = region_pix.to_mask(mode='exact')
    model_image = mask.to_image(image_size)
    sig_to_fwhm = 2*np.sqrt(2*np.log(2))
    kernel = Gaussian2DKernel(x_stddev = (image_beam.major / pixel_scale).to(u.deg/u.deg) / sig_to_fwhm, 
                              y_stddev = (image_beam.minor / pixel_scale).to(u.deg/u.deg) / sig_to_fwhm, 
                              theta = 180*u.deg - image_beam.pa)
    conv_image = convolve(model_image, kernel,preserve_nan=True)
    conv_image = conv_image/np.nanmax(conv_image) * peak_height
    image = conv_image + np.random.normal(bkg_value, bkg_mad, image_size)

    return image

def generate_random_disks(peak_height, bkg_val, bkg_mad, rad_arr,  pa_arr, incl_arr, image_beam, pixel_scale, wcs,
                          distance=5.41*u.kpc, image_size=(30,30)):
    deconv_major_arr = np.zeros((len(rad_arr), len(pa_arr), len(incl_arr)))
    deconv_minor_arr = np.zeros((len(rad_arr), len(pa_arr), len(incl_arr)))

    for i, rad in enumerate(rad_arr):
        rad_pix = (rad.to(u.au) * u.arcsec / distance.to(u.pc) / pixel_scale).to(u.dimensionless_unscaled)
        for j, pa in enumerate(pa_arr):
            for k, incl in enumerate(incl_arr):
                axis_ratio = np.sin(incl.to(u.rad))
                print(rad_pix, axis_ratio)
                print('ho',peak_height, bkg_val, bkg_mad)
                image = generate_mock_image(image_beam, image_size, peak_height, bkg_val, bkg_mad, rad_pix, rad_pix*axis_ratio, pa, pixel_scale)
                _, _, _, _, _, _, _, _, deconv_major, deconv_minor= tgif.plot_and_save_fitting_results(image, [image.shape[1]/2, image.shape[0]/2], image_beam, wcs, pixel_scale, 
                                                   fitting_size=0.6, make_plot=False) 
                #deconv_major = tab['deconvolved_major']
                #deconv_minor = tab['deconvolved_minor']
                deconv_major_arr[i,j,k] = deconv_major
                deconv_minor_arr[i,j,k] = deconv_minor
    return deconv_major_arr, deconv_minor_arr

def make_plot(deconv_major_arr, deconv_minor_arr, rad_arr, savedir=None):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    ax2 = fig.add_axes([0.5, 0.1, 0.4, 0.8])
    ax3 = ax1.twinx()
    ax4 = ax2.twinx()
    
    detection_rate_major_arr = []
    detection_rate_minor_arr = []
    for i in range(len(rad_arr)):
        ax1.scatter(rad_arr[i], deconv_major_arr[i,:,:])
        ax2.scatter(rad_arr[i], deconv_minor_arr[i,:,:])
        detection_rate_major = len(np.where(np.isfinite(deconv_major_arr[i,:,:]))[0])/len(deconv_major_arr[i,:,:])
        detection_rate_minor = len(np.where(np.isfinite(deconv_minor_arr[i,:,:]))[0])/len(deconv_minor_arr[i,:,:])
        detection_rate_major_arr.append(detection_rate_major)
        detection_rate_minor_arr.append(detection_rate_minor)
    ax3.plot(rad_arr, detection_rate_major_arr, marker='o', c='r')
    ax4.plot(rad_arr, detection_rate_minor_arr, marker='o', c='r')

    ax1.set_xlabel('Radius (au)')
    ax1.set_ylabel('Deconvolved Major (au)')
    ax2.set_xlabel('Radius (au)')
    ax2.set_ylabel('Deconvolved Minor (au)')
    ax3.set_ylabel('Detection Rate')
    ax4.set_ylabel('Detection Rate')
    plt.savefig(savedir)
    plt.show()
    plt.close()

   

def test_upper_limit(rad_arr, pa_arr, incl_arr, data, peakxy, beam, wcsNB, pixel_scale,
                         fitting_size = 4, distance=5.41*u.kpc,
                        
                        bkg_inner_width=4, bkg_annulus_width=2, bkg_inner_height=4, bkg_annulus_height=2, maximum_size=4,
                        label=None, savedir='./',
                        fix_pos_idx=[],fitting_size_dict={}):
    num_source = len(peakxy[0])
    for i in range(num_source):
        if peakxy[i,0]<0 or peakxy[i,1]<0 :
            continue
        
        if i in fitting_size_dict:
            fitting_size = fitting_size_dict[i]

        if i in fix_pos_idx:
            do_subpixel_adjust = False
        else:
            do_subpixel_adjust = True    
        positions_original = (peakxy[i,0], peakxy[i,1])
        positions = tgif.redefine_center(data, positions_original)
        results, xcen_fit_init, ycen_fit_init, peak_fit_init = tgif.fit_for_individuals(positions, data, wcsNB, beam, pixel_scale, 
                                                                                                 subpixel_adjust_angle=180*u.deg-beam.pa, plot=False, 
                                                                                                 fitting_size=fitting_size, maximum_size=maximum_size,report_fit=False, do_subpixel_adjust=do_subpixel_adjust)
        popt = results.params
        xcen_init = xcen_fit_init
        ycen_init = ycen_fit_init
        pa_init = popt['theta'] * 180 / np.pi
        fitted_major_init = popt['sigma_x']
        fitted_minor_init = popt['sigma_y']
       # print('xcen_init,ycen_init,pa_init,fitted_major_init,fitted_minor_init',xcen_init,ycen_init,pa_init,fitted_major_init,fitted_minor_init)
        bkg, bkg_mad = tgif.get_local_bkg(data, xcen_init, ycen_init, pa_init, peakxy, wcsNB, beam, pixel_scale,
                                     inner_width=bkg_inner_width*fitted_major_init, outer_width=(bkg_inner_width+bkg_annulus_width)*fitted_major_init, 
                                     inner_height=bkg_inner_height*fitted_major_init, outer_height=(bkg_inner_height+bkg_annulus_height)*fitted_major_init)
        deconv_major_arr, deconv_minor_arr = generate_random_disks(peak_fit_init, bkg, bkg_mad, rad_arr, pa_arr, incl_arr, beam, pixel_scale, wcsNB, distance=distance, image_size=(30,30))
        make_plot(deconv_major_arr, deconv_minor_arr, rad_arr, savedir=savedir+'upper_limit_%s_%05d.png'%(label,i))




