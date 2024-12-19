import scipy.ndimage
from astropy.stats import mad_std
import matplotlib as mpl
from scipy.optimize import curve_fit
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from regions import EllipsePixelRegion
from regions.shapes.circle import CirclePixelRegion
from regions.core import PixCoord
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table, MaskedColumn
from radio_beam import Beam
from astropy import coordinates
from astropy import wcs
from astropy.nddata.utils import Cutout2D
from matplotlib.patches import Ellipse
import regions
from regions import  EllipseAnnulusPixelRegion
import lmfit
from lmfit.lineshapes import gaussian2d
import pandas as pd


def gaussian(x, mu, sig, norm):
    return norm*np.exp(-np.power((x - mu)/sig, 2.)/2)


def gaussian2d(x, y, x_center, y_center, norm, theta=0, sigma_x = 10, sigma_y=10, sigma_diff=0, shift=0):
    # x_center and y_center will be the center of the gaussian, theta will be the rotation angle
    # sigma_x and sigma_y will be the stdevs in the x and y axis before rotation
    # x_size and y_size give the size of the frame 
    # theta is clockwisely measured from -x axis 

    #theta = 2*np.pi*theta/360 
   # x = np.arange(0,x_size, 1, float)
   # y = np.arange(0,y_size, 1, float)
   # y = y[:,np.newaxis]
    sx = sigma_x
    sy = sigma_y
    x0 = x_center
    y0 = y_center

    # rotation
    a=np.cos(np.pi-theta)*x -np.sin(np.pi-theta)*y
    b=np.sin(np.pi-theta)*x +np.cos(np.pi-theta)*y
    a0=np.cos(np.pi-theta)*x0 -np.sin(np.pi-theta)*y0
    b0=np.sin(np.pi-theta)*x0 +np.cos(np.pi-theta)*y0

    return norm*np.exp(-(((a-a0)**2)/(2*(sx**2)) + ((b-b0)**2) /(2*(sy**2))))+shift

def bbox_contains_bbox(bbox1,bbox2):
    """returns true if bbox2 is inside bbox1"""
    return ((bbox1.ixmax>bbox2.ixmax) & (bbox1.ixmin<bbox2.ixmin) &
            (bbox1.iymax>bbox2.iymax) & (bbox1.iymin<bbox2.iymin))

def sub_bbox_slice(bbox1, bbox2):
    """returns a slice from within bbox1 of bbox2"""
    if not bbox_contains_bbox(bbox1, bbox2):
        raise ValueError("bbox2 is not within bbox1")
    x0, dx = bbox2.ixmin-bbox1.ixmin, bbox2.ixmax-bbox2.ixmin
    y0, dy = bbox2.iymin-bbox1.iymin, bbox2.iymax-bbox2.iymin
    return (slice(y0, y0+dy), slice(x0, x0+dx),)

def slice_bbox_from_bbox(bbox1, bbox2):
    """
    Utility tool. Given two bboxes in the same coordinates, give the views of
    each box corresponding to the other.  For example, if you have an image
    ``im`` and two overlapping cutouts from that image ``cutout1`` and
    ``cutout2`` with bounding boxes ``bbox1`` and ``bbox2``, the returned views
    from this function give the regions ``cutout1[view1] = cutout2[view2]``
    """

    if bbox1.ixmin < bbox2.ixmin:
        blcx = bbox2.ixmin
    else:
        blcx = bbox1.ixmin
    if bbox1.ixmax > bbox2.ixmax:
        trcx = bbox2.ixmax
    else:
        trcx = bbox1.ixmax
    if bbox1.iymin < bbox2.iymin:
        blcy = bbox2.iymin
    else:
        blcy = bbox1.iymin
    if bbox1.iymax > bbox2.iymax:
        trcy = bbox2.iymax
    else:
        trcy = bbox1.iymax

    y0_1 = max(blcy-bbox1.iymin,0)
    x0_1 = max(blcx-bbox1.ixmin,0)
    y0_2 = max(blcy-bbox2.iymin,0)
    x0_2 = max(blcx-bbox2.ixmin,0)

    dy_1 = min(bbox1.iymax-blcy,trcy-blcy)
    dx_1 = min(bbox1.ixmax-blcx,trcx-blcx)
    dy_2 = min(bbox2.iymax-blcy,trcy-blcy)
    dx_2 = min(bbox2.ixmax-blcx,trcx-blcx)

    view1 = (slice(y0_1, y0_1+dy_1),
             slice(x0_1, x0_1+dx_1),)
    view2 = (slice(y0_2, y0_2+dy_2),
             slice(x0_2, x0_2+dx_2),)
   
    for slc in view1+view2:
        assert slc.start >= 0
        assert slc.stop >= 0
   
    return view1,view2

def subpixel_adjustment(profile1d, distarr, inclination, vec_from_cen, numpix_adjust=11,
                        verbose=False, isstrict=False,small_value=2e-7, tolerance=5, test_plot=False,):
    isnotfound = False

    half_numpoints = int(len(profile1d)/2)
    left_to_peak = half_numpoints-int(numpix_adjust/2)
    right_to_peak = half_numpoints+int(numpix_adjust/2)
    if left_to_peak<0:
        print(left_to_peak, half_numpoints, numpix_adjust)
        raise ValueError('left_to_peak cannot be negative')
    if right_to_peak>len(profile1d):
        print(right_to_peak, half_numpoints, numpix_adjust)
        raise ValueError('right_to_peak cannot be larger than the length of the profile')
    profiles_around_peak = profile1d[left_to_peak:right_to_peak+1]
    pixels_around_peak = distarr[left_to_peak:right_to_peak+1]
    
    isfinite = np.isfinite(profiles_around_peak)
    insideind = profiles_around_peak[isfinite]>-10
    try:
        profiles_around_peak = profiles_around_peak[isfinite][insideind]
    except:
        print(profiles_around_peak, insideind)
        print('error in subpixel_adjustment')
        isnotfound = True
    pixels_around_peak = pixels_around_peak[isfinite][insideind]
    try:
        peakpos = pixels_around_peak[np.argmax(profiles_around_peak)]
    except:
        print(profile1d, half_numpoints, numpix_adjust,profiles_around_peak, insideind)
        print('error in subpixel_adjustment')
        isnotfound = True

    distarr_step = np.abs(distarr[1]-distarr[0])

    try:
        popt, pcov = curve_fit(gaussian, pixels_around_peak, profiles_around_peak, 
                               bounds=((peakpos-distarr_step/2, 0, np.max(profiles_around_peak)-small_value),
                                        (peakpos+distarr_step/2, np.inf,0.999999999*np.max(profiles_around_peak))))
        adjusted_offset = popt[0]
        adjusted_peakval = popt[2]                                                                               
    except:
       # print('fitting failed in subpixel adjustment')
       # print(np.max(profiles_around_peak)-small_value, 0.999999999*np.max(profiles_around_peak))
        adjusted_offset=0
        adjusted_peakval=profile1d[half_numpoints]-small_value
    """
    if np.abs(adjusted_offset)>tolerance:
        print('the peak is made at another pixel')
        adjusted_offset=0
        adjusted_peakval=profiles_around_peak[int(len(profiles_around_peak)/2)]-small_value
    """
        
    """
    if isstrict and adjusted_offset>np.sqrt(2)/2:
        print(adjusted_offset)
        raise ValueError('the peak is made at another pixel')
    
    except:
        print('subpixel adjustment failed')
        pcov=0
        adjusted_offset = 0
        adjusted_peakval = peakval
    """
    adjusted_offset_x = adjusted_offset * np.cos(inclination)
    adjusted_offset_y = adjusted_offset * np.sin(inclination)

    xoffset = vec_from_cen[0] + adjusted_offset_x
    yoffset = vec_from_cen[1] + adjusted_offset_y

    

    #print('peak in adjust', adjusted_peakval, np.max(profiles_around_peak))
    """
    if any([np.abs(adjusted_offset_x)>0.5 , np.abs(adjusted_offset_y)>0.5]):
        print(adjusted_offset_x,adjusted_offset_y)
        raise ValueError('offset cannot excceed 0.5')
    """
    if test_plot:
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_axes([0,0,1,1])
        ax1.plot(distarr, profile1d, label='1D profile',lw=4)
        ax1.plot(distarr, gaussian(distarr, *popt), label='fit',lw=4)
        ax1.legend()
        plt.show()
        plt.close()
    if np.abs(xoffset) > 0.5 or np.abs(yoffset) > 0.5:
        pass
        #print('the peak is made at another pixel')
    if verbose:
        pass  
      #  print('adjusted_offset, adjusted_offset_x, adjusted_offset_y, adjusted_peakval, pcov = ', 
    #     adjusted_offset, adjusted_offset_x, adjusted_offset_y, adjusted_peakval, pcov)
    #update xcen,ycen
    if isnotfound:
        adjusted_peakval = profile1d[half_numpoints]
        adjusted_offset_x = 0
        adjusted_offset_y = 0
        adjusted_offset = 0

    return adjusted_offset_x, adjusted_offset_y, adjusted_offset, adjusted_peakval

def get_local_bkg(data, xcen, ycen, angle, peakxy_all, wcsNB, beam, pixel_scale, 
                  inner_width=1, outer_width=2, inner_height=1, outer_height=2, issky=False, filter_bright_pixels=False, plot=False, n_expected_neighbor=10):
    """
    Get the local background for a given source
    
    args
    -----
    data: 2D array

    xcen, ycen: float
        the pixel coordinates of source in the total image
    peakxy_all: 2D array
        the pixel coordinates of the peak of the source
    wcsNB: WCS
        the wcs of the image
    beam: radio_beam.Beam
        the beam of the image
    pixel_scale: astropy.unit
        the pixel scale of the image
    inner_width: float
        the inner width of the ellipse
    outer_width: float
        the outer width of the ellipse
    inner_height: float
        the inner height of the ellipse
    outer_height: float
        the outer height of the ellipse
    angle: astropy.unit
        the angle of the ellipse
    issky: bool
        issky is True if the input peakxy_all is in sky coordinates
    filter_bright_pixels: bool 
        filter_bright_pixels is True if you want to filter out the bright pixels in the source region
    
    return
    -------
    background: float
        the median of the background
    mad: float
        the median absolute deviation of the background
    """

    
    
        
    bkg_region = EllipseAnnulusPixelRegion(center=PixCoord(x=xcen, y=ycen),
                                       inner_width=inner_width,
                                       outer_width=outer_width,
                                       inner_height=inner_height,
                                       outer_height=outer_height,
                                       angle=angle*u.deg)
    

    mask = bkg_region.to_mask()
    data_in_cutout = mask.cutout(data)
    masked_data = data_in_cutout * mask.data
    maskbool = mask.data.astype('bool')

    beam_major = beam.major
    beam_minor = beam.minor
    beam_pa = beam.pa

    # masking the position where other neighboring sources lie
    if peakxy_all is not None:

        dist = np.sqrt((peakxy_all[:,0]-xcen)**2+(peakxy_all[:,1]-ycen)**2)
        peakxy_sorted = peakxy_all[np.argsort(dist)]
        peakxy_sorted_selected = peakxy_sorted[:n_expected_neighbor]
    

        
        if issky:
            cen_pix = wcsNB.wcs_world2pix(peakxy_sorted_selected,0)
            cen_world = peakxy_sorted_selected
        else:
            cen_pix = peakxy_sorted_selected
            cen_world = wcsNB.wcs_pix2world(peakxy_sorted_selected,0)
    
  
        
        positions_all = coordinates.SkyCoord([[cen_world[i,0],cen_world[i,1]] for i in range(n_expected_neighbor)], frame=wcs.utils.wcs_to_celestial_frame(wcsNB).name,unit=(u.deg,u.deg))

   
        test_region = EllipseAnnulusPixelRegion(center=PixCoord(x=xcen, y=ycen),
                                        outer_width=outer_width+beam_major.value/pixel_scale.value,
                                        inner_width=inner_width-beam_major.value/pixel_scale.value,
                                        outer_height=outer_height+beam_minor.value/pixel_scale.value,
                                        inner_height=inner_height-beam_minor.value/pixel_scale.value,
                                        angle=angle*u.deg)
        
        nearby_matches =[PixCoord(peakxy_sorted_selected[i,0],peakxy_sorted_selected[i,1]) in test_region for i in range(n_expected_neighbor)]
 
        if any(nearby_matches):
            
            inds = np.where(nearby_matches)[0].tolist()
            if len(inds)>1:
                dist = np.sqrt((peakxy_sorted_selected[inds,0]-xcen)**2+(peakxy_sorted_selected[inds,1]-ycen)**2)
                myself = np.argmin(dist)
                #print(inds, len(inds),myself)
                inds.remove(inds[myself])
                for ind in inds:
                    
                    maskoutreg = regions.EllipseSkyRegion(center=positions_all[ind], width=2*beam_major,
                                                            height=2*beam_minor,
                                                            angle=180*u.deg-beam.pa)
                    mpixreg = maskoutreg.to_pixel(wcsNB)
                    mmask = mpixreg.to_mask()
                    view, mview = slice_bbox_from_bbox(mask.bbox, mmask.bbox)
                    maskbool[view] &= ~mmask.data.astype('bool')[mview]
                    masked_data = masked_data * maskbool
    background_mask = maskbool.copy().astype('bool')
    #background_mask[sub_bbox_slice(mask.bbox, smaller_mask.bbox)] &= ~smaller_mask.data.astype('bool')
    masked_cutout = masked_data[background_mask]

    if filter_bright_pixels:
        source_region = EllipsePixelRegion(center=PixCoord(x=xcen, y=ycen),
                                       width=inner_width,
                                       height=inner_height,
                                       angle=angle)
        source_mask = source_region.to_mask()
        source_mask_cutout = source_mask.cutout(data)
        thres = np.min(source_mask_cutout)
       # print('thres',thres)
        masked_cutout = masked_cutout[np.where((masked_cutout<thres)&(np.isfinite(masked_cutout)))]
    background = np.nanmedian(masked_cutout)
    mad = mad_std(masked_cutout, ignore_nan=True)
    if plot:
        fig = plt.figure(figsize=(21,7))
        ax1 = fig.add_axes([0,0,0.33,1])
        ax2 = fig.add_axes([0.33,0,0.33,1])
        ax3 = fig.add_axes([0.66,0,0.33,1])
        ax1.imshow(data_in_cutout, origin='lower', cmap=plt.get_cmap('inferno'), norm=colors.PowerNorm(gamma=0.5,
                                            vmin=0,vmax=np.nanmax(masked_cutout)))
        artist = test_region.as_artist()
        ax1.add_artist(artist)
        ax2.imshow(maskbool,origin='lower', cmap=plt.get_cmap('inferno'))
        ax3.imshow(masked_data, origin='lower', cmap=plt.get_cmap('inferno'), norm=colors.PowerNorm(gamma=0.5, vmin=0, vmax=np.nanmax(masked_data)))
        plt.show()
        plt.close()
    
    
    return background, mad



def residual(params, x, y, image, x_center, y_center, norm, rad=5, lambda_factor=10):
    """
    this residual function is the function that is required to be minimized
    
    args
    -----
    params: lmfit.Parameters
        the parameters of the gaussian model
    x, y: 2D array
        the pixel coordinates of the image
    image: 2D array
        the image data (recommended to use very small central cutout of the image for better fit)
    x_center, y_center: float
        the pixel coordinates of the center of the gaussian
    norm: float
        the normalization of the gaussian
    rad: float
        the radius of the circle region
    lambda_factor: float
        a factor to control the degree of suppression of the negative residuals
    
    return
    -------
    masked_offset: 2D array
        the masked residual of the image
    
    """
    model = gaussian2d(x, y, x_center, y_center, norm, **params)
    center = PixCoord(x_center, y_center)
    yy, xx = np.mgrid[:model.shape[0], :model.shape[1]]
    dist = np.sqrt((xx-x_center)**2+(yy-y_center)**2)
    reg = CirclePixelRegion(center,rad)
    mask = reg.to_mask()
    offset = image-model
    #offset[offset>0] = offset[offset>0]/lambda_factor
    #masked_offset[masked_offset<0] = masked_offset[masked_offset<0] * 100000
    offset[offset<0] = offset[offset<0]*np.exp(lambda_factor*np.abs(offset[offset<0]))
    masked_offset = mask.multiply(offset)

    #print('size',len(masked_offset[np.isfinite(masked_offset)]))
    return masked_offset[np.isfinite(masked_offset)]
    

def fit_for_individuals(positions, data, wcsNB, beam, pixel_scale, subpixel_adjust_angle=0*u.deg, fitting_size=1, background=None, plot=False, report_fit=True,
                        subpixel_adjust_limit=5,
                        do_subpixel_adjust=True, iterstep=0.01, adjust_th=0.1, maxnumiter=10, numpoints=1999, maximum_size=4, flux_unit='Jy/beam'):
    """
    fit the gaussian model for a single source

    args
    -----
    positions: 2D array
        the pixel coordinates of the source in the original image, not cutout
    data: 2D array
        the image data
    wcsNB: WCS
        the wcs of the image
    beam: radio_beam.Beam
        the beam of the image
    pixel_scale: astropy.unit
        the pixel scale of the image
    fitting_size: float
        the maximum radius of the cutout in units of the major axis of the beam
    background: float
        the background of the image
    plot: bool
        plot is True if you want to plot the fitting results
    do_subpixel_adjust: bool
        do_subpixel_adjust is True if you want to do subpixel adjustment
    iterstep: float
        the step size of the subpixel adjustment
    adjust_th: float
        the threshold of the subpixel adjustment
    maxnumiter: int
        the maximum number of iterations for the subpixel adjustment
    numpoints: int
        the number of points for the 1D profile, so the length of the 1D profile is iterstep*numpoints

    return
    -------
    cutout: Cutout2D
        the cutout of the source
    results: lmfit.MinimizerResult
        the fitting results
    xcen: float
        the x position of the center of the source (in the original image)
    ycen: float
        the y position of the center of the source (in the original image)
    adjusted_peakval_min: float


    """
    beam_pa = beam.pa
    numpix_major = beam.major.value/pixel_scale.value
    numpix_minor = beam.minor.value/pixel_scale.value
    #print('fitting array size, ', 1+int(fitting_size*numpix_major), 1+int(fitting_size*numpix_major))
    cutout = Cutout2D(data, positions, (1+int(fitting_size*numpix_major), 1+int(fitting_size*numpix_major)), wcs=wcsNB, mode='partial')
    #print('positions',positions,1+int(fitting_size*numpix_major))
    cutout_data = cutout.data
    if background is not None:
        cutout_data = cutout_data - background
    if do_subpixel_adjust:
        xcen_subpixel = positions[0] - cutout.xmin_original
        ycen_subpixel = positions[1] - cutout.ymin_original
        distarr, profile1d_maj, profile1d_min = get_profile1d(cutout_data,xcen_subpixel,ycen_subpixel, subpixel_adjust_angle, numpoints=numpoints, distarr_step=iterstep)

        numpix_adjust =int(numpix_major/iterstep)
        #print('numpix_adjust',numpix_adjust)
        offset_major = 2*adjust_th
        offset_minor= 2*adjust_th 
        niter=0
        vec_init = (0,0)
        vec = vec_init
        while((np.abs(offset_major)>adjust_th or np.abs(offset_minor)>adjust_th) and niter < maxnumiter): 
            # run iterative process to find the exact center until the offset made each step is lower than certain threshold
            adjusted_peakpos_x, adjusted_peakpos_y, offset_major, adjusted_peakval_maj = subpixel_adjustment(profile1d_maj, distarr,subpixel_adjust_angle, vec, numpix_adjust=numpix_adjust)
           
            xcen_subpixel = xcen_subpixel + adjusted_peakpos_x
            ycen_subpixel = ycen_subpixel + adjusted_peakpos_y
            #print(xcen_subpixel, ycen_subpixel)

            distarr, profile1d_maj, profile1d_min = get_profile1d(cutout_data,xcen_subpixel, ycen_subpixel, subpixel_adjust_angle,numpoints=numpoints, distarr_step=iterstep)
            vec = (vec[0]+adjusted_peakpos_x, vec[1]+adjusted_peakpos_y)  
            

            adjusted_peakpos_x, adjusted_peakpos_y, offset_minor, adjusted_peakval_min = subpixel_adjustment(profile1d_min, distarr,subpixel_adjust_angle+90*u.deg, vec, numpix_adjust=numpix_adjust)
            xcen_subpixel = xcen_subpixel + adjusted_peakpos_x
            ycen_subpixel = ycen_subpixel + adjusted_peakpos_y
            #print(xcen_subpixel, ycen_subpixel)

            distarr, profile1d_maj, profile1d_min  = get_profile1d(cutout_data,xcen_subpixel, ycen_subpixel, subpixel_adjust_angle, numpoints=numpoints, distarr_step=iterstep)
            vec = (vec[0]+adjusted_peakpos_x, vec[1]+adjusted_peakpos_y)  

            niter+=1

        #print('total offset in subpixel_offset_adjust, ', vec)
        if isinstance(xcen_subpixel, u.Quantity):
            xcen_subpixel_val = xcen_subpixel.value
            ycen_subpixel_val = ycen_subpixel.value
        else:
            xcen_subpixel_val = xcen_subpixel
            ycen_subpixel_val = ycen_subpixel

    else:
        xcen_subpixel_val = positions[0] - cutout.xmin_original
        ycen_subpixel_val = positions[1] - cutout.ymin_original
        adjusted_peakval_min = cutout_data[int(ycen_subpixel_val), int(xcen_subpixel_val)]

    if do_subpixel_adjust and np.sqrt(vec[0]**2+vec[1]**2)>subpixel_adjust_limit:
        xcen_subpixel_val = positions[0] - cutout.xmin_original
        ycen_subpixel_val = positions[1] - cutout.ymin_original
        adjusted_peakval_min = cutout_data[int(ycen_subpixel_val), int(xcen_subpixel_val)]    
        print('subpixel adjustment takes another peak... back to the original peak')    
    y,x = np.mgrid[:cutout_data.shape[0],:cutout_data.shape[1]]

    #approx_peak = np.nanmax(cutout_data)
   
    sig_to_fwhm = 2*np.sqrt(2*np.log(2))

    init_pa = 180-beam_pa.value
    if init_pa>180:
        init_pa = init_pa-180
    p0 = [xcen_subpixel_val, ycen_subpixel_val, init_pa*np.pi/180, numpix_major/sig_to_fwhm, numpix_minor/sig_to_fwhm, adjusted_peakval_min, 0]
    #print('p0',p0)
    bounds= ((0.99*xcen_subpixel_val, 1.01*xcen_subpixel_val), 
             (0.99*ycen_subpixel_val, 1.01*ycen_subpixel_val), 
             (0, np.pi), 
             (0.8*numpix_major/sig_to_fwhm, numpix_major/sig_to_fwhm*maximum_size), 
             (0.8*numpix_minor/sig_to_fwhm, numpix_minor/sig_to_fwhm*maximum_size), 
             (0.9*adjusted_peakval_min, 1.1*adjusted_peakval_min), 
             (-1e-4, 1e-4))

    #print('bounds',bounds)
    #bounds = Bounds((0.99*xcen_subpixel, 0.99*ycen_subpixel, 0, 0.4*numpix_major, 0.4*numpix_minor, 0.9*approx_peak),
    #           (1.01*xcen_subpixel, 1.01*ycen_subpixel, np.pi, numpix_major*2, numpix_minor*2, approx_peak))
    params = lmfit.Parameters()
    #params.add('x_center', value=p0[0], min=bounds[0][0], max=bounds[0][1])
    #params.add('y_center', value=p0[1], min=bounds[1][0], max=bounds[1][1])
    #params.add('norm', value=p0[5], min=bounds[5][0], max=bounds[5][1])
    params.add('theta', value=p0[2], min=bounds[2][0], max=bounds[2][1])
    params.add('sigma_diff', value=1, min=0, max=8)
    params.add('sigma_y', value=p0[4], min=bounds[4][0], max=bounds[4][1])

    params.add('sigma_x', value=p0[3], min=bounds[3][0], max=bounds[3][1], expr='sigma_diff + sigma_y')
    params.add('shift', value=p0[6], min=bounds[6][0], max=bounds[6][1])
    results = lmfit.minimize(residual, params, 
                             args=(x, y, cutout_data, xcen_subpixel_val, ycen_subpixel_val, adjusted_peakval_min), 
                             method='leastsq')
    if report_fit:
        lmfit.report_fit(results)
    if plot:
        center = PixCoord(xcen_subpixel_val, ycen_subpixel_val)
        reg = CirclePixelRegion(center,10)
        mask = reg.to_mask()
        model =  gaussian2d(x, y, xcen_subpixel_val, ycen_subpixel_val, adjusted_peakval_min,  **results.params)
        offset = cutout_data-model
        masked_offset = mask.multiply(offset)
        masked_image = mask.multiply(cutout_data)
        masked_offset[masked_offset<0] = masked_offset[masked_offset<0]*np.exp(1e4*np.abs(masked_offset[masked_offset<0]))
        extent = (0,masked_image.shape[1],0,masked_image.shape[0])

        fig = plt.figure(figsize=(18,6))
        ax1 = fig.add_axes([0,0,0.33,1])
        imshow1=ax1.imshow(masked_image, origin='lower', cmap=plt.get_cmap('inferno'), norm=colors.PowerNorm(gamma=0.5,
                                            vmin=0,vmax=np.nanmax(masked_image)), extent=mask.bbox.extent)
        ax1.scatter(xcen_subpixel_val, ycen_subpixel_val, marker='o', c='cyan', s=50)
        ellipse = Ellipse([xcen_subpixel_val, ycen_subpixel_val],
                          width=results.params['sigma_x']*2*np.sqrt(2*np.log(2)),
                          height=results.params['sigma_y']*2*np.sqrt(2*np.log(2)),facecolor='none',
                          angle=results.params['theta']*180/np.pi,edgecolor='cyan',lw=2)
        ax1.add_patch(ellipse)

        ax2 = fig.add_axes([0.33,0,0.33,1])
        imshow2 = ax2.imshow(mask.multiply(model), origin='lower', cmap=plt.get_cmap('inferno'), norm=colors.PowerNorm(gamma=0.5,
                                            vmin=0,vmax=np.nanmax(masked_image)))
        ax3 = fig.add_axes([0.66,0,0.33,1])
        imshow3 = ax3.imshow(masked_offset, origin='lower', cmap=plt.get_cmap('inferno'), norm=colors.PowerNorm(gamma=0.5,
                                            vmin=0,vmax=np.nanmax(masked_offset)))
        
        axins1 = inset_axes(ax1, width="80%", height="7%", loc='upper center')
        axins1.xaxis.set_ticks_position("bottom")
        cbar = plt.colorbar(imshow1, cax = axins1, orientation='horizontal')
        cbar.set_label('flux (%s))'%flux_unit, color='w',fontsize=25)
        cbar.ax.xaxis.set_tick_params(color='w')
        cbar.outline.set_edgecolor('w')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')

        
        axins2 = inset_axes(ax2, width="80%", height="7%", loc='upper center')
        axins2.xaxis.set_ticks_position("bottom")
        cbar=plt.colorbar(imshow2, cax = axins2, orientation='horizontal')
        cbar.set_label('flux (%s))'%flux_unit,color='w')
        cbar.ax.xaxis.set_tick_params(color='w')
        cbar.outline.set_edgecolor('w')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')


        axins3 = inset_axes(ax3, width="80%", height="7%", loc='upper center')
        axins3.xaxis.set_ticks_position("bottom")
        cbar=plt.colorbar(imshow3, cax = axins3, orientation='horizontal')
        cbar.set_label('flux (%s)'%flux_unit,color='w')
        cbar.ax.xaxis.set_tick_params(color='w')
        cbar.outline.set_edgecolor('w')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')
        plt.show()
        plt.close()

    return results, xcen_subpixel_val + cutout.xmin_original, ycen_subpixel_val + cutout.ymin_original, adjusted_peakval_min

def add_beam(ax,xpos,ypos,beam, pixel_scale,color='w',square=False,square_size=800):
    width = beam.major / pixel_scale
    height = beam.minor /pixel_scale
    angle = beam.pa
    
    ax.add_patch(Ellipse((xpos,ypos),width.value,height.value,angle=180-angle.value,color=color))
    if square:
        ax.scatter(xpos,ypos,facecolor='none', edgecolor=color,s=square_size,marker='s')

def get_profile1d(cutout_data, xcen, ycen, inclination, numpoints=199, distarr_step=0.1):
    
    # get two end points of a line along major-axis and minor axis

    xoffset_major = np.cos(inclination) * (numpoints-1)/2*distarr_step
    yoffset_major = np.sin(inclination) * (numpoints-1)/2*distarr_step
    xoffset_minor = np.cos(inclination+90*u.deg) * (numpoints-1)/2*distarr_step
    yoffset_minor = np.sin(inclination+90*u.deg) * (numpoints-1)/2*distarr_step
     
    
    x0_maj = xcen-xoffset_major
    x1_maj = xcen+xoffset_major
    y0_maj = ycen-yoffset_major
    y1_maj = ycen+yoffset_major
    
    x0_min = xcen-xoffset_minor
    x1_min = xcen+xoffset_minor
    y0_min = ycen-yoffset_minor
    y1_min = ycen+yoffset_minor

    # extract 1d profile along major, minor axis
    xarr_maj = np.linspace(x0_maj, x1_maj, numpoints)
    yarr_maj = np.linspace(y0_maj, y1_maj, numpoints)
    xarr_min = np.linspace(x0_min, x1_min, numpoints)
    yarr_min = np.linspace(y0_min, y1_min, numpoints)
 
    distarr = np.linspace(-(numpoints-1)/2*distarr_step, (numpoints-1)/2*distarr_step, numpoints)
   
    profile1d_maj = scipy.ndimage.map_coordinates(cutout_data, np.vstack((yarr_maj,xarr_maj)), mode='constant',cval=-10,prefilter=False)
    profile1d_min = scipy.ndimage.map_coordinates(cutout_data, np.vstack((yarr_min,xarr_min)), mode='constant',cval=-10,prefilter=False)

   
    return distarr, profile1d_maj, profile1d_min

def plot_for_individual(data,  xcen, ycen, xcen_original, ycen_original, pa, major, minor, peak, pixel_scale, background, 
                         major_err, minor_err, 
                         beam, wcsNB, plot_size=4,
                        idx=0, issqrt=True, iterstep=0.01,
                         vmin=-0.00010907209521789237, vmax=0.002236069086983825, flux_unit='Jy/beam',  
                        bkg_inner_width=3, bkg_annulus_width=1, bkg_inner_height=3, bkg_annulus_height=1,
                          savedir='./',label='w51e', show=True):
    
    plt.rcParams['axes.labelsize']=25
    plt.rcParams['xtick.labelsize']=25
    plt.rcParams['ytick.labelsize']=25
    plt.rcParams['axes.titlesize']=25
    mpl.rcParams['axes.linewidth'] = 5
    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['xtick.major.width'] = 4
    mpl.rcParams['xtick.minor.size'] = 5
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['ytick.major.width'] = 4
    mpl.rcParams['ytick.minor.size'] = 5
    mpl.rcParams['ytick.minor.width'] = 2
    params = {"xtick.top": True, "ytick.right": True, "xtick.direction": "in", "ytick.direction": "in"}
    plt.rcParams.update(params)

    numpix_major = beam.major.value/pixel_scale.value
    cutout = Cutout2D(data, (xcen, ycen), (1+plot_size*int(numpix_major), 1+plot_size*int(numpix_major)), wcs=wcsNB)
    cutout_data = cutout.data             
    
    numpix_major = int(beam.major.value/pixel_scale.value)
    numpoints = int(2*numpix_major / iterstep)
    xcen_cutout = xcen
    ycen_cutout = ycen

    fig = plt.figure(figsize=(32,24))
    ax1 = fig.add_axes([0.08, 0.55, 0.27,0.4])
    ax2 = fig.add_axes([0.40, 0.55, 0.27,0.4])
    ax3 = fig.add_axes([0.72, 0.55, 0.27,0.4])
    ax4 = fig.add_axes([0.08, 0.08, 0.27,0.4])
    ax5 = fig.add_axes([0.40, 0.08, 0.27,0.4])
    ax6 = fig.add_axes([0.72, 0.08, 0.27,0.4])

    distarr, profile1d_maj, profile1d_min = get_profile1d(cutout_data, xcen_cutout - cutout.xmin_original, ycen_cutout - cutout.ymin_original, pa*u.deg, numpoints=numpoints, distarr_step=iterstep)
    profile1d_maj_bkgsub = profile1d_maj - background
    profile1d_min_bkgsub = profile1d_min - background

    cutout_bkgsub = cutout_data - background

    if vmin is None or vmax is None:
        cutout_small = Cutout2D(data, (xcen, ycen), (5*int(numpix_major), 5*int(numpix_major)), wcs=wcsNB)
        vmin = np.nanmin(cutout_small.data - background)
        vmax = 0.7*np.nanmax(cutout_small.data - background)

    ax1.plot(distarr, profile1d_maj, label='image',ls='solid',lw=4, c='k')
    ax1.plot(distarr, profile1d_maj_bkgsub, label='image bkg-subtracted',lw=4)
    
    ax1.plot(distarr, gaussian(distarr, 0, major, peak), ls='dashed', label='model',c='orange',lw=4)
    
    if major_err is not None:
        ax1.fill_between(distarr, gaussian(distarr, 0, major-major_err, peak), 
                                gaussian(distarr, 0, major+major_err, peak),
                            alpha=0.3,color='orange')
    
    ax2.plot(distarr, profile1d_min, label='image',ls='solid',lw=4, c='k')
    ax2.plot(distarr, profile1d_min_bkgsub, label='image bkg-subtracted',lw=4)
    
    ax2.plot(distarr, gaussian(distarr, 0, minor, peak), ls='dashed', label='model',c='orange',lw=4)
    if minor_err is not None:

        ax2.fill_between(distarr, gaussian(distarr, 0, minor-minor_err, peak), 
                                gaussian(distarr, 0, minor+minor_err, peak),
                            alpha=0.3,color='orange')
    ax1.set_title('along major axis',fontsize=35)
    ax2.set_title('along minor axis',fontsize=35)
    ax3.set_title('residual',fontsize=35)
    res_maj = profile1d_maj_bkgsub - gaussian(distarr, 0, major, peak)
    res_min = profile1d_min_bkgsub - gaussian(distarr, 0, minor, peak)
    ax3.plot(distarr, res_maj, label='major axis',lw=4, c='purple')
    ax3.plot(distarr, res_min, label='minor axis',lw=4, c='cyan')


    
    ax1.legend(fontsize=25)
    ax3.legend(fontsize=25)
    ax1.set_xlabel('pixel offset',fontsize=35)
    ax2.set_xlabel('pixel offset',fontsize=35)
    ax3.set_xlabel('pixel offset',fontsize=35)
    ax1.set_ylabel('flux (%s)'%flux_unit,fontsize=35)
    
    ax1.set_xlim(-(numpoints-1)/2*iterstep, (numpoints-1)/2*iterstep)
    ax2.set_xlim(-(numpoints-1)/2*iterstep, (numpoints-1)/2*iterstep)
    ax3.set_xlim(-(numpoints-1)/2*iterstep, (numpoints-1)/2*iterstep)
    
    ax1.set_ylim(np.min((np.nanmin(profile1d_maj), np.nanmin(gaussian(distarr, 0, minor, peak)))),
     np.max((1.1*np.nanmax(profile1d_maj), 1.1*np.nanmax(gaussian(distarr, 0, minor, peak)))))
    ax2.set_ylim(np.min((np.nanmin(profile1d_maj), np.nanmin(gaussian(distarr, 0, minor, peak)))),
     np.max((1.1*np.nanmax(profile1d_maj), 1.1*np.nanmax(gaussian(distarr, 0, minor, peak)))))
    ax3.set_ylim(1.1*np.nanmin([np.nanmin(res_maj),np.nanmin(res_min)]), 1.1*np.nanmax([np.nanmax(res_maj),np.nanmax(res_min)]))
    ax1.text(-(numpoints-1)/2*iterstep*0.9, np.nanmax(profile1d_maj), '#%d'%idx, fontsize=30)
    major_fwhm = major * 2*np.sqrt(2*np.log(2))
    minor_fwhm = minor * 2*np.sqrt(2*np.log(2))
    extent = (-int(cutout_data.shape[1]/2),int(cutout_data.shape[1]/2),-int(cutout_data.shape[0]/2),int(cutout_data.shape[0]/2))
    extent = (cutout.xmin_original-0.5, cutout.xmax_original+0.5, cutout.ymin_original-0.5, cutout.ymax_original+0.5)
    if issqrt:
        imshow1 = ax4.imshow(cutout_bkgsub, origin='lower', cmap=plt.get_cmap('inferno'), 
                    norm=colors.PowerNorm(gamma=0.5,
                                            vmin=vmin,vmax=vmax), extent=extent)
    else:
        imshow1 = ax4.imshow(cutout_bkgsub, origin='lower', cmap=plt.get_cmap('inferno'), 
                    vmin=vmin,vmax=vmax, extent=extent)

    ellipse = Ellipse([xcen_cutout, ycen_cutout],
                          width=major_fwhm,height=minor_fwhm,facecolor='none',
                          angle=pa,edgecolor='cyan',lw=2)
    inner_annulus = Ellipse([xcen_cutout, ycen_cutout],
                            width=bkg_inner_width*major_fwhm,height=bkg_inner_width*minor_fwhm,facecolor='none',
                        angle=pa,edgecolor='cyan',lw=2, ls='dotted')
    outer_annulus = Ellipse([xcen_cutout, ycen_cutout],
                            width=(bkg_inner_width+bkg_annulus_width)*major_fwhm, 
                            height=(bkg_inner_width+bkg_annulus_width)*minor_fwhm,facecolor='none',
                        angle=pa,edgecolor='cyan',lw=2, ls='dotted')
    ax4.add_patch(ellipse)
    ax4.add_patch(inner_annulus)
    ax4.add_patch(outer_annulus)
    
    ax4.scatter(xcen_cutout, ycen_cutout, marker='x', c='cyan')
    ax4.scatter(xcen_original, 
                ycen_original, marker='x', c='r')

    add_beam(ax4,xcen_cutout-0.4*(plot_size*numpix_major+1),ycen_cutout-0.4*(plot_size*numpix_major+1), 
                beam, pixel_scale, square=True, square_size=6000)

    y,x = np.mgrid[:cutout_data.shape[0], :cutout_data.shape[1]]
           

    model =  gaussian2d(x+cutout.xmin_original, y+cutout.ymin_original, x_center=xcen_cutout, y_center=ycen_cutout
                        , theta=pa*np.pi/180, sigma_x = major, sigma_y= minor,norm=peak)
    
 
    
    if issqrt:
        imshow2 = ax5.imshow(model, origin='lower', cmap=plt.get_cmap('inferno'), 
                    norm=colors.PowerNorm(gamma=0.5,
                                            vmin=vmin,vmax=vmax), extent=extent)
    else:
        imshow2 = ax5.imshow(model, origin='lower', cmap=plt.get_cmap('inferno'), 
                    vmin=vmin,vmax=vmax, extent=extent)
        
    ellipse = Ellipse([xcen_cutout, ycen_cutout],
                          width=major_fwhm,height=minor_fwhm,facecolor='none',
                          angle=pa,edgecolor='cyan',lw=2)
    inner_annulus = Ellipse([xcen_cutout, ycen_cutout],
                            width=bkg_inner_width*major_fwhm,height=bkg_inner_height*minor_fwhm,facecolor='none',
                        angle=pa,edgecolor='cyan',lw=2, ls='dotted')
    outer_annulus = Ellipse([xcen_cutout, ycen_cutout],
                            width=(bkg_inner_width+bkg_annulus_width)*major_fwhm, 
                            height=(bkg_inner_height+bkg_annulus_height)*minor_fwhm,facecolor='none',
                        angle=pa,edgecolor='cyan',lw=2, ls='dotted')
    ax5.add_patch(ellipse)
    ax5.add_patch(inner_annulus)
    ax5.add_patch(outer_annulus)
    #if resvmin is None or resvmax is None:
    #    resvmin = np.nanmin(cutout_small.data - background)
    #    resvmax = 0.7*np.nanmax(cutout_small.data - background)
    if issqrt:
        imshow3 = ax6.imshow(cutout_bkgsub - model, origin='lower', cmap=plt.get_cmap('inferno'), 
                    norm=colors.PowerNorm(gamma=0.5,
                                            vmin=vmin,vmax=vmax), extent=extent)
    else:
        imshow3 = ax6.imshow(cutout_bkgsub - model, origin='lower', cmap=plt.get_cmap('inferno'), 
                             
                    vmin=vmin,vmax=vmax, extent=extent)
    ellipse = Ellipse([xcen_cutout, ycen_cutout],
                          width=major_fwhm,height=minor_fwhm,facecolor='none',
                          angle=pa,edgecolor='cyan',lw=2)
    inner_annulus = Ellipse([xcen_cutout, ycen_cutout],
                            width=bkg_inner_width*major_fwhm,height=bkg_inner_height*minor_fwhm,facecolor='none',
                        angle=pa,edgecolor='cyan',lw=2, ls='dotted')
    outer_annulus = Ellipse([xcen_cutout, ycen_cutout],
                            width=(bkg_inner_width+bkg_annulus_width)*major_fwhm, 
                            height=(bkg_inner_height+bkg_annulus_height)*minor_fwhm,facecolor='none',
                        angle=pa,edgecolor='cyan',lw=2, ls='dotted')
    ax6.add_patch(ellipse)
    ax6.add_patch(inner_annulus)
    ax6.add_patch(outer_annulus)
    
    
    ax4.set_title('image',fontsize=35)
    ax5.set_title('model',fontsize=35)
    ax6.set_title('residual',fontsize=35)
   
    ax4.set_xlim(cutout.xmin_original, cutout.xmax_original)
    ax4.set_ylim(cutout.ymin_original, cutout.ymax_original)
    ax5.set_xlim(cutout.xmin_original, cutout.xmax_original)
    ax5.set_ylim(cutout.ymin_original, cutout.ymax_original)
    ax6.set_xlim(cutout.xmin_original, cutout.xmax_original)
    ax6.set_ylim(cutout.ymin_original, cutout.ymax_original)

    ax5.set_xlabel('RA pixel coordinates',fontsize=35)
    ax4.set_ylabel('DEC pixel coordinates',fontsize=35)

    axins1 = inset_axes(ax4, width="80%", height="7%", loc='upper center')
    axins1.xaxis.set_ticks_position("bottom")
    cbar = plt.colorbar(imshow1, cax = axins1, orientation='horizontal')
    cbar.set_label('flux (%s)'%flux_unit, color='w',fontsize=25)
    cbar.ax.xaxis.set_tick_params(color='w')
    cbar.outline.set_edgecolor('w')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')

    
    axins2 = inset_axes(ax5, width="80%", height="7%", loc='upper center')
    axins2.xaxis.set_ticks_position("bottom")
    cbar=plt.colorbar(imshow2, cax = axins2, orientation='horizontal')
    cbar.set_label('flux (%s)'%flux_unit,color='w')
    cbar.ax.xaxis.set_tick_params(color='w')
    cbar.outline.set_edgecolor('w')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')


    axins3 = inset_axes(ax6, width="80%", height="7%", loc='upper center')
    axins3.xaxis.set_ticks_position("bottom")
    cbar=plt.colorbar(imshow3, cax = axins3, orientation='horizontal')
    cbar.set_label('flux (%s)'%flux_unit,color='w')
    cbar.ax.xaxis.set_tick_params(color='w')
    cbar.outline.set_edgecolor('w')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')
    fig.patch.set_facecolor('white')
    plt.savefig(savedir+label+'_%06d.png'%idx)
    if show:
        plt.show()
    plt.close()

def get_integrated_flux(norm, sigma_x, sigma_y, sigma_x_err, sigma_y_err,beam, pixel_scale, flux_unit='Jy/beam'):
    if flux_unit == 'Jy/beam':
        flux_in_pix = norm / (np.pi * beam.major/2 * beam.minor/2) * (pixel_scale)**2 * u.Jy
    elif flux_unit == 'mJy/beam':
        flux_in_pix = norm / (np.pi * beam.major/2 * beam.minor/2) * (pixel_scale)**2 * u.mJy # mJy/beam -> mJy/pix**2
    
    flux = 2*np.pi*flux_in_pix*sigma_x*sigma_y
    if sigma_x_err is not None:
        fluxerr = flux * ((np.array(sigma_x_err)/np.array(sigma_x))**2 + (np.array(sigma_y_err)/np.array(sigma_y))**2)
        return flux.to(u.Jy), fluxerr.to(u.Jy) 

    else:
        fluxerr = np.nan
        return flux.to(u.Jy), fluxerr

    #fluxerr = flux * np.sqrt((norm_err/norm)**2 + (sigma_x_err/sigma_x)**2 + (sigma_y_err/sigma_y)**2)
    


       
def save_fitting_results( fitted_major, fitted_minor, major_err, minor_err, pa, pa_err,
                         flux, flux_err, deconvolved_major_arr, deconvolved_minor_arr,
                         peak_arr,
                         savedir='./', label='w51e',):
    
    """
    saves the fitting results in the fits format

    args
    ----
    fitted_major: fitted major axis FWHM. The unit is equal to the pixel_scale unit.
    fitted_minor: fitted minor axis FWHM. The unit is equal to the pixel_scale unit.
    major_err: error of the fitted major axis FWHM.
    minor_err: error of the fitted minor axis FWHM. 
    pa: fitted position angle in degree.
    pa_err: error of the fitted position angle in degree.
    flux: flux of the source from the 2D gaussian fitting. The unit is the same as the flux unit of the image.
    flux_err: error of the flux of the source from the 2D gaussian fitting.
    deconvolved_major_arr: deconvolved major axis FWHM. The unit is equal to the pixel_scale unit.
    deconvolved_minor_arr: deconvolved minor axis FWHM. The unit is equal to the pixel_scale unit.
    peak_arr: peak flux of the source from the 2D gaussian fitting. 
    savedir: directory to save the fitting result table in the fits format (default: './')
    label: label of the fits file. The directory of the fits file will be savedir+'%s.fits'%label. (default: 'w51e')

    return
    ------
    tab: astropy table of the fitting results

    """
    
    from astropy.table import QTable

    
    tab = QTable([flux, flux_err, 
                 pa, pa_err, 
                 fitted_major, major_err,
                 fitted_minor, minor_err,
                deconvolved_major_arr, deconvolved_minor_arr, peak_arr],
                names=('flux', 'flux_err',
                        'pa', 'pa_err',
                       'fitted_major', 'fitted_major_err',
                       'fitted_minor', 'fitted_minor_err', 
                       'deconvolved_major', 'deconvolved_minor', 
                       'peak_flux'
                       ))

    tab.pprint_all()

    if savedir is not None:
        tab.write(savedir+'%s.fits'%label, format='fits', overwrite=True)

    return tab

def redefine_center(img, positions, searching_rad=4):
    """
    find the peak of the image if the original source position is not at the peak.
    """

    cutout = Cutout2D(img, positions, (searching_rad, searching_rad))
    ycen, xcen = np.unravel_index(np.nanargmax(cutout.data), cutout.data.shape)
    return xcen+cutout.xmin_original, ycen+cutout.ymin_original

    

def plot_and_save_fitting_results(data, peakxy, beam, wcsNB, pixel_scale,
                         fitting_size_default = 4,
                        issqrt=True, vmin=None, vmax=None,
                        flux_unit='Jy/beam', do_subpixel_adjust=True,
                        bkg_inner_width=4, bkg_annulus_width=2, bkg_inner_height=4, bkg_annulus_height=2, maximum_size=4,
                        saveimgdir=None, savefitsdir=None,  make_plot=True, show=True, label_img=None, label_fits='w51e_b6_test',
                        fix_pos_idx=[],fitting_size_dict={}, idx=0, subpixel_adjust_limit=4):
    """

    main function to plot and save the fitting results


    args
    ----
    data: 2d array of the image
    peakxy: list of the pixel coordinates of the sources
    beam: Beam object of the image beam
    wcsNB: WCS object of the image
    pixel_scale: pixel scale of the image
    fitting_size_default: default size of the fitting box
    issqrt: if True, the image is shown in sqrt scale (default: True)
    vmin: minimum value for the color scale of the image (default: None)
    vmax: maximum value for the color scale of the image (default: None)
    flux_unit: unit of the flux (default: 'Jy/beam')
    do_subpixel_adjust: if True, the subpixel adjustment is done (default: True)
    bkg_inner_width: inner width of the background annulus in the unit of pixel (default: 4)
    bkg_annulus_width: outer width of the background annulus in the unit of pixel (default: 2)
    bkg_inner_height: inner height of the background annulus in the unit of pixel (default: 4)
    bkg_annulus_height: outer height of the background annulus in the unit of pixel (default: 2)
    maximum_size: maximum size of the fitting cutout in the unit of pixel (default: 4)
    saveimgdir: directory to save the fitting result image (default: None)
    savefitsdir: directory to save the fitting result table in the fits format (default: None)
    make_plot: if True, the fitting results are plotted (default: True)
    show: if True, the fitting results are shown (default: True)
    label_img: label of the image file name. The directory of the image will be saveimgdir+label_img+'_%06d.png'%idx. (default: None)
    label_fits: label of the fits file. The directory of the fits file will be savefitsdir+'%s.fits'%label_fits. (default: 'w51e_b6_test')
    fix_pos_idx: list of the index of the source to fix the position. Useful when no significant peak is found at the original position of the source. (default: [])
    fitting_size_dict: dictionary of the fitting size for each source. The key is the index of the source and the value is the size of the fitting box. (default: {})
    idx: index of the source (default: 0)
    subpixel_adjust_limit: the maximum limit of the subpixel adjustment in units of pixel.  (default: 4)

    return
    ------
    tab: astropy table of the fitting results


    """
    sig_to_fwhm = 2*np.sqrt(2*np.log(2))

    if isinstance(peakxy, list) and len(peakxy)==2: #when the coordinates of a single source are given
        positions = redefine_center(data, peakxy)
        results, xcen_fit_init, ycen_fit_init, peak_fit_init = fit_for_individuals(positions, data, wcsNB, beam, pixel_scale, 
                                                                                                    subpixel_adjust_angle=180*u.deg-beam.pa, plot=False, 
                                                                                                    fitting_size=fitting_size_default, maximum_size=maximum_size,report_fit=False, do_subpixel_adjust=do_subpixel_adjust,
                                                                                                    subpixel_adjust_limit=subpixel_adjust_limit)
        popt = results.params
        xcen_init = xcen_fit_init
        ycen_init = ycen_fit_init
        pa_init = popt['theta'] * 180 / np.pi
        fitted_major_init = popt['sigma_x']
        fitted_minor_init = popt['sigma_y']
        bkg, bkg_mad = get_local_bkg(data, xcen_init, ycen_init, pa_init, None, wcsNB, beam, pixel_scale,
                                    inner_width=bkg_inner_width*fitted_major_init, outer_width=(bkg_inner_width+bkg_annulus_width)*fitted_major_init, 
                                    inner_height=bkg_inner_height*fitted_minor_init, outer_height=(bkg_inner_height+bkg_annulus_height)*fitted_minor_init)
        results, xcen_fit, ycen_fit, peak_fit  = fit_for_individuals(positions, data, wcsNB, beam, pixel_scale, 
                                                                                subpixel_adjust_angle=pa_init*u.deg,background = bkg, plot=False, fitting_size=fitting_size_default, flux_unit=flux_unit, maximum_size=maximum_size, report_fit=False, do_subpixel_adjust=do_subpixel_adjust,
                                                                                subpixel_adjust_limit=subpixel_adjust_limit)
        popt = results.params

        xcen = xcen_fit
        ycen = ycen_fit
        pa = popt['theta'] * 180 / np.pi
        fitted_major = popt['sigma_x']
        fitted_minor = popt['sigma_y']
        peak = peak_fit
       #print('xcen, ycen, pa, fitted_major, fitted_minor, peak',xcen, ycen, pa, fitted_major, fitted_minor, peak)
        if results.params['theta'].stderr is not None:
            pa_err = results.params['theta'].stderr * 180 / np.pi
        else:
            pa_err = np.nan
        fitted_major_err = results.params['sigma_x'].stderr
        fitted_minor_err = results.params['sigma_y'].stderr

        if make_plot:
            plot_for_individual(data, xcen, ycen, peakxy[0], peakxy[1], pa, fitted_major, fitted_minor, peak, pixel_scale, bkg, fitted_major_err, fitted_minor_err,  
                                    beam, wcsNB,
                                    idx=idx, issqrt=issqrt,
                                    vmin=vmin, vmax=vmax,  
                                    plot_size=10, flux_unit = flux_unit,
                                    bkg_inner_width=bkg_inner_width, bkg_annulus_width=bkg_annulus_width,
                                    bkg_inner_height=bkg_inner_height, bkg_annulus_height=bkg_annulus_height,
                                    savedir=saveimgdir,label=label_img, show=show)
        if fitted_major_err is not None:
            flux, flux_err = get_integrated_flux(peak, fitted_major.value, fitted_minor.value, fitted_major_err, fitted_minor_err, beam, pixel_scale, flux_unit=flux_unit)
        else:
            flux, flux_err = np.nan, np.nan
        
        major_fwhm = np.array(fitted_major.value) * 2*np.sqrt(2*np.log(2))
        minor_fwhm = np.array(fitted_minor.value) * 2*np.sqrt(2*np.log(2))

        fwhm_major_sky = major_fwhm * pixel_scale 
        fwhm_minor_sky = minor_fwhm * pixel_scale 

        fitted_gaussian_as_beam = Beam(major=fwhm_major_sky, minor=fwhm_minor_sky, pa=pa)
    
        try:
            deconvolved = fitted_gaussian_as_beam.deconvolve(beam)
            deconvolved_major = deconvolved.major.value
            deconvolved_minor = deconvolved.minor.value
        except:
            deconvolved_major =0
            deconvolved_minor =0   
        if isinstance(flux_err, u.Quantity):
            flux_err = flux_err.value

        return flux, flux_err, pa, pa_err, fitted_major, fitted_major_err, fitted_minor, fitted_minor_err, deconvolved_major, deconvolved_minor

    else: #when the coordinates of multiple sources are given

        num_source = len(peakxy[:,0])
        
        fitted_major_arr = []
        fitted_minor_arr = []
        peak_arr = []
        pa_arr = []
        fitted_major_err_arr = []
        fitted_minor_err_arr = []
        peak_err_arr = []
        pa_err_arr = []
        flux_arr = []
        flux_err_arr = []  
        deconvolved_major_arr = []
        deconvolved_minor_arr = []

        for i in range(num_source):
        
        
            if peakxy[i,0]<0 or peakxy[i,1]<0 :
                fitted_major_arr.append(np.nan)
                fitted_minor_arr.append(np.nan)
                peak_arr.append(np.nan)
                fitted_major_err_arr.append(np.nan)
                fitted_minor_err_arr.append(np.nan)
                peak_err_arr.append(np.nan)
                pa_arr.append(np.nan)
                pa_err_arr.append(np.nan)
                flux_arr.append(np.nan)
                flux_err_arr.append(np.nan)
                deconvolved_major_arr.append(np.nan)
                deconvolved_minor_arr.append(np.nan)
                continue
            
            if i in fitting_size_dict:
                fitting_size = fitting_size_dict[i]
            else:
                fitting_size = fitting_size_default
            print('idx, fitting_size', i, fitting_size)
            if i in fix_pos_idx:
                do_subpixel_adjust = False
            else:
                do_subpixel_adjust = True    
            positions_original = (peakxy[i,0], peakxy[i,1])
            positions = redefine_center(data, positions_original)
            results, xcen_fit_init, ycen_fit_init, peak_fit_init = fit_for_individuals(positions, data, wcsNB, beam, pixel_scale, 
                                                                                                    subpixel_adjust_angle=180*u.deg-beam.pa, plot=False, 
                                                                                                    fitting_size=fitting_size, maximum_size=maximum_size,report_fit=False, do_subpixel_adjust=do_subpixel_adjust)
            popt = results.params
            xcen_init = xcen_fit_init
            ycen_init = ycen_fit_init
            pa_init = popt['theta'] * 180 / np.pi
            fitted_major_init = popt['sigma_x']
            fitted_minor_init = popt['sigma_y']
            bkg, bkg_mad = get_local_bkg(data, xcen_init, ycen_init, pa_init, peakxy, wcsNB, beam, pixel_scale,
                                        inner_width=bkg_inner_width*fitted_major_init, outer_width=(bkg_inner_width+bkg_annulus_width)*fitted_major_init, 
                                        inner_height=bkg_inner_height*fitted_major_init, outer_height=(bkg_inner_height+bkg_annulus_height)*fitted_major_init)
            
        
            results, xcen_fit, ycen_fit, peak_fit  = fit_for_individuals(positions, data, wcsNB, beam, pixel_scale, 
                                                                                subpixel_adjust_angle=pa_init*u.deg,background = bkg, plot=False, fitting_size=fitting_size, flux_unit=flux_unit, maximum_size=maximum_size, report_fit=False, do_subpixel_adjust=do_subpixel_adjust)
            popt = results.params

            xcen = xcen_fit
            ycen = ycen_fit
            pa = popt['theta'] * 180 / np.pi
            fitted_major = popt['sigma_x'] * sig_to_fwhm #Gausian width -> FWHM
            fitted_minor = popt['sigma_y'] * sig_to_fwhm
            peak = peak_fit
            if peak is not None:
                peak_arr.append(peak)
            else:
                peak_arr.append(np.nan)
            #xcen_err = results.params['x_center'].stderr
            #ycen_err = results.params['y_center'].stderr
            if results.params['theta'].stderr is not None:
                pa_err = results.params['theta'].stderr * 180 / np.pi
            else:
                pa_err = np.nan
            if results.params['sigma_x'].stderr is not None:
                fitted_major_err = results.params['sigma_x'].stderr * sig_to_fwhm
                fitted_minor_err = results.params['sigma_y'].stderr * sig_to_fwhm
            else:
                fitted_major_err = np.nan
                fitted_minor_err = np.nan
            if fitted_major is not None:
                fitted_major_arr.append(fitted_major*pixel_scale.value)
            else:
                fitted_major_arr.append(np.nan)
            if fitted_minor is not None:
                fitted_minor_arr.append(fitted_minor*pixel_scale.value)
            else:
                fitted_minor_arr.append(np.nan)
            if fitted_major_err is not None:
                fitted_major_err_arr.append(fitted_major_err*pixel_scale.value)
            else:
                fitted_major_err_arr.append(np.nan)
            if fitted_minor_err is not None:
                fitted_minor_err_arr.append(fitted_minor_err*pixel_scale.value)
            else:
                fitted_minor_err_arr.append(np.nan)
                
            if make_plot:
                plot_for_individual(data, xcen, ycen, peakxy[i,0], peakxy[i,1], pa, fitted_major/sig_to_fwhm, fitted_minor/sig_to_fwhm, peak, pixel_scale, bkg, fitted_major_err/sig_to_fwhm, fitted_minor_err/sig_to_fwhm,  
                                    beam, wcsNB,
                                    idx=i, issqrt=issqrt,
                                    vmin=vmin, vmax=vmax,  
                                    plot_size=10, flux_unit = flux_unit,
                                    bkg_inner_width=bkg_inner_width, bkg_annulus_width=bkg_annulus_width,
                                    bkg_inner_height=bkg_inner_height, bkg_annulus_height=bkg_annulus_height,
                                    savedir=saveimgdir,label=label_img, show=show)
            
            flux, flux_err = get_integrated_flux(peak, fitted_major / sig_to_fwhm, fitted_minor / sig_to_fwhm, fitted_major_err /sig_to_fwhm, fitted_minor_err /sig_to_fwhm, beam, pixel_scale, flux_unit=flux_unit)
            
            major_fwhm = np.array(fitted_major) 
            minor_fwhm = np.array(fitted_minor)

            fwhm_major_sky = major_fwhm * pixel_scale 
            fwhm_minor_sky = minor_fwhm * pixel_scale 

            fitted_gaussian_as_beam = Beam(major=fwhm_major_sky, minor=fwhm_minor_sky, pa=pa)
        
            try:
                deconvolved = fitted_gaussian_as_beam.deconvolve(beam)
                deconvolved_major = deconvolved.major.value
                deconvolved_minor = deconvolved.minor.value
            except:
                deconvolved_major =0
                deconvolved_minor =0

            deconvolved_major_arr.append(deconvolved_major)
            deconvolved_minor_arr.append(deconvolved_minor)
        

            pa_arr.append(pa)
            pa_err_arr.append(pa_err)
            if isinstance(flux, u.Quantity):
                flux_arr.append(flux.value)
            else:
                flux_arr.append(flux)

            if isinstance(flux_err, u.Quantity):
                flux_err_arr.append(flux_err.value)
            else:
                flux_err_arr.append(flux_err)

          

            #print('i, xcen, ycen, pa, fitted_major, fitted_minor, peak,  pa_err, fitted_major_err, fitted_minor_err,',
            #    i, xcen, ycen, pa, fitted_major.value, fitted_minor.value, peak,  pa_err, fitted_major_err, fitted_minor_err )
        if pixel_scale.unit == u.arcsec:
            fwhm_unit = u.arcsec
        elif pixel_scale.unit == u.deg:
            fwhm_unit = u.deg

        fitted_major_column = MaskedColumn(data=fitted_major_arr, name='fitted_major', mask=np.isnan(fitted_major_arr), unit=fwhm_unit, fill_value=-999)
        fitted_minor_column = MaskedColumn(data=fitted_minor_arr, name='fitted_minor', mask=np.isnan(fitted_minor_arr), unit=fwhm_unit, fill_value=-999)
        fitted_major_err_column = MaskedColumn(data=fitted_major_err_arr, name='fitted_major_err', mask=np.isnan(fitted_major_err_arr), unit=fwhm_unit, fill_value=-999)  
        fitted_minor_err_column = MaskedColumn(data=fitted_minor_err_arr, name='fitted_minor_err', mask=np.isnan(fitted_minor_err_arr), unit=fwhm_unit, fill_value=-999)  
        pa_column = MaskedColumn(data=pa_arr, name='pa', mask=np.isnan(pa_arr), unit=u.deg, fill_value=-999) 
        pa_err_column = MaskedColumn(data=pa_err_arr, name='pa_err', mask=pd.isnull(pa_err_arr), unit=u.deg, fill_value=-999) 
        flux_column = MaskedColumn(data=flux_arr, name='flux', mask=np.isnan(flux_arr), unit=u.Jy, fill_value=-999)
        flux_err_column = MaskedColumn(data=flux_err_arr, name='flux_err', mask=np.isnan(flux_err_arr), unit=u.Jy, fill_value=-999)
        deconvolved_major_column = MaskedColumn(data=deconvolved_major_arr, name='deconvolved_major', mask=np.isnan(deconvolved_major_arr), unit=fwhm_unit, fill_value=-999)
        deconvolved_minor_column = MaskedColumn(data=deconvolved_minor_arr, name='deconvolved_minor', mask=np.isnan(deconvolved_minor_arr), unit=fwhm_unit, fill_value=-999)
        peak_column = MaskedColumn(data=peak_arr, name='peak', mask=np.isnan(peak_arr), unit=flux_unit, fill_value=-999)

        tab = save_fitting_results(fitted_major_column, fitted_minor_column, fitted_major_err_column, fitted_minor_err_column, pa_column, pa_err_column, 
                                flux_column, flux_err_column, deconvolved_major_column, deconvolved_minor_column, peak_column, savedir=savefitsdir, label=label_fits)
                      
        return tab








