import numpy as np
import matplotlib.pyplot as plt

def find_velrange_centvel(vels, v_helio, profile, mean_thresh = 0.7, diag = False):
    '''this function finds the flux-weighted central velocity and the region of integration for the curve of growth.'''
    #arguments: 
    #    vels: velocities corresponding to the channels of the profile
    #    v_helio: preliminary central velocity guess
    #    profile: profile flux values
    #    mean_thresh: the fraction of the maximum mean flux from a contiguous region above which we have confidence that the flux is real
    #                 here, and in the original paper, this value is chosen via trial and error.
    #    diag: useful for debugging - plots diagnostic plots if true.
    highvel = v_helio + 500.
    lowvel = v_helio - 500.
    
    #the paper defines the range of velocities to search as the 1000 km/s around the profile's heliocentric velocity. this finds the indices related to those velocities
    search_highvelind = np.argmin([np.abs(v-highvel) for v in vels])
    search_lowvelind = np.argmin([np.abs(v-lowvel) for v in vels])
    lowind = np.minimum(search_highvelind, search_lowvelind)
    highind = np.maximum(search_highvelind, search_lowvelind)

    #now we need to go through lowind, highind and find all the places where there are three or more consecutive positive points.
    flux_of_interest = profile[lowind:highind]
    vels_of_interest = vels[lowind:highind]
    inds_gtr0 = [i for i in range(len(flux_of_interest)) if flux_of_interest[i] > 0]
    consecutive_inds = []
    i = 0
    while i < len(inds_gtr0)-1:
        current_cons = []
        #this iterates through the consecutive indices in order to tell you how big one consecutive chunk is
        while inds_gtr0[i+1] - inds_gtr0[i] == 1 and i < len(inds_gtr0)-2:
            current_cons.append(inds_gtr0[i])
            i += 1
        if inds_gtr0[i] - inds_gtr0[i-1] == 1:
            current_cons.append(inds_gtr0[i])
        #if the consecutive chunk is > 3 indices long, you can include it as a chunk of interest.
        if len(current_cons) >= 3:
            consecutive_inds.append(current_cons)
        i += 1
    #find the mean fluxes of the consecutive chunks    
    consec_flux = [[flux_of_interest[i] for i in cons_group] for cons_group in consecutive_inds]
    mean_cs_fluxes = [np.mean(gp) for gp in consec_flux]
    #now we choose the regions that have flux > mean_thresh*mean maximum flux.
    max_mean = np.max(mean_cs_fluxes)
    abovethresh_inds = [consecutive_inds[i] for i in range(len(consec_flux)) if mean_cs_fluxes[i] > mean_thresh*max_mean]
    abovethresh_fluxes = [consec_flux[i] for i in range(len(consec_flux)) if mean_cs_fluxes[i] > mean_thresh*max_mean]
    #now that we have the regions above the threshold, we can flatten and then use the minimum and maximum values to get the 
    #region that is most likely to be real flux.
    flat_threshinds = [ind for gp in abovethresh_inds for ind in gp]
    vel_lowind = np.min(flat_threshinds)
    vel_upind = np.max(flat_threshinds)
    
    int_vels = vels_of_interest[vel_lowind: vel_upind]
    int_fluxes = flux_of_interest[vel_lowind: vel_upind]
    #intensity weighted mean velocity is: divide the (channel flux*velocity) / (integrated flux)
    int_meanvel = np.trapz([v*f for v,f in zip(int_vels, int_fluxes)], int_vels) / np.trapz(int_fluxes, int_vels)

    #FOR DIAGNOSTIC / DEBUGGING PURPOSES:
    #this plot makes sure we're selecting sensical reasons with our consec_flux.
    if diag:
        print int_meanvel, v_helio
        fig, ax = plt.subplots(1)
        ax.plot(flux_of_interest)
        for i in range(len(consec_flux)):
            ax.fill_between(consecutive_inds[i], consec_flux[i], color = 'black', alpha = 0.2)
        for i in range(len(abovethresh_inds)):
            ax.fill_between(abovethresh_inds[i], abovethresh_fluxes[i], color = 'black', alpha = 0.7)
        plt.show()

    #we need to also return indices for the region of interest we integrate the curve of growth over...
    #we will also take care of the case that this is beyond the boundaries of the profile here as well...
    low_integvel = np.maximum(vel_lowind - (vel_upind-vel_lowind) + lowind,0)
    high_integvel = np.minimum(len(profile)-1, vel_upind + (vel_upind-vel_lowind) + lowind)

    return int_meanvel, low_integvel, high_integvel
    
def find_first_zero(array, interpolate = False):
    '''given an array, finds the index of the first place where the array is either equal to zero or crosses zero'''
    cross_inds = [i for i in range(len(array)-1) if np.sign(array[i]) != np.sign(array[i+1]) or array[i] == 0]
    if len(cross_inds) == 0:
        zval = np.argmin(array)
    elif not interpolate:
        zval = np.min(cross_inds)
    else:
        #in this case, we interpolate linearly to find the place where our curve equals zero
        lower = np.min(cross_inds)
        upper = lower+1
        slope = (array[upper]-array[lower])
        b = array[upper]-slope*float(upper)
        
        if slope != 0:
            zval = -b/slope
        else: 
            zval = lower + 0.5
    return zval

def calculate_full_integral(flux, velocities, vel_cent, low_bound, high_bound, which_integral = 0, diag = False):
    '''this returns the flux integral as a function of velocity from center. the additional argument tells if you
    are calculating the lower velocity integral (-1), the upper velocity integral (1), or both (0)'''
    #arguments:
    #    flux: flux values
    #    velocities: velocities that correspond to the channels in flux
    #    vel_cent: the rounded channel index of the intensity weighted mean velocity for the real emission in the profile
    #    low_bound, high_bound: the bounds for the region used to calculate the curve of growth
    
    #the 0.001 here converts the fluxes from mJy km/s to Jy km/s
    left_vels = [0.001*np.trapz(flux[i:vel_cent], -velocities[i:vel_cent]) for i in np.arange(vel_cent-1, low_bound, -1)]
    right_vels = [0.001*np.trapz(flux[vel_cent:i], -velocities[vel_cent:i]) for i in np.arange(vel_cent+1, high_bound, 1)]
    full_vels = [l+r for l,r in zip(left_vels,right_vels)]
    
    #now we select which vels we are using here:
    if which_integral == 0:
        sel_int = full_vels
    elif which_integral == -1:
        sel_int = left_vels
    else:
        sel_int = right_vels
    
    #these help us determine where the "flat part of our curve of growth" begins, which is important for normalization and finding the flux
    velgrad = np.gradient(sel_int)
    cross_ind = find_first_zero(velgrad)

    flux_val = np.median(sel_int[cross_ind:])
    if flux_val == 0:
        #this is a contingency measure 
        flux_val = np.amax(sel_int[cross_ind:])
    normalized_cog = [f/flux_val for f in sel_int]
    
    if diag:
        plt.plot([0,len(velgrad)],[0,0])
        plt.scatter(cross_ind, 0)
        plt.plot(velgrad)
        plt.title('Gradient')
        plt.show()  

        plt.plot(left_vels)
        plt.scatter(cross_ind, full_vels[cross_ind])
        plt.plot(right_vels)
        plt.plot(full_vels)
        plt.show()
    return flux_val, normalized_cog

def find_velocity(normalized_curve, velocity_thresh, velocities, interpolate = False):
    '''given a normalized curve of growth, finds the width of the profile at velocity_thresh percent of the flux density.'''
    curve_through_thresh = [n-velocity_thresh for n in normalized_curve]

    first_zero_ind = find_first_zero(curve_through_thresh, interpolate = interpolate)
    if first_zero_ind.is_integer():
        rotvel = velocities[first_zero_ind]
    else:
        #this interpolates the velocity to the non-integer index found if interpolate = True
        int_index = int(first_zero_ind)
        delta = velocities[int_index+1] - velocities[int_index]
        vel_delt = (first_zero_ind - float(int_index))*delta
        rotvel = velocities[int_index] + vel_delt
    return rotvel


def cog_velocities(velocities, flux, v_helio, vel_thresh = 0.85, which_cog = 0, diagnose = False, interp = False):
    ''' this is really the wrapper that you call outside of this script to get er all done in one place.'''
    #arguments:
    #    vel_thresh: the fraction of the integrated flux that defines the velocity width
    #    which_cog: this tells whether you calculate using just the profile integrated to higher velocities (-1), lower velocities (1), or using the full profile (0)
    #    diag: for diagnosing problems - if true, each method spits out diagnostic images.
    center, low_integrange, high_integrange = find_velrange_centvel(velocities, v_helio, flux, diag = diagnose)
    centind = np.argmin([np.abs(v-center) for v in velocities])
    
    if which_cog == 0:
        vels_for_cog = [velocities[centind]-velocities[centind+j] for j in range(1,np.minimum(centind-low_integrange, high_integrange-centind))]
    elif which_cog == -1:
        vels_for_cog = [velocities[centind-j]-velocities[centind] for j in range(1,centind-low_integrange)]
    else:
        vels_for_cog = [velocities[centind]-velocities[centind+j] for j in range(1,high_integrange-centind)]
        
    flux, norm_cog = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = which_cog, diag = diagnose)
    
    vel = find_velocity(norm_cog, vel_thresh, vels_for_cog, interpolate = interp)
    return vel

def old_cog_velocities(velocities, flux, v_helio, vel_thresh = 0.85, which_cog = 0, diagnose = False, interp = False):
    ''' this is really the wrapper that you call outside of this script to get er all done in one place.'''
    #arguments:
    #    vel_thresh: the fraction of the integrated flux that defines the velocity width
    #    which_cog: this tells whether you calculate using just the profile integrated to higher velocities (-1), lower velocities (1), or using the full profile (0)
    #    diag: for diagnosing problems - if true, each method spits out diagnostic images.

    #first, find the center and edges of the "range of interest" - N channels around center define the probable line, low_integrange = 1.5N lower index than center, high_integrange = 1.5N higher index than center
    center, low_integrange, high_integrange = find_velrange_centvel(velocities, v_helio, flux, diag = diagnose)
    #the center above is an intensity weighted average, so it doesn't necessarily align exactly with a channel index - this finds the closest channel
    centind = np.argmin([np.abs(v-center) for v in velocities])
    #this redefines the velocity axis relative to the central velocity channel    
    vels_for_cog = [velocities[centind]-velocities[centind+j] for j in range(1,np.minimum(centind-low_integrange, high_integrange-centind))]
    #this calculates the normalized full curve of growth (returned as norm_cog), where the cog = 1 when the integrated flux under the line = flux.
    flux, norm_cog = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = which_cog, diag = diagnose)
    #this uses the normalized curve of growth, along with the fractional threshold for the velocity
    vel = find_velocity(norm_cog, vel_thresh, vels_for_cog, interpolate = interp)
    return vel

       

