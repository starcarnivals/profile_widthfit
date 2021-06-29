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
    
    ### UPDATE 12/2: using max flux density rather than max flux per channel. helps with narrow line features outside of line center. to ignore, comment to END OF UPDATE 12/2
    sum_cs_fluxes = [np.sum(gp) for gp in consec_flux]
    max_mean = mean_cs_fluxes[np.argmax(sum_cs_fluxes)]
    ### END OF UPDATE 12/2
    
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
    meanvel_ind = np.argmin([np.abs(v-int_meanvel) for v in vels_of_interest])
    # ADDITION 3/5:: one issue for some profiles is that for some reason the integral doesn't give a mean velocity in the region of interest.... this seems strange and problematic to me, but for the time being so things don't crash, we're adding this stopgap measure so if it chooses a not-sensible central velocity, we choose the velocity at the center of the region of interest instead.
    if meanvel_ind < vel_lowind or meanvel_ind > vel_upind:
        #print 'uh oh, outside region of interest!'
        meanvel_ind = (vel_lowind + vel_upind)/2
        int_meanvel = vels_of_interest[meanvel_ind]
    
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
        #we also want to plot the line of heliocentric velocity
        ax.plot([meanvel_ind, meanvel_ind],[np.amin(profile), np.amax(profile)])
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
    if np.isnan(zval):
        zval = 0.
    return zval

def calculate_full_integral(flux, velocities, vel_cent, low_bound, high_bound, which_integral = 0, diag = False, find_slope = False):
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
    ## UPDATE FROM BELOW:
    '''if which_integral == 0:
        sel_int = full_vels
    elif which_integral == -1:
        sel_int = left_vels
    else:
        sel_int = right_vels'''

    if which_integral == -1 and len(left_vels) > 1:
        sel_int = left_vels
    elif which_integral == 1 and len(right_vels) > 1:
        sel_int = right_vels
    elif which_integral == 0 and len(full_vels) > 1:
        sel_int = full_vels
    else:
        if len(left_vels) > 1:
            sel_int = left_vels
        else:
            sel_int = right_vels
            if len(right_vels) < 1:
                print 'bad news sir!'
    
    #these help us determine where the "flat part of our curve of growth" begins, which is important for normalization and finding the flux
    velgrad = np.gradient(sel_int)
    cross_ind = find_first_zero(velgrad)

    flux_val = np.median(sel_int[cross_ind:])
    if flux_val == 0:
        #this is a contingency measure 
        flux_val = np.amax(sel_int[cross_ind:])
    normalized_cog = [f/flux_val for f in sel_int]
    
    ## Find slopes is for my use in making shapes....
    #okay, i think we do polyfit for the normalized_cog[0:crossind] and sel_int[0:crossind]?
    if find_slope and cross_ind != 0:
        slope, intc = np.polyfit(sel_int[0:cross_ind], normalized_cog[0:cross_ind],1)
    elif find_slope:
        slope = -100.
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
    if find_slope:
        retvals = flux_val, normalized_cog, slope
    else:
        retvals = flux_val, normalized_cog
    return retvals

def find_velocity(normalized_curve, velocity_thresh, velocities, interpolate = False):
    '''given a normalized curve of growth, finds the width of the profile at velocity_thresh percent of the flux density.'''
    curve_through_thresh = [n-velocity_thresh for n in normalized_curve]

    first_zero_ind = float(find_first_zero(curve_through_thresh, interpolate = interpolate))
    if first_zero_ind.is_integer():
        rotvel = velocities[int(first_zero_ind)]
    else:
        #this interpolates the velocity to the non-integer index found if interpolate = True
        int_index = int(first_zero_ind)
        #okay... this is a TEMPORARY STOPGAP to prevent 
        if int_index < len(velocities)-1 and int_index > 0:
            delta = velocities[int_index+1] - velocities[int_index]
            vel_delt = (first_zero_ind - float(int_index))*delta
            rotvel = velocities[int_index] + vel_delt
        elif int_index == 0:
            rotvel = velocities[0]
        elif len(velocities) == 0:
            rotvel = 0.
        else:
            rotvel = velocities[len(velocities)-1]
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


#def w_mean50(velocities, flux, v_helio, rms, diagnose = False):
def cog_shapes(velocities, flux, v_helio, diagnose = False):
    '''calculates the shape parameters C_V, A_F, A_C for the given profiles'''
    center, low_integrange, high_integrange = find_velrange_centvel(velocities, v_helio, flux, diag = diagnose)
    centind = np.argmin([np.abs(v-center) for v in velocities])
    
    vels_tot = [velocities[centind]-velocities[centind+j] for j in range(1,np.minimum(centind-low_integrange, high_integrange-centind))]
    vels_n1 = [velocities[centind-j]-velocities[centind] for j in range(1,centind-low_integrange)]
    vels_1 = [velocities[centind]-velocities[centind+j] for j in range(1,high_integrange-centind)]

    fl_tot, cog_tot = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = -1, diag = diagnose)
    ## FIRST -- A_F = flux from one side to the other.
    fl_1, cog_1, slope_1 = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = 1, find_slope = True, diag = diagnose)
    fl_n1, cog_n1, slope_n1 = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = -1, find_slope = True, diag = diagnose)
    #a_f is the ratio of fluxes in the two sides of the profile.... thats whats below:
    a_f = np.nanmax([fl_1, fl_n1])/np.nanmin([fl_1, fl_n1])
    
    #c_v is the concentration of the v_85/v_25
    v_85 = find_velocity(cog_tot, 0.85, vels_tot, interpolate = True)
    v_25 = find_velocity(cog_tot, 0.25, vels_tot, interpolate = True)
    if v_25 != 0:
        c_v = v_85/v_25
    else:
        c_v = np.inf
    
    #a_c is the ratio of slopes in the flat parts of the cogs on either side...?
    a_c = np.nanmax([slope_1, slope_n1])/np.nanmin([slope_1, slope_n1])
    
    return c_v, a_f, a_c
    

