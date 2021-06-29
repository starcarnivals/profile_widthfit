import numpy as np
import matplotlib.pyplot as plt
import cog_velocities as cogvel

###------------------------------------ DEPENDENCIES ------------------------- 
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
    
    ## THIS IS THE DIFFERENCE from cog_velocities.py -- we use the maximum contiguous sum rather than the maximum mean value.
    sum_cs_fluxes = [np.sum(gp) for gp in consec_flux]

    #now we choose the regions that have flux > mean_thresh* flux of "realest" part of profile
    #we've tested calling the realest the highest flux per channel, but this often gave noise spikes. instead, we use the highest flux integral because this means a region has many high significance channels
    max_mean = mean_cs_fluxes[np.argmax(sum_cs_fluxes)]

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
        ax.plot([meanvel_ind, meanvel_ind],[np.amin(profile), np.amax(profile)])
        ax.plot([vel_lowind, vel_lowind],[np.amin(profile), np.amax(profile)], c = 'black')
        ax.plot([vel_upind, vel_upind],[np.amin(profile), np.amax(profile)], c = 'black')
        plt.show()
        
        print 'int_meanvel = ',int_meanvel, 'numerator trapz', np.trapz([v*f for v,f in zip(int_vels, int_fluxes)], int_vels), 'denom trapz', np.trapz(int_fluxes, int_vels)
        print 'interval of interest...', vel_lowind,vel_upind

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

def find_velocity(normalized_curve, velocity_thresh, velocities, interpolate = False):
    '''given a normalized curve of growth, finds the width of the profile at velocity_thresh percent of the flux density.'''
    curve_through_thresh = [n-velocity_thresh for n in normalized_curve]

    first_zero_ind = find_first_zero(curve_through_thresh, interpolate = interpolate)
    if first_zero_ind.is_integer():
        rotvel = velocities[int(first_zero_ind)]
    else:
        #this interpolates the velocity to the non-integer index found if interpolate = True
        int_index = int(first_zero_ind)
        #okay... this is a TEMPORARY STOPGAP to prevent 
        if int_index < len(velocities) and int_index > 0:
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

#### --------------------------------- VELOCITIES ---------------------------------------
def cog_velocities(velocities, flux, v_helio, vel_thresh = 0.85, which_cog = 0, diagnose = False, interp = False):
    ''' this gives the curve of growth velocities for the profile, and can be called outside of this script to calculate a curve of growth velocity.'''
    #arguments:
    #    vel_thresh: the fraction of the integrated flux that defines the velocity width
    #    which_cog: this tells whether you calculate using just the profile integrated to higher velocities (-1), lower velocities (1), or using the full profile (0)
    #    diag: for diagnosing problems - if true, each method spits out diagnostic images.
    #just changed from cogvel version
    center, low_integrange, high_integrange = find_velrange_centvel(velocities, v_helio, flux, diag = diagnose)
    centind = np.argmin([np.abs(v-center) for v in velocities])
    
    if which_cog == 0:
        vels_for_cog = [velocities[centind]-velocities[centind+j] for j in range(1,np.minimum(centind-low_integrange, high_integrange-centind))]
    elif which_cog == -1:
        vels_for_cog = [velocities[centind-j]-velocities[centind] for j in range(1,centind-low_integrange)]
    else:
        vels_for_cog = [velocities[centind]-velocities[centind+j] for j in range(1,high_integrange-centind)]
    #just changed from cogvel version...    
    flux, norm_cog = calculate_full_integral(flux, velocities, centind, low_integrange, high_integrange, which_integral = which_cog, diag = diagnose)
    
    vel = find_velocity(norm_cog, vel_thresh, vels_for_cog, interpolate = interp)
    return vel


### I think one thing that can improve our program is changing calculate_full_integral to define the range used to determine the flux 
def calculate_full_integral(flux, velocities, vel_cent, low_bound, high_bound, rms, which_integral = 0, diag = False):
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

#### final width-finders -- these are the programs that should be called outside of this program once we're done fixing the dependencies - they calculate the profile widths.

def w_mean50(fullprof,fullvels, vhel, rms, diagnose = False, w50 = 0, sn_prof = 10., pass_vhel = False, lowind = 0, highind = 0):
    #find the (intensity weighted) central velocity of the profile
    if pass_vhel:
        #this is so that you can pass the heliocentric velocity from a different v_hel finding method (E.g., from the matched filtering method....
        mean = vhel
        low = lowind
        high = highind
    else:
        mean, low, high = find_velrange_centvel(fullvels, vhel, fullprof, diag = diagnose)

    centind = np.argmin([np.abs(v-mean) for v in fullvels])
    vels_for_cog_1 = [fullvels[centind-j]-fullvels[centind] for j in range(1,centind-low)]
    flux, norm_cog_1 = calculate_full_integral(fullprof, fullvels, centind, low, high, rms, which_integral = -1, diag = diagnose)

    vels_for_cog_2 = [fullvels[centind]-fullvels[centind+j] for j in range(1,high-centind)] 
    flux, norm_cog_2 = calculate_full_integral(fullprof, fullvels, centind, low, high, rms, which_integral = 1, diag = diagnose)
   
    #if len(norm_cog_2) != len(vels_for_cog_2):
    #if len(norm_cog_1) != len(vels_for_cog_1):
    #this finds 95% of the integrated flux on either side of the profile
    vel_1 = find_velocity(norm_cog_1, 0.95, vels_for_cog_1, interpolate = True)
    vel_2 = find_velocity(norm_cog_2, 0.95, vels_for_cog_2, interpolate = True)
    
    leftind = np.argmin([np.abs(v-(mean-vel_1)) for v in fullvels])
    rightind = np.argmin([np.abs(v-(mean+vel_2)) for v in fullvels])
    
    #print 'the indices of the left and right sides of the profile:',leftind,rightind
    #we want to use 50% of the mean flux between the leftind and rightind values?
    meanflux = np.mean(fullprof[rightind:leftind])
    flux_thresh_50 = 0.5*meanflux
    
    if low < centind:
        low_max_ind = np.argmax(fullprof[low:centind]) + low
    else:
        low_max_ind = low
    if centind < high:
        high_max_ind = np.argmax(fullprof[centind:high]) + centind
    else:
        high_max_ind = centind
    
    
    if float(low_max_ind - low) < 0.25*float(centind - low) and int(1.1*low_max_ind) < centind:
        low_max_ind = np.argmax(fullprof[int(1.1*low_max_ind):centind]) + int(1.1*low_max_ind)

    low_region_interest = fullprof[low:low_max_ind] - flux_thresh_50
    high_region_interest = fullprof[high_max_ind:high] - flux_thresh_50
    
    #we do need to unfortunately reverse the low_Region_index range
    #so that the first intersect is the one that's closest to the peak.
    low_region_interest = low_region_interest[::-1]
    
    if len(low_region_interest) != 0 and len(high_region_interest) != 0:

        index_low_wm50 = low_max_ind - find_first_zero(low_region_interest, interpolate = True)
        index_high_wm50 = high_max_ind + find_first_zero(high_region_interest, interpolate = True)
        low_region_interest = low_region_interest[::-1]
        
        low_z_inds = low + np.where(np.diff(np.sign(low_region_interest)))[0]
        high_z_inds = high_max_ind + np.where(np.diff(np.sign(high_region_interest)))[0]
        #calculate the flux per channel for the region mean:possible edge, and just like,,,,, where the average stops increasing 
        #by more than the noise?
        #so i think in some cases, you may end up with regions that don't switch signs within the "region of interest".....
        if len(low_z_inds) == 0:
            low_z_inds = [low_max_ind]
        if len(high_z_inds) == 0:
            high_z_inds = [high_max_ind]
        
        low_z_fluxes = [0.001*np.trapz(fullprof[i:centind], -fullvels[i:centind]) for i in low_z_inds]
        low_z_pcs = [f/(np.sqrt(float(centind-i))*rms) for f,i in zip(low_z_fluxes, low_z_inds)]        
        low_z_fluxes = low_z_fluxes / np.std(low_z_fluxes)

        high_z_fluxes = [0.001*np.trapz(fullprof[centind:i], -fullvels[centind:i]) for i in high_z_inds]
        high_z_pcs = [f/(rms*np.sqrt(float(i-centind))) for f,i in zip(high_z_fluxes, high_z_inds)]        
        high_z_fluxes = high_z_fluxes / np.std(high_z_fluxes)
        
        #ok, FOR NOW: i'm gonna use the highest per channel flux value
        low_maxind_dist = low_z_inds[np.argmax(low_z_pcs)] - low
        high_maxind_dist = high_z_inds[np.argmax(high_z_pcs)] - high_max_ind
                
        #print 'my attempt to do max per chan / integral limiting: ',low_maxind_dist + low, high_maxind_dist+high_max_ind
        #print 'current indices: ',index_low_wm50, index_high_wm50
        int_index_low = low_maxind_dist + low
        int_index_high = high_maxind_dist + high_max_ind
        index_low_wm50 = int_index_low
        index_high_wm50 = int_index_high
        
        int_index_low = int(index_low_wm50)
        delta = fullvels[int_index_low+1] - fullvels[int_index_low]
        vel_delt = (index_low_wm50 - float(int_index_low))*delta
        rotvel_low = fullvels[int_index_low] + vel_delt

        int_index_high = int(index_high_wm50)
        delta = fullvels[int_index_high+1] - fullvels[int_index_high]
        vel_delt = (index_high_wm50 - float(int_index_high))*delta
        rotvel_high = fullvels[int_index_high] + vel_delt

        w_m50 = np.abs(rotvel_high - rotvel_low)
        
        inner_mpc = np.mean(fullprof[int_index_low:int_index_high])
        outer_rms = np.std(np.append(fullprof[low:int_index_low], fullprof[int_index_high:high]))
        sn_prof = inner_mpc / outer_rms
        
        #using the courtois et al. 2009 formalism
        if sn_prof > 17:
            w_err = 8.
        elif 2 < sn_prof <= 17:
            w_err = 21.6 - 0.8*sn_prof
        else:
            w_err = 70. - 25.*sn_prof
        if diagnose:#np.abs(w50 - w_m50) > 20.*w_err and diagnose:#5.*alferr:
            print 'for the below, ALFALFA profile width: ',w50, ' and mine: ',w_m50
            fig, ax = plt.subplots(1)
            ax.plot(fullvels, fullprof, c = 'dodgerblue')
            ax.plot([mean, mean], [np.amin(fullprof[low:high]),np.amax(fullprof[low:high])], c = 'grey', label = 'Central velocity')
            ax.plot([fullvels[leftind+5],fullvels[rightind-5]],[meanflux,meanflux], label = 'Mean flux', c = 'goldenrod')
            ax.plot([fullvels[leftind+5],fullvels[rightind-5]],[0.5*meanflux,0.5*meanflux], label = 'Profile threshold', c = 'forestgreen')

        #so he plotted points tell us about the position of the region we'll search to find the intersect between the profile and flux_thresh_50
            #ax.scatter([fullvels[leftind], fullvels[rightind]], [fullprof[leftind],fullprof[rightind]])
            ax.fill_between(fullvels[rightind:leftind+1], fullprof[rightind:leftind+1], color = 'dodgerblue', alpha = 0.2, label = 'Values included for mean')
            #plt.scatter(fullvels[low:low_max_ind],low_region_interest + flux_thresh_50)
            #plt.scatter(fullvels[high_max_ind:high],high_region_interest + flux_thresh_50)
            ax.scatter([rotvel_low, rotvel_high],[flux_thresh_50, flux_thresh_50], label = 'Profile width', marker = '*', c = 'black', s = 50)
            #plt.scatter([fullvels[rightind], fullvels[leftind]],[fullprof[rightind], fullprof[leftind]], marker = '*', c = 'black')
            ax.legend()
            plt.xlim(fullvels[low],fullvels[high])
            plt.title(r'$W_{50, mean}$ Program')
            plt.show()
    else:
        w_m50 = 0
        w_err = 1e3
        
    return w_m50, w_err