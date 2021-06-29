### THIS SCRIPT UPDATES THE COG-BASED VELOCITY METHODS IN ORDER TO INCORPORATE A V_HELIO-FINDING METHOD WHICH USES MATCHED FILTERING -- THIS METHOD PERFORMS BETTER IN LOW S/N PROFILES, MEANING IT HAS SIGNIFICANT UTILITY FOR US.
import pickle
import numpy as np
import matplotlib.pyplot as plt
import w_50mean as tw50
import cog_velocities as cog

### THE DIFFERENCE BETWEEN THIS SCRIPT AND THE OTHER ONES IN THIS FOLDER IS THE FOLLOWING METHOD --- here, this uses matched filtering to measure v_hel; other methods used some creative averaging over the full profile.
def sigma_from_dist(peakdist):
    #this just puts everything into the one function that you run. peakdist can be in indices, we'll convert to km/s
    a0oa2 = 1.75    
    coeff = 1./np.sqrt(5./2.-a0oa2/np.sqrt(2.))
    if peakdist < 60.0 and peakdist > 6.0:
        pdcorr = (19.3+0.66*peakdist)
        sigma = pdcorr/2.0*coeff
    elif peakdist <= 6.0:
        #these ones are just going to be gaussians.
        sigma = peakdist
    elif peakdist >= 60.0:
        sigma = (peakdist/2.0)*coeff
    return sigma

def binProfFac(fullprofile, binfactor):
    #this creates lower-resolution versions of profiles by combining the flux from the binfactor adjacent bins.
    binnedprof = np.zeros(len(fullprofile)/binfactor)
    for i in range(len(binnedprof)):
        for j in range(binfactor):
            binnedprof[i]+=fullprofile[binfactor*i + j]
    return binnedprof

def findZeroes(array):
    #this finds the indices that bracket zeroes in an array with discrete values - it accounts both for zero values and also for places where the gradient sign changes from one position to the next, which means that the zero is between the two indices
    zeroes = []
    for i in range(len(array)-1):
        currval = array[i]
        nextval = array[i+1]
        if currval == 0.0 and i not in zeroes:
            zeroes.append(i)
        elif nextval == 0.0 and i+1 not in zeroes:
            zeroes.append(i+1)
        else:
            currsign = currval / np.abs(currval)
            nextsign = nextval / np.abs(nextval)
            if not currsign == nextsign:
                #this tells you there's a sign change between the indices. approximate using the closer index.
                if np.abs(currval) < np.abs(nextval) and i not in zeroes:
                    zeroes.append(i)
                elif i+1 not in zeroes:
                    zeroes.append(i+1)
    return zeroes

def findhigh(array,maxzeroes):
    if len(array) == 0:
        maxval = 0.0
        maxind = 512
    else:
        maxval = np.nanmax(array)
        maxind = maxzeroes[np.argmax(array)]
    return maxval, maxind

def correlationFinder(profile, indlow, indhigh, templates):
    '''provides an initial estimate at the width of the profile and the center of the profile using cross-correlation functions between Hermite polynomials and the profile. Runs through Hermite polynomials that have peaks at distances that are equal to the distance between peaks in the profile, in order to lower the number of CCFs that must be computed.'''
    truncprof = profile[indlow:indhigh]
    #this section of the function finds the distances between peaks by finding zeroes in the gradient of the binned profile - the profile is binned to make these peaks less sensitive to spurious noise peaks, and we use the zeroes of the gradient so that we are selecting local maxima
    binnedprofile = binProfFac(truncprof,4)

    lquad = int(0.2*float(len(binnedprofile)))
    rquad = int(0.85*float(len(binnedprofile)))
    peakfinder = binnedprofile[lquad:rquad]
    if len(peakfinder) <= 5:
        peakfinder = binnedprofile
    grad = np.gradient(peakfinder)
    zpf = findZeroes(grad) 

    zeroes = [4*(p + lquad)+indlow for p in zpf]
    locmaxes = [z-4+np.argmax(profile[z-4:z+8]) for z in zeroes]
    distances = np.array([[np.abs(locmaxes[i] - locmaxes[j]) for j in range(len(locmaxes))] for i in range(len(locmaxes))])
    
    flatdists = distances.flatten()

    sing, freq = np.unique(flatdists, return_counts=True)
    flatdists = np.trim_zeros(sing)
    #flatdists contains the distances between peaks that we will use to select profile templates for use in our CCF
    flatdists = [f for f in flatdists if f > 2 and f <= len(templates)]
    
    #we compute the profile fft once for use in all of our CCFs (which we compute using fft of profile * fft of template)
    proffft = np.fft.fft(profile)
    profacf = np.fft.ifft(proffft*np.conj(proffft)).real
    N = 1024.
    gsquared = [p**2.0 for p in profile]
    sigg = np.sqrt(1./N*np.sum(gsquared))
    
    #GIVEN SOME LIST OF **ROLLED** TEMPLATE FFTS, WHERE THE TEMPLATE INDEX CORRESPONDS TO THE DIST PARAMETER.
    #we have tbankfile so we can load in the same file of templates every time.
    #oops... The profiles in templatebank are 1 indexed (i.e. the 0th entry has a distance of 1 points between entries)
    selectedtemps = [templates[d-1] for d in flatdists]
    crosscorrs = [np.fft.ifft(proffft*tempfft) for tempfft in selectedtemps]
    
    #finds the peak of each ccf (maxvalues), as well as the position of that maximum within the profile (maxinds)
    maxfinder = [np.gradient(ccorr[indlow:indhigh]) for ccorr in crosscorrs]
    maxzeroes = [[z for z in findZeroes(maxfinderforprof)] for maxfinderforprof in maxfinder]
    localextrvals = [[crosscorrs[i][z+indlow] for z in maxzeroes[i]] for i in range(len(maxzeroes))]
    maxzsinds = [findhigh(localextrvals[i],maxzeroes[i]) for i in range(len(localextrvals))]
    maxvalues = [maxvals[0] for maxvals in maxzsinds]
    maxinds = [maxvals[1] + indlow for maxvals in maxzsinds]    

    if len(maxvalues) > 0:
        maxcorr = np.nanmax(maxvalues)
        maxdists = [flatdists[i] for i in range(len(flatdists)) if maxvalues[i] == maxcorr]
    else:
        maxcorr = 0
        maxdists = 0    
   
    maxtemp = np.fft.ifft(selectedtemps[np.argmax(maxvalues)])
    return maxvalues, maxinds, flatdists, maxtemp


##----------------------------------------- THE FUNCTION THAT FINDS THE CORRELATIONS --------------------------------------## 

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


def calculate_full_integral(flux, velocities, vel_cent, low_bound, high_bound, which_integral = 0, diag = False, find_slope = False, plot_title = ''):
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

        plt.plot(left_vels, c = 'grey', linestyle = 'dashed', label = 'Left COG')
        plt.scatter(cross_ind, full_vels[cross_ind], label = 'COG turnover')
        plt.plot(right_vels, c = 'grey', label = 'Right COG')
        plt.plot(full_vels, c = 'black', label = 'Symmetric COG')
        #add the median value
        plt.plot([flux_val for v in range(len(full_vels))], c = 'red', label = 'Median integrated value')
        plt.legend()
        plt.title(plot_title+' Curve of Growth')
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


def cog_vels_and_shapes(velocities, flux, tbank, indlow, indhigh, diagnose = False, batchrun = False, interp = True, which_cog = 0, rms = 0, returnFlux = False, pt = ''):
    #this way you have to run the costly part of the program (finding the v_Helio)
    if not batchrun:
        templatebank = pickle.load(open(tbank, 'rb')) 
    else:
        templatebank = tbank
    
    corrvals, corrinds, peakdists, centtemp = correlationFinder(flux, indlow, indhigh,tbank)
    corrindstrunc = [c - indlow for c in corrinds]
    profcenter = int(corrindstrunc[np.argmax(corrvals)]) 
    centvel = velocities[profcenter]
        #ok now we need the boundaries
    mostlikelydistance = peakdists[np.argmax(corrvals)]
    sig = sigma_from_dist(mostlikelydistance)
    low_integrange = np.maximum(0, int(profcenter-6*mostlikelydistance))
    high_integrange = np.minimum(int(profcenter+6*mostlikelydistance), len(velocities)-1)
    #print 'centerind', profcenter, 'low for cog',low_integrange, 'high for cog', high_integrange, 'length of truncated profile', len(velocities), 'index of low part of truncated profile', indlow, indhigh
    
    ## okay bruv luv let me just make sure the issue here is the stupid v_hel thing
    mvel, low_old, high_old = cog.find_velrange_centvel(velocities, velocities[len(velocities)/2], flux[indlow:indhigh])
    
    #print 'MF method: ',low_integrange, centvel, high_integrange,'old method',low_old,mvel, high_old
    flux = flux[indlow:indhigh]
    vels_tot = [velocities[profcenter]-velocities[profcenter+j] for j in range(1,np.minimum(profcenter-low_integrange, high_integrange-profcenter))]
    vels_n1 = [velocities[profcenter-j]-velocities[profcenter] for j in range(1,profcenter-low_integrange)]
    vels_1 = [velocities[profcenter]-velocities[profcenter+j] for j in range(1,high_integrange-profcenter)]

    fl_tot, cog_tot = calculate_full_integral(flux, velocities, profcenter, low_integrange, high_integrange, which_integral = 0, diag = diagnose, plot_title = pt)
    ## FIRST -- A_F = flux from one side to the other.
    fl_1, cog_1, slope_1 = calculate_full_integral(flux, velocities, profcenter, low_integrange, high_integrange, which_integral = 1, find_slope = True, diag = False)
    fl_n1, cog_n1, slope_n1 = calculate_full_integral(flux, velocities, profcenter, low_integrange, high_integrange, which_integral = -1, find_slope = True, diag = False)
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
    
    if which_cog == 0:
        vels_for_cog = [velocities[profcenter]-velocities[profcenter+j] for j in range(1,np.minimum(profcenter-low_integrange, high_integrange-profcenter))]
    elif which_cog == -1:
        vels_for_cog = [velocities[profcenter-j]-velocities[profcenter] for j in range(1,profcenter-low_integrange)]
    else:
        vels_for_cog = [velocities[profcenter]-velocities[profcenter+j] for j in range(1,high_integrange-profcenter)]
    if rms != 0:
        wm50, err = tw50.w_mean50(flux, velocities, centvel, rms, pass_vhel = True, lowind = low_integrange, highind = high_integrange)
    else:
        wm50 = 0.   
    flx, norm_cog = calculate_full_integral(flux, velocities, profcenter, low_integrange, high_integrange, which_integral = which_cog, diag = False)
    
    vel_65 = find_velocity(norm_cog, 0.65, vels_for_cog, interpolate = interp)
    vel_75 = find_velocity(norm_cog, 0.75, vels_for_cog, interpolate = interp)
    vel_85 = find_velocity(norm_cog, 0.85, vels_for_cog, interpolate = interp)

    ## OKAY.... we can do a diagnostic image???
    #PLOT: the profile, the central velocity.... the vel_65, 75 and 85 on either side of the central velocity?
    if diagnose:
        plt.plot([centvel, centvel],[0,1], c = 'grey', label = 'V_hel')
        plt.plot(velocities, flux, c = 'black')
        plt.plot([centvel+vel_65, centvel+vel_65], [0,1], c = 'dodgerblue', linestyle = 'dashed', label = 'V65')
        plt.plot([centvel-vel_65, centvel-vel_65], [0,1], c = 'dodgerblue', linestyle = 'dashed')
        plt.plot([centvel+vel_75, centvel+vel_75], [0,1], c = 'forestgreen', label = 'V75')
        plt.plot([centvel-vel_75, centvel-vel_75], [0,1], c = 'forestgreen')
        plt.plot([centvel+vel_85, centvel+vel_85], [0,1], c = 'gold', linestyle = 'dashdot', label = 'V85')
        plt.plot([centvel-vel_85, centvel-vel_85], [0,1], c = 'gold', linestyle = 'dashdot')
        if rms != 0:
            plt.plot(velocities, [rms for i in range(len(flux))], c = 'grey', linestyle = 'dashed', label = 'RMS')
            plt.plot(velocities, [-rms for i in range(len(flux))], c = 'grey', linestyle = 'dashed')
        plt.plot(velocities, np.zeros(len(flux)), c = 'grey')
        plt.legend()
        plt.xlim(centvel-3*vel_85, centvel+3*vel_85)
        plt.title(pt+' COG Velocities Plotted')
        plt.show()
    if returnFlux: 
        return vel_65, vel_75, vel_85, wm50, centvel, c_v, a_f, a_c, flx
    else:
        return vel_65, vel_75, vel_85, wm50, centvel, c_v, a_f, a_c

    
