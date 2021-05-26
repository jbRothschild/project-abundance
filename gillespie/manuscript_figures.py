import os, glob, csv, pickle, copy, time, scipy

import matplotlib as mpl; from matplotlib import colors, ticker
import matplotlib.pyplot as plt; from matplotlib.lines import Line2D
import numpy as np; import pandas as pd
import scipy.io as sio;
from scipy.signal import savgol_filter, find_peaks, argrelextrema
from autocorrelation import autocorrelation_spectrum as autospec
from autocorrelation import average_timescale_autocorrelation, exponential_fit_autocorrelation

np.seterr(divide='ignore', invalid='ignore')

from gillespie_models import RESULTS_DIR, MultiLV
import theory_equations as theqs
from settings import VAR_NAME_DICT, COLOURS, IMSHOW_KW, NPZ_SHORT_FILE\
                    , VAR_SYM_DICT
from manual_revision import DICT_REVISION

MANU_FIG_DIR = 'figures' + os.sep + 'manuscript'
while not os.path.exists( os.getcwd() + os.sep + MANU_FIG_DIR ):
    os.makedirs(os.getcwd() + os.sep + MANU_FIG_DIR);

plt.style.use('custom.mplstyle')


POINTS_BETWEEN_X_TICKS = 20; POINTS_BETWEEN_Y_TICKS = 20

def deterministic_mean(nbr_species, mu, rho, rplus, rminus, K):
    # Deterministic mean fixed point
    det_mean = K*( ( 1. + np.sqrt( 1.+ 4.*mu*( 1. + rho*( nbr_species - 1. ) ) /
                (K*(rplus-rminus)) ) ) / ( 2.*( 1. + rho*(nbr_species-1.) ) ) )
    return int(det_mean)

def meanJ_sim(dstbn, nbr_species):
    return nbr_species * np.dot(dstbn, np.arange( len(dstbn) ) )

def meanJ_est(dstbn, nbr_species):
    meanJ = np.zeros((np.shape(dstbn)[0],np.shape(dstbn)[1]))
    for i in np.arange(np.shape(dstbn)[0]):
        for j in np.arange(np.shape(dstbn)[1]):
            dstbnij = dstbn[i,j]
            meanJ[i,j] = meanJ_sim(dstbnij, nbr_species)
    #f = plt.figure(); fig = plt.gcf(); ax = plt.gca() im = ax.imshow(meanJ.T);
    #plt.colorbar(im,ax=ax); ax.invert_yaxis(); plt.show()
    return meanJ

def death_rate( n, dstbn, rplus, rminus, K, rho, S):
    J = meanJ_sim(dstbn, S) # could change to just meanJ
    r = rplus - rminus
    return rminus * n + r * n * ( ( 1. - rho ) + rho * J ) / K

def richness_from_rates( dstbn, rplus, rminus, K, rho, mu, S ):
    deathTimesP1 = death_rate(1, dstbn, rplus, rminus, K, rho, S)*dstbn[1]
    return ( mu*S - deathTimesP1 )/( deathTimesP1 - mu )

def mfpt_a2b( dstbn, a, b, mu=1.0, rplus=2.0 ):
    """
    From distribution, get the mfpt <T_{b}(a)>, a<b
    """
    mfpt = 0.0
    for i in np.arange(a,b):
        mfpt += np.divide( np.sum( dstbn[:i+1] )  , ( rplus*i + mu )*dstbn[i] )
    return mfpt

def mfpt_b2a( dstbn, a, b, mu=1.0, rplus=2.0 ):
    """
    From distribution, get the mfpt <T_{a}(b)>
    """
    mfpt = 0
    for i in np.arange(a,b):
        mfpt += np.divide( np.sum(dstbn[i+1:]) , ( rplus*i + mu ) * dstbn[i] )
    return mfpt

def mfpt_020( dstbn, mu=1.0 ):
    """
    From distribution, get the mfpt <T(0\rightarrow 0)>
    """
    return np.divide( 1., ( dstbn[0] * mu ) )

def mfpt_a2a( dstbn, a, mu=1.0, rplus=2.0, rminus=1.0, K=100, rho=1.0, S=30 ):
    """
    From distribution, get the mfpt <T_{a}(a)> for a!=0
    """
    return np.divide( ( 1. + dstbn[a] ) , ( dstbn[a]
            * ( rplus*a + mu + a*( rminus + ( rplus-rminus )
            * ( ( 1. - rho ) * a + rho * meanJ_sim(dstbn, S) ) / K ) ) ) )

def smooth(x,window_len=11,window='hanning'):
    """
    Code taken from Scipy Cookbook
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd
        integer window: the type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman' flat window produces a moving average smoothing.

    output:
        the smoothed signal

    see also:
        np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
        scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of
    a string
    NOTE: length(output) != length(input), to correct this:
            return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming'" +
                            ", 'bartlett', 'blackman'")

    s = np.r_[ x [window_len-1:0:-1], x, x[-2:-window_len-1:-1] ]

    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval( 'np.' + window + '(window_len)' )

    y = np.convolve( w / w.sum(), s, mode='valid')
    return y

def correlations_fix(model, filesave):

    print("Warning, previously MLV6 calculated coefficient of variation wrong")

    model.results['corr_ni_nj'] = 0.0
    model.results['coeff_ni_nj'] = 0.0
    model.results['corr_J_n']  = 0.0
    model.results['coeff_J_n']  = 0.0
    model.results['corr_Jminusn_n']  = 0.0
    model.results['coeff_Jminusn_n']  = 0.0
    for i in np.arange(0, model.nbr_species):
            var_J_n = ( ( np.sqrt( model.results['av_ni_sq_temp'][i]
                        - model.results['av_ni_temp'][i]**2 ) ) * (
                        np.sqrt( model.results['av_J_sq']
                        - model.results['av_J']**2 ) ) )
            var_Jminusn_n = ( ( np.sqrt( model.results['av_ni_sq_temp'][i]
                            - model.results['av_ni_temp'][i]**2 ) ) * (
                            np.sqrt( model.results['av_Jminusn_sq'][i]
                            - model.results['av_Jminusn'][i]**2 ) ) )
            # cov(J,n)
            cov_J_n = (model.results['av_J_n'][i] - model.results['av_ni_temp'][i]
                        * model.results['av_J'] )
            # cov(J-n,n)
            cov_Jminusn_n = (model.results['av_Jminusn_n'][i]
                            - model.results['av_ni_temp'][i] *
                            model.results['av_Jminusn'][i] )
            # coefficients of variation
            if model.results['av_J'] != 0.0 and model.results['av_ni_temp'][i] != 0.0:
                model.results['coeff_J_n']+= ( cov_J_n /( model.results['av_J']
                                            * model.results['av_ni_temp'][i] ) )
            if model.results['av_Jminusn'][i] != 0.0 and model.results['av_ni_temp'][i] != 0.0:
                model.results['coeff_Jminusn_n'] += ( cov_Jminusn_n
                                            / ( model.results['av_Jminusn'][i]
                                            * model.results['av_ni_temp'][i] ) )
            # Pearson correlation
            if var_J_n != 0.0:
                model.results['corr_J_n'] += cov_J_n / var_J_n
            if var_Jminusn_n != 0.0:
                model.results['corr_Jminusn_n'] += cov_Jminusn_n / var_Jminusn_n

            for j in np.arange(i+1, model.nbr_species):
                var_nm = ( ( np.sqrt( model.results['av_ni_sq_temp'][i]
                - model.results['av_ni_temp'][i]**2 ) ) * (
                np.sqrt( model.results['av_ni_sq_temp'][j]
                - model.results['av_ni_temp'][j]**2 ) ) )
                # cov(n_i,n_j)
                cov_ni_nj = ( model.results['av_ni_nj_temp'][i][j]
                            - model.results['av_ni_temp'][i] *
                            model.results['av_ni_temp'][j] )
                # coefficients of variation
                if model.results['av_ni_nj_temp'][i][j] != 0.0:
                    model.results['coeff_ni_nj'] += ( cov_ni_nj
                                    / ( model.results['av_ni_temp'][i]
                                    * model.results['av_ni_temp'][j] ) )
                # Pearson correlation
                if var_nm != 0.0:
                    model.results['corr_ni_nj'] += cov_ni_nj / var_nm

    # Taking the average over all species
    model.results['corr_ni_nj'] /= ( model.nbr_species*(model.nbr_species-1)/2)
    model.results['coeff_ni_nj'] /= ( model.nbr_species*(model.nbr_species-1)/2)
    model.results['corr_J_n'] /= ( model.nbr_species)
    model.results['coeff_J_n'] /= ( model.nbr_species)
    model.results['corr_Jminusn_n'] /= ( model.nbr_species)
    model.results['coeff_Jminusn_n'] /= ( model.nbr_species)

    with open(filesave, 'wb') as handle:
        pickle.dump(model.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0

def mlv_consolidate_sim_results(dir, parameter1='immi_rate'
                                    , parameter2='comp_overlap'):
    """
    Analyze how the results from different simulations differ for varying
    (parameter)

    Input :
        dir       : directory that we're plotting from
        parameter1 : the parameter that changes between different simulations
                    (string)
        parameter2 : second parameter that changes between different simulations
                    (string)
    """
    filename =  dir + os.sep + NPZ_SHORT_FILE

    # count number of subdirectories
    nbr_sims        = len( next( os.walk(dir) )[1] )

    # initialize the
    param1          = np.zeros(nbr_sims); param2            = np.zeros(nbr_sims)
    sim_dist_vary   = []                ; rich_dist_vary    = []
    conv_dist_vary  = []                ; mf_dist_vary      = []
    coeff_ni_nj     = np.zeros(nbr_sims); corr_ni_nj        = np.zeros(nbr_sims)
    coeff_J_n       = np.zeros(nbr_sims); corr_J_n          = np.zeros(nbr_sims)
    coeff_Jminusn_n = np.zeros(nbr_sims); corr_Jminusn_n    = np.zeros(nbr_sims)
    # TEMP TIMES
    time_autocor_spec = np.zeros( nbr_sims )
    mean_time_autocor_abund = np.zeros( nbr_sims )
    std_time_autocor_abund  = np.zeros( nbr_sims )
    dominance_turnover      = np.zeros( nbr_sims )
    suppress_turnover       = np.zeros( nbr_sims )
    dominance_return        = np.zeros( nbr_sims )
    suppress_return         = np.zeros( nbr_sims )

    # TODO change to dictionary
    for i in np.arange(nbr_sims):
        sim_nbr = i + 1
        if not os.path.exists(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
                   'results_0.pickle'):
            print( "Missing simulation: " + str(sim_nbr) )
            rich_dist_vary.append( np.array( [0] ) )
            sim_dist_vary.append( np.array( [0] ) )
            #conv_dist_vary.append( np.array( ss_dist_conv ) )
            mf_dist_vary.append( np.array( [0] ) )
        else:
            with open(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
                       'results_0.pickle', 'rb') as handle:
                param_dict  = pickle.load(handle)

            model           = MultiLV(**param_dict)
            theory_model    = theqs.Model_MultiLVim(**param_dict)

            # distribution
            start = time.time()
            ss_dist_sim     = model.results['ss_distribution'] \
                                    / np.sum(model.results['ss_distribution'])
            #ss_dist_conv, _ = theory_model.abund_1spec_MSLV()
            ss_dist_mf, _   = theory_model.abund_sid()
            richness_dist   = model.results['richness']
            rich_dist_vary.append( np.array( richness_dist ) )
            sim_dist_vary.append( np.array( ss_dist_sim ) )
            #conv_dist_vary.append( np.array( ss_dist_conv ) )
            mf_dist_vary.append( np.array( ss_dist_mf ) )

            # TEMP FIX FOR SOME WRONG COEFFICIENT OF VARIATION
            #correlations_fix(model, dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
            #           'results_0.pickle')

            corr_ni_nj[sim_nbr-1] = model.results['corr_ni_nj']
            coeff_ni_nj[sim_nbr-1] = model.results['coeff_ni_nj']
            corr_J_n[sim_nbr-1] = model.results['corr_J_n']
            coeff_J_n[sim_nbr-1] = model.results['coeff_J_n']
            corr_Jminusn_n[sim_nbr-1] = model.results['corr_Jminusn_n']
            coeff_Jminusn_n[sim_nbr-1] = model.results['coeff_Jminusn_n']

            ############################################################################################################################
            # TEMP AUTOCORRELATION TIME SAVING. This is an aweful way of doing it. Fix it.
            ############################################################################################################################
            n=2; fracTime=100
            autocor, _, specAutocor, _, newTimes =\
                    autospec(model.results['times'][n:],\
                    model.results['trajectory'][n:])

            _, time_autocor_spec[sim_nbr-1] = exponential_fit_autocorrelation(specAutocor, newTimes, fracTime)
            mean_time_autocor_abund[sim_nbr-1], std_time_autocor_abund[sim_nbr-1] =\
                    average_timescale_autocorrelation( autocor, newTimes, fracTime)

            S = model.nbr_species; K = model.carry_capacity
            mu = model.immi_rate; rho = model.comp_overlap
            rplus = model.birth_rate; rminus = model.death_rate
            nbr_species = int( S*(1.0-ss_dist_sim[0]) )
            nbr = deterministic_mean(nbr_species, mu, rho, rplus, rminus, K)

            dominance_turnover[sim_nbr-1] = mfpt_a2a(ss_dist_sim, nbr, mu, rplus, rminus, K, rho, S)
            suppress_turnover[sim_nbr-1]  = mfpt_020(ss_dist_sim, mu)
            dominance_return[sim_nbr-1]   = mfpt_a2b(ss_dist_sim, 0, nbr, mu, rplus)
            suppress_return[sim_nbr-1]    = mfpt_b2a(ss_dist_sim, 0, nbr, mu, rplus)

            ############################################################################################################################
            # TEMP AUTOCORRELATION TIME SAVING. This is an aweful way of doing it. Fix it.
            ############################################################################################################################

            end = time.time()
            hours, rem = divmod( end-start, 3600 )
            minutes, seconds = divmod( rem, 60 )
            print(">>{}Time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(i,int(hours)
                                                        , int(minutes),seconds))
            # Value of parameters
            param1[i] = param_dict[parameter1]
            param2[i] = param_dict[parameter2]

    # making all sims have same distribution length
    len_longest_sim     = len( max(sim_dist_vary,key=len) )
    length_longest_rich = len( max(rich_dist_vary,key=len) )
    sim_dist            = np.zeros( ( nbr_sims,len_longest_sim ) )
    conv_dist           = np.zeros( ( nbr_sims,len_longest_sim ) )
    mf_dist             = np.zeros( ( nbr_sims,len_longest_sim ) )
    rich_dist           = np.zeros( ( nbr_sims,length_longest_rich ) )

    for i in np.arange(nbr_sims):
        #conv_idx    = np.min( [ len(conv_dist_vary[i]), len_longest_sim ] )
        mf_idx      = np.min( [ len(mf_dist_vary[i]), len_longest_sim ] )
        sim_dist[i,:len(sim_dist_vary[i])]      = sim_dist_vary[i]
        #conv_dist[i,:conv_idx]                  = conv_dist_vary[i][conv_idx]
        mf_dist[i,:mf_idx]                      = mf_dist_vary[i][:mf_idx]
        rich_dist[i,:len(rich_dist_vary[i])]    = rich_dist_vary[i]

    # For heatmap stuff
    param1_2D = np.unique(param1); param2_2D = np.unique(param2)
    dim_1     = len(param1_2D)   ; dim_2      = len(param2_2D)

    # initialize
    mf_dist2D           = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    conv_dist2D         = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    sim_dist2D          = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    rich_dist2D         = np.zeros( ( dim_1,dim_2,length_longest_rich ) )
    corr_ni_nj2D        = np.zeros( ( dim_1,dim_2 ) )
    coeff_ni_nj2D       = np.zeros( ( dim_1,dim_2 ) )
    corr_J_n2D          = np.zeros( ( dim_1,dim_2 ) )
    coeff_J_n2D         = np.zeros( ( dim_1,dim_2 ) )
    corr_Jminusn_n2D    = np.zeros( ( dim_1,dim_2 ) )
    coeff_Jminusn_n2D   = np.zeros( ( dim_1,dim_2 ) )

    time_autocor_spec2D       = np.zeros( ( dim_1,dim_2 ) )
    mean_time_autocor_abund2D = np.zeros( ( dim_1,dim_2 ) )
    std_time_autocor_abund2D  = np.zeros( ( dim_1,dim_2 ) )
    dominance_turnover2D      = np.zeros( ( dim_1,dim_2 ) )
    suppress_turnover2D       = np.zeros( ( dim_1,dim_2 ) )
    dominance_return2D        = np.zeros( ( dim_1,dim_2 ) )
    suppress_return2D         = np.zeros( ( dim_1,dim_2 ) )

    # put into a 2d array all the previous results
    for sim in np.arange(nbr_sims):
        i                       = np.where( param1_2D==param1[sim] )[0][0]
        j                       = np.where( param2_2D==param2[sim] )[0][0]
        sim_dist2D[i,j]         = sim_dist[sim]
        mf_dist2D[i,j]          = mf_dist[sim]
        #conv_dist2D[i,j]       = conv_dist[sim]
        rich_dist2D[i,j]        = rich_dist[sim]
        corr_ni_nj2D[i,j]       = corr_ni_nj[sim]
        coeff_ni_nj2D[i,j]      = coeff_ni_nj[sim]
        corr_J_n2D[i,j]         = corr_J_n[sim]
        coeff_J_n2D[i,j]        = coeff_J_n[sim]
        corr_Jminusn_n2D[i,j]   = corr_Jminusn_n[sim]
        coeff_Jminusn_n2D[i,j]  = coeff_Jminusn_n[sim]

        #----------------------------------------------------------------------------------------------------------------------------
        # TEMP AUTOCORRELATION TIME SAVING. This is an aweful way of doing it. Fix it.
        #----------------------------------------------------------------------------------------------------------------------------

        time_autocor_spec2D[i,j]        = time_autocor_spec[sim]
        mean_time_autocor_abund2D[i,j]  = mean_time_autocor_abund[sim]
        std_time_autocor_abund2D[i,j]   = std_time_autocor_abund[sim]
        dominance_turnover2D[i,j]       = dominance_turnover[sim]
        suppress_turnover2D[i,j]        = suppress_turnover[sim]
        dominance_return2D[i,j]         = dominance_return[sim]
        suppress_return2D[i,j]          = suppress_return[sim]

        #----------------------------------------------------------------------------------------------------------------------------
        # TEMP AUTOCORRELATION TIME SAVING. This is an aweful way of doing it. Fix it.
        #----------------------------------------------------------------------------------------------------------------------------

    # arrange into a dictionary to save
    dict_arrays = { 'sim_dist'  : sim_dist2D, 'mf_dist'     : mf_dist2D
                                        , 'conv_dist'       : conv_dist2D
                                        , 'rich_dist'       : rich_dist2D
                                        , 'rich_dist2D'     : rich_dist
                                        , 'corr_ni_nj2D'    : corr_ni_nj2D
                                        , 'coeff_ni_nj2D'   : coeff_ni_nj2D
                                        , 'corr_J_n2D'      : corr_J_n2D
                                        , 'coeff_J_n2D'     : coeff_J_n2D
                                        , 'corr_Jminusn_n2D' : corr_Jminusn_n2D
                                        , 'coeff_Jminusn_n2D': coeff_Jminusn_n2D
                                        , 'carry_capacity': model.carry_capacity
                                        , 'birth_rate'      : model.birth_rate
                                        , 'death_rate'      : model.death_rate
                                        , 'nbr_species'     : model.nbr_species
                                        , 'immi_rate'       : model.immi_rate
                                        , 'comp_overlap'    : model.comp_overlap
                                        , 'time_autocor_spec2D'       : time_autocor_spec2D
                                        , 'mean_time_autocor_abund2D' : mean_time_autocor_abund2D
                                        , 'std_time_autocor_abund2D'  : std_time_autocor_abund2D
                                        , 'dominance_turnover2D'      : dominance_turnover2D
                                        , 'suppress_turnover2D'       : suppress_turnover2D
                                        , 'dominance_return2D'        : dominance_return2D
                                        , 'suppress_return2D'         : suppress_return2D
                                        }
    dict_arrays[parameter1] = param1_2D
    dict_arrays[parameter2] = param2_2D
    # save results in a npz file
    np.savez(filename, **dict_arrays)

    return filename, dict_arrays

def mlv_multiple_folder_consolidate(list_dir, consol_name_dir, parameter1=None
                                            , parameter2=None, consol = False):
    """
    Takes simulations from multiple folders and combines them by averaging their
    results all together. The only reason I do this is to average the
    distributions... no need to average other quantities

    Input
        list_dir        : list of directories to average
        consol_name_dir : What to name the directory of averaged folders
        parameter1      : 1st paramater to vary (generally )
        parameter2      : 2nd parameter to vary
        consol          : Whether we want to use previously calculated short.npz
    """
    dict_arr = []
    for dir in list_dir:
        # Get dictionary of results from each of these sets of simualtions
        if not consol:
            _, dict_temp = mlv_consolidate_sim_results(dir, parameter1
                                                        , parameter2)
        else:
            dict_temp = dict( np.load(dir + os.sep + NPZ_SHORT_FILE) )
            dict_temp['carry_capacity'] = 100.0; dict_temp['birth_rate'] = 2.0;
            dict_temp['death_rate'] = 1.0; dict_temp['nbr_species'] = 30
        dict_arr.append(dict_temp)
        del dict_temp

    df = pd.DataFrame(dict_arr); mean_dict = {}
    for column in df:
        # average of each of these simulations. Might be a problem if they're
        # different length arrays (n may vary greatly)!
        mean_dict[column] = df[column].mean()

    while not os.path.exists( consol_name_dir ):
        # make a new directory that is the combination of these different files
        os.makedirs( consol_name_dir );

    filename = consol_name_dir + os.sep + NPZ_SHORT_FILE

    # save results in a npz file
    np.savez(filename, **mean_dict)

    return 0

def determine_modality( arr, plot=True, revisionmanual=None, sampleP0 = True ):
    """
    This is a fine art, which I do not truly believe in

    Input
        arr     : 3D array of distributions
        plot    : Whether to plot the results from the array
        multimodmanual : Whether to override the multimodal check manually, use
                        number of simulation (defined in manual_multimodality)
    """

    line_colours = ['khaki','slateblue','mediumturquoise','lightcoral'\
                            ,'lightgray']
    line_names   = ['unimodal\n peak at 0','unimodal\n peak at >0'\
                        ,'bimodal', 'multimodal', r'$P(0)$ unsampled']
    bounds       = [-0.5, 0.5, 1.5, 2.5, 3.5];

    modality_arr = 2*np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )

    for i in np.arange(np.shape(arr)[0]):
        for j in np.arange(np.shape(arr)[1]):
            #smooth_arr = smooth(arr[i,j,:],21)
            smooth_arr = smooth(arr[i,j,:65],4) # 71, 78
            #smooth_arr = smooth(arr[i,j,:40],4)
            #smooth_arr = smooth(arr[i,j,:100],15) # 79 inset
            max_idx, _ = find_peaks( smooth_arr )

            if arr[i,j,0] > arr[i,j,1]:
                modality_arr[i,j] = line_names.index('bimodal')
                if len(max_idx) == 1 and max_idx[0]<3:
                    modality_arr[i,j] = line_names.index('unimodal\n peak at 0')
                elif len(max_idx) > 2:
                    if revisionmanual == None:
                        pass
                    else:
                        modality_arr[i,j] = line_names.index('multimodal')
            else:
                modality_arr[i,j] = line_names.index('unimodal\n peak at >0')

            if arr[i,j,0] == 0.0:
                if sampleP0 == True:
                    modality_arr[i,j] = line_names.index(r'$P(0)$ unsampled')

    if revisionmanual != None:
        for keys_revision in DICT_REVISION[revisionmanual]:
            for idx_revision in DICT_REVISION[revisionmanual][keys_revision]:
                i = idx_revision[0]; j=idx_revision[1]
                modality_arr[i,j] = line_names.index('multimodal')
    if plot:
        for i in np.arange(np.shape(arr)[0]):
            for j in np.arange(np.shape(arr)[1]):
                plot_prob(arr[i,j,:],i, j, line_colours[int(modality_arr[i,j])])
    return modality_arr, line_names, line_colours

def determine_bimodality_mf( arr ):
    modality_arr = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    for i in np.arange(np.shape(arr)[0]):
        for j in np.arange(np.shape(arr)[1]):
            max_idx, _ = find_peaks( arr[i,j] )
            if max_idx.tolist() != []:
                modality_arr[i,j] = 1.
    return modality_arr

def plot_prob(probability, i, j, colour):
    DIST_DIR = MANU_FIG_DIR + os.sep + 'distributions'
    os.makedirs(DIST_DIR, exist_ok=True)
    f = plt.figure(figsize=(1.75,1.25)); fig = plt.gcf(); ax = plt.gca()
    plt.plot(probability[:125], color=colour, linewidth=2, zorder=10)
    plt.xticks([0,50,100])
    size = 7; ax.axes.labelsize = size
    plt.xlabel(r'n',fontsize=size, labelpad=2);
    plt.ylabel(r'P(n)', fontsize=size, labelpad=2);
    plt.xlim([0,125]); plt.yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=size)
    #ax.tick_params(axis='both', which='minor', labelsize=6)

    if save:
        plt.savefig(DIST_DIR + os.sep + "dstbn_" + str(i) + '_' + str(j)+'.pdf')
    else:
        plt.show()
    plt.close()

    return 0

def set_axis(ax, plt, POINTS_BETWEEN_X_TICKS, POINTS_BETWEEN_Y_TICKS, rangex
                , rangey, xlog, ylog, xdatalim, ydatalim, xlabel, ylabel):
    ax.set_xticks([i for i, xval in enumerate(rangex)
                        if i % POINTS_BETWEEN_X_TICKS == 0])
    ax.set_yticks([i for i, yval in enumerate(rangey)
                        if i % POINTS_BETWEEN_Y_TICKS == 0])
    if xlog:
        ax.set_xticklabels([r'$10^{%d}$' % np.log10(round(xval, 13))
                            for i, xval in enumerate(rangex)
                            if (i % POINTS_BETWEEN_X_TICKS==0)])
    else:
        ax.set_xticklabels([r'$%.0f$' % round(xval, 2)
                            for i, xval in enumerate(rangex)
                            if (i % POINTS_BETWEEN_X_TICKS==0)])
    if ylog:
        ax.set_yticklabels([r'$10^{%d}$' % np.log10(round(yval, 13))
                            for i, yval in enumerate(rangey)
                            if i % POINTS_BETWEEN_Y_TICKS==0])
    else:
        ax.set_yticklabels([r'$%.2f$' % round(yval, 2)
                            for i, yval in enumerate(rangey)
                            if i % POINTS_BETWEEN_Y_TICKS==0])
    ax.invert_yaxis();
    if ydatalim != None:
        plt.ylim([ydatalim[0],ydatalim[1]])
    if xdatalim != None:
        plt.xlim([xdatalim[0],xdatalim[1]])

    plt.xlabel(VAR_NAME_DICT[xlabel]); plt.ylabel(VAR_NAME_DICT[ylabel])

    return ax, plt

def fig2A(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    """
    Heatmap of richness, with
    """
    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')

    # species simulations didn't all work, erase the 0s
    start = 0
    if xlabel=='nbr_species':
        start = 1

    # transform
    rangex = data[xlabel][start:]; rangey = data[ylabel][:]

    # simulation/mf results
    sim_rich = 1 - data['sim_dist'][start:,:,0]
    mf_rich  = 1 - data['mf_dist'][start:,:,0]

    # if nbr_species varies in simulation
    arr = np.ones( ( np.shape(sim_rich)[0], np.shape(sim_rich)[1] ) )
    if xlabel=='nbr_species':
        sim_nbr_spec = data['nbr_species'][start:,None]*sim_rich
        S =  data['nbr_species'][start:,None]*arr
        sim_frac_spec_minus1 = (data['nbr_species'][start:,None]-1.)/(data['nbr_species'][start:,None])*arr
        mf_nbr_spec = data['nbr_species'][start:,None]*mf_rich
    elif ylabel=='nbr_species':
        sim_nbr_spec = data['nbr_species'][start:,None]*sim_rich
        S =  data['nbr_species'][start:,None]*arr
        sim_frac_spec_minus1 = (data['nbr_species']-1.)/(data['nbr_species'])[start:,None]*arr
        mf_nbr_spec = S*mf_rich
    else:
        sim_nbr_spec = data['nbr_species']*sim_rich
        mf_nbr_spec = data['nbr_species']*mf_rich
        sim_frac_spec_minus1 = (data['nbr_species']-1.)/data['nbr_species']*arr

    # _cat signifies categories (one of the richnesses)
    sim_rich_cat = np.zeros( ( np.shape(sim_rich)[0], np.shape(sim_rich)[1] ) )
    mf_rich_cat = np.zeros( ( np.shape(mf_rich)[0], np.shape(mf_rich)[1] ) )

    # richness determination
    for i in np.arange(np.shape(sim_rich_cat)[0]):
        for j in np.arange(np.shape(sim_rich_cat)[1]):
            if sim_rich[i,j] >= sim_frac_spec_minus1[i,j]:
                sim_rich_cat[i,j] = 1.0
            elif sim_nbr_spec[i,j] < 1.5:
                sim_rich_cat[i,j] = -1.0
            else: pass

            if mf_rich[i,j] >= sim_frac_spec_minus1[i,j]:
                mf_rich_cat[i,j] = 1.0
            elif mf_nbr_spec[i,j] < 2:
                mf_rich_cat[i,j] = -1.0
            else: pass

    # plots
    f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()

    # boundaries and colours
    lines = [-1,0,1]; bounds = [-1.5, -0.5, 0.5, 1.5];
    line_colours = ['royalblue','cornflowerblue','lightsteelblue']
    cmap = colors.ListedColormap( line_colours )
    norm = colors.BoundaryNorm( bounds, cmap.N )

    im = plt.imshow(sim_rich_cat.T, cmap=cmap, norm=norm, aspect='auto')

    # contours
    mf_rich_cat = scipy.ndimage.filters.gaussian_filter(mf_rich_cat,0.7) #smooth
    if start == 0:
        MF = ax.contour( mf_rich_cat.T, [-0.5], linestyles='solid', colors = 'k'
                                    , linewidths = 2)
    MF2 = ax.contour( mf_rich_cat.T, [0.5], linestyles='solid', colors = 'k'
                                , linewidths = 2)

    set_axis(ax, plt, POINTS_BETWEEN_X_TICKS, POINTS_BETWEEN_Y_TICKS, rangex
                    , rangey, xlog, ylog, xdatalim, ydatalim, xlabel, ylabel)

    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "richness_"+ xlabel + '.pdf');
        #plt.savefig(MANU_FIG_DIR + os.sep + "richness" + '.png');
    else:
        plt.show()
    plt.close()

    # 2D heatmap of species present
    if xlabel != 'nbr_species':
        f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()
        R=np.shape(data['rich_dist'])[2]-1
        plt.plot(data['rich_dist'][0,R], color=line_colours[0]
                    ,marker='s', markeredgecolor='None', linewidth=2, zorder=10)
        plt.plot(data['rich_dist'][int(5*(R)/10), int(8*(R)/10)]
                    , color=line_colours[1], marker='o', markeredgecolor='None'
                    ,linewidth=2, zorder=10)
        plt.plot(data['rich_dist'][R,0], color=line_colours[2]
                    , marker='D', markeredgecolor='None',linewidth=2, zorder=10)
        plt.ylim([0,1.0])
        plt.xlim([0,np.shape(data['rich_dist'])[2]-1])
        plt.xlabel(r'species present, $S^*$')
        plt.ylabel(r'P($S^*$)')

        if save:
            plt.savefig(MANU_FIG_DIR + os.sep + "richness_dist_"+xlabel+'.pdf');
            #plt.savefig(MANU_FIG_DIR + os.sep + "richness" + '.png');
        else:
            plt.show()
        plt.close()

    return sim_rich_cat, mf_rich_cat, lines

def compare_richness(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    """
    compare sim richness with the richness from the rates found in Appendix I
    """
    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')

    # transform
    rangex = data[xlabel][:]; rangey = data[ylabel][:]

    # simulation/mf results
    K = data['carry_capacity']; rminus = data['death_rate'];
    rplus = data['birth_rate'];
    mu = data['immi_rate']; S = data['nbr_species']
    sim_rich = 1 - data['sim_dist'][:,:,0]
    rates_rich = np.zeros( ( np.shape(sim_rich)[0], np.shape(sim_rich)[1] ) )
    for i in np.arange( np.shape(sim_rich)[0] ):
        for j in np.arange(  np.shape(sim_rich)[1] ):
            dstbn = data['sim_dist'][i,j]
            rates_rich[i,j] = richness_from_rates( dstbn, rplus, rminus, K
                                                    , rangey[i], rangex[j], S )

    # Fig SImualtion richness
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(( 0,0,0 ))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None}
    # heatmap
    im = plt.imshow( sim_rich.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle S^* \rangle$ simulation')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "richness_simulation" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "richness_simulation" + '.png');
    else:
        plt.show()
    plt.close()

    # Fig rrates richness
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(( 0,0,0 ))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None}
    # heatmap
    im = plt.imshow( sim_rich.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle S^* \rangle$ simulation')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "richness_from_rates" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "richness_from_rates" + '.png');
    else:
        plt.show()
    plt.close()


    return 0


def fig2B(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    """
    Heatmap of modality
    Need to smooth out simulation results. How to compare?
    """
    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')

    # exclude first row in nbr_species because certain of those sims failed
    start = 0
    if xlabel=='nbr_species':
        start = 1

    # setting simulation parameters
    rangex = data[xlabel][start:]; rangey = data[ylabel][:]
    K = data['carry_capacity']; rminus = data['death_rate'];
    rplus = data['birth_rate'];
    mu = data['immi_rate']; S = data['nbr_species']

    # simulation results
    sim_dist = data['sim_dist'][start:,:,:]
    mf_dist = data['mf_dist'][start:,:,:]
    rich_dist = data['rich_dist'][start:,:,:]

    # calculating modality
    modality_sim, line_names, line_colours\
                    = determine_modality( sim_dist, distplots, revision, False )
    modality_mf, _, _  = determine_modality( mf_dist, False, sampleP0 = False )

    # boundaries of modality
    lines = [float(i) for i in list( range( 0, len(line_names) ) ) ]
    bounds = [ i - 0.5 for i in lines + [lines[-1] + 1.0]  ]
    lines_center = [ a + b for a, b in zip( lines, [0]*len(line_names) ) ]

    # calculating mean J
    mf_meanJ = meanJ_est(mf_dist, (np.shape(rich_dist)[2]-1))
    mf_rich = np.shape(mf_dist)[2] * ( 1 - mf_dist )
    mf_meanJ = meanJ_est(sim_dist, (np.shape(rich_dist)[2]-1))

    if xlabel == 'immi_rate':
        mu  = (rangex*np.ones( (np.shape(sim_dist)[0]
                                , np.shape(sim_dist)[1])).T).T
    # 2D rho-mu arrays
    rho = (rangey*np.ones( (np.shape(sim_dist)[0]
                                , np.shape(sim_dist)[1])))

    mf_unimodal = (1./mu)*(rminus + (rplus-rminus)*( 1. +
                                rho*( mf_meanJ - 1. ) )/K )

    f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()

    # plots
    cmap = colors.ListedColormap( line_colours )
    norm = colors.BoundaryNorm( bounds, cmap.N )

    im = plt.imshow(modality_sim.T, cmap=cmap, norm=norm, aspect='auto')
    modality_mf = scipy.ndimage.filters.gaussian_filter(modality_mf, 0.7)
    mf_unimodal = scipy.ndimage.filters.gaussian_filter(mf_unimodal, 0.7)
    MF = ax.contour( modality_mf.T, [0.5], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    MF2 = ax.contour( mf_unimodal.T, [1.], linestyles='solid'
                        , colors = 'k', linewidths = 2)

    set_axis(ax, plt, POINTS_BETWEEN_X_TICKS, POINTS_BETWEEN_Y_TICKS, rangex
                    , rangey, xlog, ylog, xdatalim, ydatalim, xlabel, ylabel)
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "modality_" + xlabel + '.pdf');
        #plt.savefig(MANU_FIG_DIR + os.sep + "modality" + '.png');
        print(MANU_FIG_DIR+ os.sep + "modality_" + xlabel + '.pdf')
    else:
        plt.show()

    return modality_sim, modality_mf, mf_unimodal, lines, line_colours

def fig2(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    """
    Heatmap of modality
    Need to smooth out simulation results. How to compare?
    """
    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    sim_rich_cat, mf_rich_cat, lines_rich =\
        fig2A(filename, save, xlabel, ylabel, xlog, ylog, ydatalim, xdatalim
                            , None, distplots, pbx, pby)
    modality_sim, modality_mf, mf_unimodal, lines_mod, colours_mod =\
    fig2B(filename, save, xlabel, ylabel, xlog, ylog, ydatalim, xdatalim
                        , revision, distplots, pbx, pby)

    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')

    start = 0
    #certain species sims did not end correctly. Set them to 0, need to ignpre
    if xlabel=='nbr_species':
        start = 1

    rangex = data[xlabel][start:]; rangey = data[ylabel][:]

    sep = 10
    combined_sim = modality_sim + sep*sim_rich_cat; #defining all categories

    K = data['carry_capacity']; rminus = data['death_rate']; rplus = data['birth_rate'];
    mu = data['immi_rate']; S = data['nbr_species']
    lines = []
    for j in lines_rich:
        lines.extend([ float(i)+sep*j for i in list( range( 0, len(lines_mod) ) ) ])

    bounds = [ i - 0.5 for i in lines + [lines[-1] + 1.0]  ]
    lines_center = [ a + b for a, b in zip( lines, [0]*len(lines) ) ]

    # plots
    f = plt.figure(figsize=(4,3)); fig = plt.gcf(); ax = plt.gca()
    #f = plt.figure(figsize=(2,2)); fig = plt.gcf(); ax = plt.gca()
    line_colours = ['darkgoldenrod', 'darkslateblue', 'blue' , 'indianred', 'slategrey']
    line_colours.extend(colours_mod)
    line_colours.extend(['lemonchiffon', 'mediumpurple', 'paleturquoise', 'mistyrose', 'whitesmoke'])

    # plots
    cmap = colors.ListedColormap( line_colours )
    norm = colors.BoundaryNorm( bounds, cmap.N )

    im = plt.imshow(combined_sim.T, cmap=cmap, norm=norm, aspect='auto')

    MF = ax.contour( modality_mf.T, [0.5], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    MF2 = ax.contour( mf_unimodal.T, [1.], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    if start == 0:
        MF3 = ax.contour( mf_rich_cat.T, [-0.5], linestyles='solid', colors = 'k'
                                    , linewidths = 2)
    MF4 = ax.contour( mf_rich_cat.T, [0.5], linestyles='solid', colors = 'k'
                                , linewidths = 2)

    set_axis(ax, plt, POINTS_BETWEEN_X_TICKS, POINTS_BETWEEN_Y_TICKS, rangex
                    , rangey, xlog, ylog, xdatalim, ydatalim, xlabel, ylabel)

    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "combined_" + xlabel + '.pdf');
        #plt.savefig(MANU_FIG_DIR + os.sep + "modality" + '.png');
    else:
        plt.show()

    return 0

def fig3A(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    """
    Turnover rate
    """
    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')

    # parameter range
    rangex = data[xlabel]; rangey = data[ylabel]
    K = data['carry_capacity']; rminus = data['death_rate'];
    rplus = data['birth_rate']; S = data['nbr_species']

    # 2D mu-rho
    mu  = (rangex*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])).T).T

    rho = (rangey*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])))

    # calculating turnover rate
    nbr = 0
    dist = 'sim_dist' # sim or mf
    meanJ = meanJ_est(data[dist], (np.shape(data['rich_dist'])[2]-1))
    turnover_nava = 1. / (data[dist][:,:,nbr]*mu)
    turnover_calc = ( 1. + data[dist][:,:,nbr] ) / ( data[dist][:,:,nbr]
                        * ( rplus*nbr + mu + nbr*( rminus + ( rplus-rminus )
                        * ( ( 1. - rho ) * nbr + rho * meanJ )/K  ) ) )
    turnover = turnover_nava

    # plots
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('cividis_r')) # copy the default cmap
    my_cmap.set_bad( color_bad )
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    im = plt.imshow( turnover.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)

    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])

    plt.title(r'$\langle T(0\rightarrow 0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "turnover" + '.png');
    else:
        plt.show()

    return 0

def figTaylor(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'):
    """
    Taylors Law
    """
    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')
    rangex = data[xlabel]; rangey = data[ylabel]
    K = data['carry_capacity']; rminus = data['death_rate']; rplus = data['birth_rate'];
    S = data['nbr_species']

    dist = 'mf_dist'
    start = 1
    prob = data[dist][:,:,start:]\
            /np.sum(data[dist][:,:,start:],axis=2)[:,:,np.newaxis]

    square      = np.tensordot( prob,np.arange(start,np.shape(data[dist])[2])**2
                                , axes=(2,0) )
    square_flat = square.flatten()
    mean        = np.tensordot( prob, np.arange(start,np.shape(data[dist])[2])
                                    , axes=(2,0) )
    mean_flat   = mean.flatten()
    variance    = square - mean**2
    variance_flat = square_flat - mean_flat**2

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    """
    plt.scatter(mean,variance,linewidths=0.01)
    plt.xscale('log');plt.yscale('log')
    plt.xlabel(r'$\langle n \rangle$'); plt.ylabel(r'$var(n)$')
    plt.title(r"Taylor's Law")
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "Taylorlaw" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "Taylorlaw" + '.png');
    else:
        plt.show()
    """

    # plot TL scaled
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    cmap = mpl.cm.get_cmap("plasma")
    for i in np.arange(np.shape(mean)[0]):
        for j in np.arange(np.shape(mean)[1]):
            plt.scatter(mean[i,j],variance[i,j], s=(float(j)/float(np.shape(mean)[1]))
            , marker='+', edgecolors=None, c=cmap(i/np.shape(mean)[0]) )
            #plt.scatter(mean[i,j],variance[i,j], s=(float(j)/float(np.shape(mean)[1]))
            #, marker='o', edgecolors=None, c=cmap(i/np.shape(mean)[0]) )
    plt.xscale('log');plt.yscale('log')
    plt.xlabel(r'$\langle n \rangle$'); plt.ylabel(r'$var(n)$')
    plt.title(r"Taylor's Law")
    plt.legend([r"colour : $rho$",r"size : $mu$"])
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "Taylorlaw_SC" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "Taylorlaw_SC" + '.png');
    else:

        plt.show()

def fig3(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    """
    Time to extinction, time to dominance
    """
    # loading
    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')
    K = data['carry_capacity']; rminus = data['death_rate']; rplus = data['birth_rate'];
    S = data['nbr_species']

    start = 0
    if xlabel=='nbr_species':
        start = 1

    # setting simulation parameters
    rangex = data[xlabel][start:]; rangey = data[ylabel][:]
    K = data['carry_capacity']; rminus = data['death_rate'];
    rplus = data['birth_rate'];
    mu = data['immi_rate']; S = data['nbr_species']

    # simulation results
    sim_dist = data['sim_dist'][start:,:,:]
    mf_dist = data['mf_dist'][start:,:,:]
    rich_dist = data['rich_dist'][start:,:,:]

    # 2D rho-mu
    mu  = (rangex*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])).T).T

    rho = (rangey*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])))

    arr             = data['sim_dist']
    meanJ           = meanJ_est(arr, (np.shape(data['rich_dist'])[2]-1))
    max_arr         = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    dom_turnover    = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    sub_turnover    = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    fpt_dominance   = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    fpt_submission  = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    nbr_species_arr = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    for i in np.arange(np.shape(arr)[0]):
        for j in np.arange(np.shape(arr)[1]):
            nbr_species_arr[i,j] = S*(1.0-arr[i,j,0])#np.dot(data['rich_dist'][i,j]
                            #    , np.arange(len(data['rich_dist'][i,j])) )
            nbr = deterministic_mean(nbr_species_arr[i,j], mu[i,j],rho[i,j], rplus
                                                , rminus, K)
            #max_arr[i,j] = nbr
            max_arr[i,j] = 1 + np.argmax( arr[i,j,1:] )
            sub_turnover[i,j]   = mfpt_020( arr[i,j], mu[i,j] )
            dom_turnover[i,j]   = mfpt_a2a( arr[i,j], nbr, mu[i,j], rplus, rminus
                                        , K, rho[i,j], S )
            fpt_submission[i,j] = mfpt_b2a( arr[i,j], 0, nbr, mu[i,j], rplus)
            fpt_dominance[i,j]  = mfpt_a2b( arr[i,j], 0, nbr, mu[i,j], rplus)

    fpt_cycling    = fpt_dominance + fpt_submission
    ratio_turnover = sub_turnover / dom_turnover
    ratio_dominance_loss = fpt_submission / dom_turnover
    ratio_suppression_loss = fpt_dominance / sub_turnover
    ratio_switch   = fpt_dominance / fpt_submission
    weighted_timescale = ( ( S - nbr_species_arr) * sub_turnover
                                + nbr_species_arr * dom_turnover  ) / S

    color_bad = (211/256,211/256,211/256)
    # Fig3C
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( dom_turnover.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_dom_turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_dom_turnover" + '.png');
    else:
        plt.show()
    plt.close()

    # FIG
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( sub_turnover.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(0\rightarrow 0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_sub_turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_sub_turnover" + '.png');
    else:
        plt.show()
    plt.close()

    ## FIGURE 3D
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( fpt_dominance.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    # title
    plt.title(r'$\langle T(0\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_dominance" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_dominance" + '.png');
    else:
        plt.show()
    plt.close()

    ## FIGURE 3E
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( fpt_submission.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(\tilde{n}\rightarrow 0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_supression" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_supression" + '.png');
    else:
        plt.show()
    plt.close()

    ## FIGURE 3F
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( fpt_cycling.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(\tilde{n}\rightarrow 0) \rangle+\langle T(0\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_cycling" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_cycling" + '.png');
    else:
        plt.show()
    plt.close()

    ## RATIO turnover
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('plasma_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( ratio_turnover.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(0\rightarrow 0) \rangle / \langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_turnover" + '.png');
    else:
        plt.show()
    plt.close()

    ## Ratio go to
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('plasma_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( ratio_switch.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(\tilde{n}\rightarrow 0) \rangle / \langle T(0\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_switch" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_switch" + '.png');
    else:
        plt.show()
    plt.close()

    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    sim_rich_cat, mf_rich_cat, lines_rich =\
        fig2A(filename, False, xlabel, ylabel, xlog, ylog, ydatalim, xdatalim
                            , None, distplots, pbx, pby)
    modality_sim, modality_mf, mf_unimodal, lines_mod, colours_mod =\
    fig2B(filename, False, xlabel, ylabel, xlog, ylog, ydatalim, xdatalim
                        , revision, distplots, pbx, pby)

    ## Ratio dominance loss
    f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('plasma_r')) # copy the default cmap

    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    font_label_contour = 8
    im = plt.imshow( ratio_dominance_loss.T, **imshow_kw)
    #ax1 = ax.contour( ratio_dominance_loss.T, [10.0, 100.0, 10**5], linestyles=['dotted','dashed','solid']
    boundary = 100.0; boundary_colour = 'gray'

    MF = ax.contour( modality_mf.T, [0.5], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    MF2 = ax.contour( mf_unimodal.T, [1.], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    if start == 0:
        MF3 = ax.contour( mf_rich_cat.T, [-0.5], linestyles='solid', colors = 'k'
                                    , linewidths = 2)
    MF4 = ax.contour( mf_rich_cat.T, [0.5], linestyles='solid', colors = 'k'
                                , linewidths = 2)

    ax1 = ax.contour( ratio_dominance_loss.T, [boundary], linestyles=['dashed']
                        , colors = boundary_colour, linewidths = 1)

    #ax.clabel(ax1, inline=True, fmt=ticker.LogFormatterMathtext(), fontsize=font_label_contour)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    #plt.title(r'$\langle T(\tilde{n}\rightarrow 0) \rangle / \langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_dominance_loss" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_dominance_loss" + '.png');
    else:
        plt.show()
    plt.close()

    colours_line = ['r','g','b']; plots = [40,20,0];
    markerline = ['o-','x-','D-']
    f = plt.figure(figsize=(3.25,2.5))
    fig = plt.gcf(); ax = plt.gca()
    # lineplot
    font_label_contour = 8
    for i, val in enumerate(plots):
        array = np.array( ratio_dominance_loss.T[:,val] )
        array[array == np.inf] = np.nan
        plt.plot(rangey[ydatalim[0]:ydatalim[1]+1], array[ydatalim[0]:ydatalim[1]+1]
                    , markerline[i], c=colours_line[i]
                    , label=VAR_SYM_DICT[xlabel] + ": " + r'$10^{%d}$' %
                    np.log10(round(rangex[val], 13) ) )
    plt.axhline(y=boundary, color=boundary_colour, linestyle='--')
    plt.xlim((rangey[ydatalim[0]],rangey[ydatalim[1]]))
    # legend
    plt.legend(loc='best')
    plt.xscale('log'); plt.yscale('log')
    plt.ylabel(r'$\langle T(\tilde{n}\rightarrow 0) \rangle / \langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')
    plt.xlabel(VAR_NAME_DICT[ylabel])
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_dominance_loss_lineplot" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_dominance_loss_lineplot" + '.png');
    else:
        plt.show()
    plt.close()

    ## Ratio supression loss
    f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('plasma_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    boundary = 1.0
    im = plt.imshow( ratio_suppression_loss.T, **imshow_kw)

    MF = ax.contour( modality_mf.T, [0.5], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    MF2 = ax.contour( mf_unimodal.T, [1.], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    if start == 0:
        MF3 = ax.contour( mf_rich_cat.T, [-0.5], linestyles='solid', colors = 'k'
                                    , linewidths = 2)
    MF4 = ax.contour( mf_rich_cat.T, [0.5], linestyles='solid', colors = 'k'
                                , linewidths = 2)

    ax1 = ax.contour( ratio_suppression_loss.T, [boundary], linestyles=['dashed']
                        , colors = boundary_colour, linewidths = 1)
    #ax.clabel(ax1, inline=True, fmt=ticker.LogFormatterMathtext(), fontsize=font_label_contour)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    #plt.title(r'$\langle T(0\rightarrow \tilde{n}) \rangle / \langle T(0\rightarrow 0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_suppression_loss" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_suppression_loss" + '.png');
    else:
        plt.show()
    plt.close()

    f = plt.figure(figsize=(3.25,2.5))
    fig = plt.gcf(); ax = plt.gca()
    # lineplot
    font_label_contour = 8
    for i, val in enumerate(plots):
        array = np.array( ratio_suppression_loss.T[:,val] )
        array[array == np.inf] = np.nan
        plt.plot(rangey[ydatalim[0]:ydatalim[1]+1], array[ydatalim[0]:ydatalim[1]+1]
                    , markerline[i], c=colours_line[i]
                    , label=VAR_SYM_DICT[xlabel] + ": " + r'$10^{%d}$' %
                    np.log10(round(rangex[val], 13) ) )
    plt.axhline(y=boundary, color=boundary_colour, linestyle='--')
    plt.xlim((rangey[ydatalim[0]],rangey[ydatalim[1]]))
    # legend
    plt.legend(loc='best')
    plt.xscale('log'); plt.yscale('log')
    plt.ylabel(r'$\langle T(0\rightarrow \tilde{n}) \rangle / \langle T(0\rightarrow 0) \rangle$')
    plt.xlabel(VAR_NAME_DICT[ylabel])
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_suppression_loss_lineplot" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_suppression_loss_lineplot" + '.png');
    else:
        plt.show()
    plt.close()

    ## Weighted timescales
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( weighted_timescale.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\frac{S-\langle S^* \rangle }{S}\langle T(0\rightarrow 0) \rangle + \frac{\langle S^* \rangle }{S}\langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_weighted_timescale" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_weighted_timescale" + '.png');
    else:
        plt.show()
    plt.close()

    return 0

def fig3F(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'):
    """
    Something to do with the deterministic mean?
    """

    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')
    rangex = data[xlabel]; rangey = data[ylabel]
    K = data['carry_capacity']; rminus = data['death_rate']; rplus = data['birth_rate'];
    S = data['nbr_species']
    mu  = (rangex*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])).T).T

    rho = (rangey*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])))

    nbr_species = S # Or something different?
    det_mean = K*( ( 1. + np.sqrt( 1.+ 4.*mu*( 1. + rho*( nbr_species - 1. ) ) /
                (K*(rplus-rminus)) ) ) / ( 2.*( 1. + rho*(nbr_species-1.) ) ) )

def many_parameters_dist(filename, save=False, range='comp_overlap', fixed = 2
                                , start=0, xlabel='immi_rate'
                                , ylabel='comp_overlap'):

    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')
    rangex = data[xlabel]; rangey = data[ylabel][start:]
    if range == xlabel: other = rangey; otherLabel = ylabel; plotRange = rangex; nbr=fixed
    else: other = rangex; otherLabel = xlabel; plotRange = rangex; nbr=start+fixed

    plotRange = data[range]
    # plots
    f = plt.figure(figsize=(3.5,2.5)); fig = plt.gcf(); ax = plt.gca()
    cmap = mpl.cm.get_cmap("viridis")
    divide = 2; plotRange = plotRange[::divide] #divide reduces number points

    for j, element in enumerate(plotRange):
        color = cmap(j/len(plotRange ) )
        if range == xlabel:
            plt.plot(np.arange(0,np.shape(data['sim_dist'])[2])
                    , data['sim_dist'][divide*j][start+fixed], lw=2, c=color )
        else:
            plt.plot(np.arange(0,np.shape(data['sim_dist'])[2])
                    , data['sim_dist'][fixed][start+divide*j], lw=2, c=color )

    plt.yscale('log')
    plt.xlim(0,int(1.5*data['carry_capacity']))
    plt.ylim(0.00001,.1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(
                vmin=np.min(plotRange), vmax=np.max(plotRange)))
    clb = plt.colorbar(sm)
    clb.set_label(VAR_SYM_DICT[range], labelpad=-30, y=1.1, rotation=0)
    plt.xlabel(r'$n$'); plt.ylabel(r'$P(n)$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + str(other[nbr]) +'.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + str(other[nbr]) +'.png');
    else:
        plt.show()

    return 0

def maximum_plot(filename, save=False, range='comp_overlap', start=0
                                , xlabel='immi_rate', ylabel='comp_overlap'):

    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')
    rangex = data[xlabel]; rangey = data[ylabel][start:]
    if range == xlabel: other = rangey; otherLabel = ylabel; plotRange = rangex
    else: other = rangex; otherLabel = xlabel; plotRange = rangey

    # plots
    f = plt.figure(figsize=(3.5,2.5)); fig = plt.gcf(); ax = plt.gca()
    cmap = mpl.cm.get_cmap("viridis")
    divide = 4; plotRange = plotRange[::divide] #divide reduces number points

    for j, element in enumerate(plotRange):
        color = cmap(j/len(plotRange ) )
        max0 = []; loc_max0 = [];
        maxN = []; loc_maxN = []
        for i, otherVal in enumerate(other):
            if range == xlabel:
                dstbn =  data['sim_dist'][divide*j][start+i]
            else:
                dstbn =  data['sim_dist'][i][start+divide*j]
            # Max at 0
            if dstbn[0] > dstbn[1]:
                max0.append(0); loc_max0.append(otherVal)
            max_excl0 = np.argmax(dstbn[1:])
            # Max at N
            if max_excl0 != 0:
                maxN.append(max_excl0+1); loc_maxN.append(otherVal)

        plt.scatter(loc_max0, max0, s=10/np.sqrt(j+1), lw=2, c=color
                            , edgecolor='none' )
        plt.scatter(loc_maxN, maxN, s=10/np.sqrt(j+1), lw=2, c=color
                            , edgecolor='None' )

    plt.xscale('log'); plt.xlim(np.min(other), np.max(other))
    plt.ylim(0.,data['carry_capacity'])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(
                vmin=np.min(plotRange), vmax=np.max(plotRange)))
    clb = plt.colorbar(sm)
    clb.set_label(VAR_SYM_DICT[range], labelpad=-30, y=1.1, rotation=0)
    plt.xlabel(VAR_NAME_DICT[otherLabel]); plt.ylabel(r'$\tilde{n}$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + 'maximums' +'.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + 'maximums' +'.png');
    else:
        plt.show()

    return 0

def convert_npz_mat(filename):
    data = np.load(filename)
    data_dict = {}
    for key in list(data.keys()):
        data_dict[key] = data[key]

    sio.savemat(filename[:-3] + 'mat', mdict=data_dict)
    print('done', filename[:-3] + 'mat')

    return 0

def fig_corr(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):

    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')

    # transform
    start = 0
    if xlabel=='nbr_species': start = 1

    rangex = data[xlabel][start:]; rangey = data[ylabel][:]

    K = data['carry_capacity']; rminus = data['death_rate']
    rplus = data['birth_rate']; S = data['nbr_species']

    correff = [ 'corr_ni_nj2D', 'corr_J_n2D', 'corr_Jminusn_n2D'\
                , 'coeff_ni_nj2D', 'coeff_J_n2D', 'coeff_Jminusn_n2D'
                    ]
    corref_name = [r'cov($n_i,n_j$)/$\sigma_{n_i}\sigma_{n_j}$'\
                    , r'cov($J,n$)/$\sigma_{J}\sigma_{n}$'\
                    , r'cov($J-n,n$)/$\sigma_{J-n}\sigma_{n}$'\
                    , r'cov($n_i,n_j$)/$\langle n_i \rangle \langle n_j\rangle$'\
                    , r'cov($J,n$)/$\langle J \rangle \langle n \rangle$'\
                    , r'cov($J-n,n$)/$\langle (J-n) \rangle \langle n \rangle$'\
                    ]

    # plots
    #divnorm = colors.DivergingNorm(vmin=-0.1, vcenter=0.0, vmax=.0)
    divnorm = None
    imshow_kw = { 'aspect' : None, 'interpolation' : None, 'norm' : divnorm }
    imshow_kw['cmap'] = copy.copy(mpl.cm.get_cmap('RdBu'))

    for i, score in enumerate( correff ):
        max = np.max([-np.min(data[score]),np.max(data[score])])
        min = np.min([-np.max(data[score]),np.min(data[score])])
        divnorm = colors.DivergingNorm(vmin=min, vcenter=0.0, vmax=max)
        imshow_kw['norm'] = divnorm
        """
        if score[:3] == 'cor':
            my_cmap = copy.copy(mpl.cm.get_cmap('magma_r')) # copy the default cmap
            my_cmap.set_bad(( 0,0,0 ))
        else:
            my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
            my_cmap.set_bad(( 0,0,0 ))
        imshow_kw['cmap'] = my_cmap
        """

        f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()
        im = plt.imshow( (data[score]).T, **imshow_kw)
        set_axis(ax, plt, pbx, pby,rangex, rangey, xlog, ylog, xdatalim
                        , ydatalim, xlabel, ylabel)

        cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
        #tick_locator = mpl.ticker.MaxNLocator(nbins=6)
        #cb.locator = tick_locator
        #cb.update_ticks()
        plt.title(corref_name[i])
        if save:
            plt.savefig(MANU_FIG_DIR + os.sep + score + '.pdf');
            #plt.savefig(MANU_FIG_DIR + os.sep + score + '.png');
        else:
            plt.show()
    return 0

def fig_timecorr(sim, sim_nbr, save=False, xlabel='immi_rate'
                    , ylabel='comp_overlap', start=0, xlog=True, ylog=True
                    , ydatalim=None, xdatalim=None, revision=None, pbx=20
                    , pby=20 ):

    filename = sim + os.sep + 'sim' + str(sim_nbr) + os.sep + 'results_0.pickle'
    with open(filename, 'rb') as handle: data = pickle.load(handle)

    plt.style.use('custom_heatmap.mplstyle')

    n = 2 # TODO : Need the [2:] for now, not sure why...
    autocor, spectrum, specAutocor, specSpectrum, newTimes =\
            autospec(data['results']['times'][n:],\
            data['results']['trajectory'][n:])

    conditions = VAR_SYM_DICT[xlabel] + ': ' + str(data[xlabel])[:6] + ' and '\
                        + VAR_SYM_DICT[ylabel] + ': ' + str(data[ylabel])[:5]
    fracTime = 10

    #
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    plt.plot(newTimes[:len(specAutocor)//fracTime], specAutocor[:len(specAutocor)//fracTime], 'k')
    #plt.yscale('log');
    plt.xlim(left=0.0)
    plt.title('time correlation richness ' + conditions )
    figname = 'rich_corr.pdf'
    if save:
        plt.savefig(sim + os.sep + 'sim' + str(sim_nbr) + os.sep + figname);
        plt.close()
    else: plt.show()

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    for i in np.random.randint(low=0, high=len(autocor), size=(6,)):
        plt.plot(newTimes[:len(autocor[i])//fracTime], autocor[i][:len(autocor[i])//fracTime])
    plt.xlim(left=0.0); plt.yscale('log');
    plt.title('time correlation abundance ' + conditions)
    figname = 'abund_corr.pdf'
    if save:
        plt.savefig(sim + os.sep + 'sim' + str(sim_nbr) + os.sep + figname);
        plt.close()
    else: plt.show()

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    plt.plot(specSpectrum, 'k');
    plt.xscale('log'); plt.yscale('log')
    plt.title('spectrum richness')
    figname = 'rich_spectrum.pdf'
    if save:
        plt.savefig(sim + os.sep + 'sim' + str(sim_nbr) + os.sep + figname);
        plt.close()
    else: plt.show()

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    for i in np.random.randint(low=0, high=len(spectrum), size=(6,)):
        plt.plot( spectrum[i] )
    plt.yscale('log'); plt.xscale('log')
    plt.title('spectrum abundance')
    figname = 'abund_spectrum.pdf'
    if save:
        plt.savefig(sim + os.sep + 'sim' + str(sim_nbr) + os.sep + figname);
        plt.close()
    else: plt.show()

    return 0

def fig_timescales_autocor(sim, save=False, range='comp_overlap', start=0
                                , xlabel='immi_rate', ylabel='comp_overlap'):

    filename = sim + os.sep + NPZ_SHORT_FILE
    with open(filename, 'rb') as handle: data = pickle.load(handle)

    conditions = VAR_SYM_DICT[xlabel] + ': ' + str(data[xlabel])[:7] + ' and '\
                        + VAR_SYM_DICT[ylabel] + ': ' + str(data[ylabel])[:6]

    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')
    rangex = data[xlabel]; rangey = data[ylabel][start:]
    if range == xlabel: other = rangey; otherLabel = ylabel; plotRange = rangex
    else: other = rangex; otherLabel = xlabel; plotRange = rangey

    ######################### plot Species exponential #########################
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    cmap = mpl.cm.get_cmap("viridis")

    for j, element in enumerate(plotRange):
        color = cmap(j/len(plotRange ) )
        ax.plot(other, data['time_autocor_spec2D'][i,:], lw=2, fmt='-o'
                    , c=color)
        plt.scatter(other, data['suppress_return2D'][i,:]+['dominance_turnover2D'][i,:]
                    , lw=2, c=color, edgecolor='none' )
    ax.plot(np.NaN, np.NaN,yerr=np.NaN, fmt='-o', color='silver'
                    , label=r'$T_{exp}$ : $e^{-t/T_{exp}}$')
    plt.scatter(np.NaN, np.Nan, lw=2, c='silver', edgecolor='none'
            , label=r'$\langle T_0(\tilde{n}) \rangle + \langle T(0\rightarrow \tilde{n}) \rangle$')

    ax.set_title(r'Time scale species correlation $e^{-t/T_{exp}}$');
    plt.legend(); plt.xscale('log'); plt.yscale('log')
    plt.xlim(np.min(other), np.max(other)) #plt.ylim( np.min(,),np.max() )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(
                vmin=np.min(plotRange), vmax=np.max(plotRange)))
    clb = plt.colorbar(sm)
    clb.set_label(VAR_SYM_DICT[range], labelpad=-30, y=1.1, rotation=0)
    plt.xlabel(VAR_NAME_DICT[otherLabel]); plt.ylabel(r'times')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + range
                                    + '_' + 'species_timescale_cor' +'.pdf');
        #plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + 'maximums' +'.png');
        plt.close()
    else: plt.show()

    time_autocor_spec2D[i,:]        = time_autocor_spec[sim]
    mean_time_autocor_abund2D[i,j]  = mean_time_autocor_abund[sim]
    std_time_autocor_abund2D[i,j]   = std_time_autocor_abund[sim]
    dominance_turnover2D[i,j]       = dominance_turnover[sim]
    suppress_turnover2D[i,j]        = suppress_turnover[sim]
    dominance_return2D[i,j]         = dominance_return[sim]
    suppress_return2D[i,j]          = suppress_return[sim]

    ######################### plot abundance exponential #######################

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    cmap = mpl.cm.get_cmap("viridis")

    for j, element in enumerate(plotRange):
        color = cmap(j/len(plotRange ) )
        ax.errorbar(other, data['mean_time_autocor_abund2D'][i,:]
                        , yerr=data['std_time_autocor_abund2D'][i,:], fmt='-o'
                        , c=color)
        plt.scatter(other, data['suppress_turnover2D'][i,:], lw=2, c=color
                        , edgecolor='none', marker='o' )
        plt.scatter(other, data['dominance_turnover2D'][i,:], lw=2, c=color
                        , edgecolor='none', marker='x' )
    ax.errorbar(np.NaN, np.NaN,yerr=np.NaN, fmt='-o', color='silver'
                        , label=r'$T_{exp}$ : $e^{-t/T_{exp}}$')
    plt.scatter(np.NaN, np.Nan, lw=2, c='silver', edgecolor='none', marker='o'
                        , label=r'$\langle T(0\rightarrow 0) \rangle$')
    plt.scatter(np.NaN, np.Nan, lw=2, c='silver', edgecolor='none', marker='x'
                        , label=r'$\langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')

    ax.set_title(r'Time scale abundance correlation $e^{-t/T_{exp}}$')
    plt.legend(); plt.xscale('log'); plt.yscale('log')
    plt.xlim(np.min(other), np.max(other)) #plt.ylim( np.min(,),np.max() )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(
                vmin=np.min(plotRange), vmax=np.max(plotRange)))
    clb = plt.colorbar(sm)
    clb.set_label(VAR_SYM_DICT[range], labelpad=-30, y=1.1, rotation=0)
    plt.xlabel(VAR_NAME_DICT[otherLabel]); plt.ylabel(r'times')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + range
                                    + '_' + 'abundance_timescale_cor'+'.pdf');
        #plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + 'maximums' +'.png');
        plt.close()
    else: plt.show()

    return 0

def plot_trajectory(dir, sim_nbr, nbr_steps, plot_dstbn=True, colour='red'
                        , save=False):
    """
    Plot species trajectories for the t=time last reactions. May also print the
    distribution on the side if needed.
    """
    with open(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
               'results_0.pickle', 'rb') as handle:
        data  = pickle.load(handle)

    # distribution
    ss_dist_sim     = data['results']['ss_distribution'] \
                            / np.sum(data['results']['ss_distribution'])

    # TODO : NEW BETTER WAY OF DOING IT
    trajectories = data['results']['trajectory'][-nbr_steps:]
    times        = data['results']['times'][-nbr_steps:]

    # DUMB OLD WAY
    #trajectories = np.loadtxt(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
    #                                'trajectory_0.txt')[-nbr_steps:]
    #times        = np.loadtxt(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
    #                                'trajectory_0_time.txt')[-nbr_steps:]


    if plot_dstbn: fig, (ax1, ax2) = plt.subplots(1, 2
                                        , gridspec_kw={'width_ratios': [3, 1]}
                                        , figsize=(4,2.5))
    else: fig, ax1 = plt.figure(figsize=(3.25,2.5))

    focus_species = 15
    for i in np.arange( np.shape( trajectories )[1] ):
        ax1.plot(times, trajectories[:,i], alpha=0.1, color='gray')
    ax1.plot(times, trajectories[:,focus_species], color='k')

    ax1.set_xlabel('time'); ax1.set_ylabel(r'abundance, $n$')
    ax1.set_xlim( left=min(times), right=max(times) )

    if plot_dstbn:
        ax2.plot(ss_dist_sim, np.arange(0,len( ss_dist_sim )), color=colour
                            , linewidth=3)
        ax2.set_xscale('log')
        ax2.set_xlabel(r'$P(n)$')

    # set 0 abundance as limit
    ax1.set_ylim(bottom=0, top=int(1.5*data['carry_capacity']) )
    ax2.set_ylim(bottom=0, top=int(1.5*data['carry_capacity']) )

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    ax2.label_outer()

    if save:
        plt.savefig(dir + os.sep + 'traj_' + str(sim_nbr) + '.pdf')
        #plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + 'maximums' +'.png');
        plt.close()
    else: plt.show()

    return 0


if __name__ == "__main__":

    sim_immi        = RESULTS_DIR + os.sep + 'multiLV71'
    sim_immi_inset  = RESULTS_DIR + os.sep + 'multiLV79'
    sim_spec        = RESULTS_DIR + os.sep + 'multiLV77'
    sim_corr        = RESULTS_DIR + os.sep + 'multiLV80'
    sim_time        = RESULTS_DIR + os.sep + 'multiLV6'


    #mlv_consolidate_sim_results( sim_spec, 'nbr_species', 'comp_overlap')
    #mlv_consolidate_sim_results( sim_immi, 'immi_rate', 'comp_overlap')
    #mlv_consolidate_sim_results( sim_corr, 'immi_rate', 'comp_overlap')


    save = True
    #for i in np.arange(31,37):
        #fig_timecorr( sim_time, i , save=save); print("done ",i)

    #mlv_consolidate_sim_results(sim_time, parameter1='immi_rate', parameter2='comp_overlap')


    #many_parameters_dist(npz_file, save)

    #many_parameters_dist(sim_immi+os.sep+NPZ_SHORT_FILE,save=True, fixed=4, start=20)
    #many_parameters_dist(sim_immi+os.sep+NPZ_SHORT_FILE,save=True, fixed=20, start=20)
    #many_parameters_dist(sim_immi+os.sep+NPZ_SHORT_FILE, range='immi_rate', save=True, start=20, fixed=30)
    #maximum_plot(sim_immi+os.sep+NPZ_SHORT_FILE, range='immi_rate', save=True, start=20)

    # THESE ARE GOOD
    #fig2(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #fig2(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species',xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)
    #compare_richness(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #fig_corr(sim_corr+os.sep+NPZ_SHORT_FILE, save, revision='80')
    #fig_timecorr(sim_time + os.sep + "sim1" + os.sep + "results_0.pickle")
    #fig3A(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    fig3(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')

    #plot_trajectory(sim_time, 19, 100000, colour='mediumturquoise', save=True)
    #plot_trajectory(sim_time, 34, 100000, colour='khaki', save=True)


    # OLD STUFF
    #fig2A(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40))
    #fig2B(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #fig2A(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species', xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)
    #fig2B(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species', distplots=False, xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)
    #fig2(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species',xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)
    #fig2B(sim_immi_inset+os.sep+NPZ_SHORT_FILE, save, ylog=False, xlog=False, distplots=False, pby=15)
    #fig2B(sim_spec+os.sep+NPZ_SHORT_FILE, save)
    #fig3A(npz_file, save)
    #fig3B(npz_file, save)
    #fig3CDE(npz_file, save)
