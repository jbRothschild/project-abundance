import os, glob, csv, pickle, copy, time, scipy

import matplotlib as mpl; from matplotlib import colors
import matplotlib.pyplot as plt; from matplotlib.lines import Line2D
import numpy as np; from scipy.signal import argrelextrema
import pandas as pd
import scipy.io as sio
from scipy.signal import savgol_filter, find_peaks
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


    # TODO change to dictionary
    for i in np.arange(nbr_sims):
        sim_nbr = i + 1
        if not os.path.exists(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
                   'results_0.pickle'):
            print("Missing simulation: " + str(sim_nbr))
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
            corr_ni_nj[sim_nbr-1] = model.results['corr_ni_nj']
            coeff_ni_nj[sim_nbr-1] = model.results['coeff_ni_nj']
            corr_J_n[sim_nbr-1] = model.results['corr_J_n']
            coeff_J_n[sim_nbr-1] = model.results['coeff_J_n']
            corr_Jminusn_n[sim_nbr-1] = model.results['corr_Jminusn_n']
            coeff_Jminusn_n[sim_nbr-1] = model.results['coeff_Jminusn_n']
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

def deterministic_mean(nbr_species, mu, rho, rplus, rminus, K):
    # Deterministic mean fixed point
    det_mean = K*( ( 1. + np.sqrt( 1.+ 4.*mu*( 1. + rho*( nbr_species - 1. ) ) /
                (K*(rplus-rminus)) ) ) / ( 2.*( 1. + rho*(nbr_species-1.) ) ) )
    return int(det_mean)

def mfpt_a2b(dstbn, a, b, mu, rplus, rminus, K):
    """
    From distribution, get the mfpt <T_{b}(a)>, a<b
    """
    mfpt = 0.0
    for i in np.arange(a,b):
        mfpt += ( np.sum(dstbn[:i+1] ) ) / ( ( rplus*i + mu )*dstbn[i] )
    return mfpt

def mfpt_b2a(dstbn, a, b, mu, rplus, rminus, K):
    """
    From distribution, get the mfpt <T_{a}(b)>
    """
    mfpt = 0
    for i in np.arange(a,b):
        mfpt += ( np.sum(dstbn[i+1:]) ) / ( ( rplus*i + mu ) * dstbn[i] )
        return mfpt

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

    #modality_arr = 2*np.ones( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    modality_arr = 2*np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )

    for i in np.arange(np.shape(arr)[0]):
        for j in np.arange(np.shape(arr)[1]):
            #smooth_arr = smooth(arr[i,j,:],21)
            smooth_arr = smooth(arr[i,j,:65],4) # 71, 78
            #smooth_arr = smooth(arr[i,j,:100],15) # 79 inset
            max_idx, _ = find_peaks( smooth_arr )

            if arr[i,j,0] > arr[i,j,1]:
                modality_arr[i,j] = line_names.index('bimodal')
                if len(max_idx) == 1 and max_idx[0]<3:
                    modality_arr[i,j] = line_names.index('unimodal\n peak at 0')
                elif len(max_idx) > 2:
                    if revisionmanual == None:
                        pass
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

def meanJ_est(dstbn, nbr_species):
    meanJ = np.zeros((np.shape(dstbn)[0],np.shape(dstbn)[1]))
    for i in np.arange(np.shape(dstbn)[0]):
        for j in np.arange(np.shape(dstbn)[1]):
            dstbnij = dstbn[i,j]
            meanJ[i,j] = nbr_species*np.dot(dstbnij,np.arange(len(dstbnij)))
    #f = plt.figure(); fig = plt.gcf(); ax = plt.gca() im = ax.imshow(meanJ.T);
    #plt.colorbar(im,ax=ax); ax.invert_yaxis(); plt.show()
    return meanJ

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

    # transform
    rangex = data[xlabel][start:]; rangey = data[ylabel][:]

    # species simulations didn't all work, erase the 0s
    start = 0
    if xlabel=='nbr_species':
        start = 1

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
    my_cmap.set_bad((0,0,0))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    im = plt.imshow( turnover.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)

    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])

    plt.title(r'$\langle T_0(0) \rangle$')
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
    rangex = data[xlabel]; rangey = data[ylabel]
    K = data['carry_capacity']; rminus = data['death_rate']; rplus = data['birth_rate'];
    S = data['nbr_species']

    # 2D rho-mu
    mu  = (rangex*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])).T).T

    rho = (rangey*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])))

    arr             = data['sim_dist']
    meanJ           = meanJ_est(arr, (np.shape(data['rich_dist'])[2]-1))
    max_arr         = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    dom_turnover    = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    fpt_dominance   = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    fpt_submission  = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    for i in np.arange(np.shape(arr)[0]):
        for j in np.arange(np.shape(arr)[1]):
            smooth_arr = arr[i,j,:]
            smooth_arr = smooth(arr[i,j,:-25],7)
            max_idx, _ = find_peaks( smooth_arr )
            if len(max_idx) == 0:
                print("No maximum's found, problematic")
                nbr_species = np.dot(data['rich_dist'][i,j]
                                    , np.arange(len(data['rich_dist'][i,j])) )
                nbr = deterministic_mean(nbr_species, mu[i,j],rho[i,j], rplus
                                                    , rminus, K)
                # Or something else
            else:
                nbr = max_idx[-1]
            max_arr[i,j] = nbr
            dom_turnover[i,j] = ( ( 1. + arr[i,j,nbr] ) / ( arr[i,j,nbr]
                    * ( rplus*nbr + mu[i,j] + nbr*( rminus + ( rplus-rminus )
                    * ( ( 1. - rho[i,j] ) * nbr + rho[i,j] * meanJ[i,j] )/K ))))
            fpt_dominance[i,j] = mfpt_a2b(arr[i,j], 0, nbr, mu[i,j], rplus
                                                    , rminus, K)
            fpt_submission[i,j] = mfpt_b2a(arr[i,j], 0, nbr, mu[i,j], rplus
                                                    , rminus, K)

    fpt_dom_turnover = fpt_dominance + fpt_submission

    # Fig3C
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('cividis_r')) # copy the default cmap
    my_cmap.set_bad((0,0,0))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( dom_turnover.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T_{n*}(n*) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "dominance_turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "dominance_turnover" + '.png');
    else:
        plt.show()

    ## FIGURE 3D
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('cividis_r')) # copy the default cmap
    my_cmap.set_bad((0,0,0))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( fpt_dominance.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    # title
    plt.title(r'$\langle T_{n*}(0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_dominance" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_dominance" + '.png');
    else:
        plt.show()

    ## FIGURE 3E
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('cividis_r')) # copy the default cmap
    my_cmap.set_bad((0,0,0))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( fpt_submission.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T_{0}(n*) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_submission" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_submission" + '.png');
    else:
        plt.show()

    ## FIGURE 3F
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('cividis_r')) # copy the default cmap
    my_cmap.set_bad((0,0,0))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( fpt_dom_turnover.T, **imshow_kw)
    set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T_{0}(n^*) \rangle+\langle T_{n^*}(0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_submission" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_submission" + '.png');
    else:
        plt.show()


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
    plt.xlabel(VAR_NAME_DICT[otherLabel]); plt.ylabel(r'$n^*$')
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
                    , r'cov($n_i,n_j$)/$\langle n_i n_j \rangle$'\
                    , r'cov($J,n$)/$\langle J n \rangle$'\
                    , r'cov($J-n,n$)/$\langle (J-n) n \rangle$'\
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
            my_cmap.set_bad((0,0,0))
        else:
            my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
            my_cmap.set_bad((0,0,0))
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

if __name__ == "__main__":

    sim_immi        = RESULTS_DIR + os.sep + 'multiLV71'
    sim_immi_inset  = RESULTS_DIR + os.sep + 'multiLV79'
    sim_spec        = RESULTS_DIR + os.sep + 'multiLV77'
    sim_corr        = RESULTS_DIR + os.sep + 'multiLV80'


    #mlv_consolidate_sim_results( sim_spec, 'nbr_species', 'comp_overlap')
    #mlv_consolidate_sim_results( sim_immi, 'immi_rate', 'comp_overlap')
    #mlv_consolidate_sim_results( sim_corr, 'immi_rate', 'comp_overlap')


    save = True
    #many_parameters_dist(npz_file, save)

    #many_parameters_dist(sim_immi+os.sep+NPZ_SHORT_FILE,save=True, fixed=4, start=20)
    #many_parameters_dist(sim_immi+os.sep+NPZ_SHORT_FILE,save=True, fixed=20, start=20)
    many_parameters_dist(sim_immi+os.sep+NPZ_SHORT_FILE, range='immi_rate', save=True, start=20, fixed=30)
    maximum_plot(sim_immi+os.sep+NPZ_SHORT_FILE, range='immi_rate', save=True, start=20)


    #fig2(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #fig_corr(sim_corr+os.sep+NPZ_SHORT_FILE, save, revision='80')
    fig3(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #fig_autocorr()

    #fig2A(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40))
    #fig2B(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #fig2A(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species', xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)
    #fig2B(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species', distplots=False, xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)
    #fig2B(sim_immi_inset+os.sep+NPZ_SHORT_FILE, save, ylog=False, xlog=False, distplots=False, pby=15)
    #fig2B(sim_spec+os.sep+NPZ_SHORT_FILE, save)
    #fig3A(npz_file, save)
    #fig3B(npz_file, save)
    #fig3CDE(npz_file, save)
