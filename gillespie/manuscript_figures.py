import os, glob, csv, pickle, copy, time

import matplotlib as mpl; from matplotlib import colors
import matplotlib.pyplot as plt; from matplotlib.lines import Line2D
import numpy as np; from scipy.signal import argrelextrema
from scipy.signal import savgol_filter, find_peaks
np.seterr(divide='ignore', invalid='ignore')

from gillespie_models import RESULTS_DIR, MultiLV, SIR
import theory_equations as theqs
from settings import VAR_NAME_DICT, COLOURS, IMSHOW_KW, NPZ_SHORT_FILE

MANU_FIG_DIR = 'figures' + os.sep + 'manuscript'
while not os.path.exists( os.getcwd() + os.sep + MANU_FIG_DIR ):
    os.makedirs(os.getcwd() + os.sep + MANU_FIG_DIR);

plt.style.use('custom.mplstyle')

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
    #print(len(s))
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

    # TODO change to dictionary
    for i in np.arange(nbr_sims):
        sim_nbr = i + 1
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
        end = time.time()
        hours, rem = divmod( end-start, 3600 )
        minutes, seconds = divmod( rem, 60 )
        print(">>{} Time elapsed : {:0>2}:{:0>2}:{:05.2f}".format(i, int(hours)
                                                    , int(minutes), seconds) )
        # Value of parameters
        param1[i] = param_dict[parameter1]; param2[i] = param_dict[parameter2]

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
    mf_dist2D   = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    conv_dist2D = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    sim_dist2D  = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    rich_dist2D = np.zeros( ( dim_1,dim_2,length_longest_rich ) )

    # put into a 2d array all the previous results
    for sim in np.arange(nbr_sims):
        i                   = np.where( param1_2D==param1[sim] )[0][0]
        j                   = np.where( param2_2D==param2[sim] )[0][0]
        sim_dist2D[i,j]     = sim_dist[sim]
        mf_dist2D[i,j]      = mf_dist[sim]
        #conv_dist2D[i,j]    = conv_dist[sim]
        rich_dist2D[i,j]    = rich_dist[sim]

    # arrange into a dictionary to save
    dict_arrays = {  parameter1 : param1_2D, parameter2     : param2_2D
                                            , 'sim_dist'    : sim_dist2D
                                            , 'mf_dist'     : mf_dist2D
                                            , 'conv_dist'   : conv_dist2D
                                            , 'rich_dist'   : rich_dist2D
                                            }
    # save results in a npz file
    np.savez(filename, **dict_arrays)

def deterministic_mean(nbr_species, mu, rho):
    K = 50.; rminus = 1.; rplus = 2.; S = 30
    det_mean = K*( ( 1. + np.sqrt( 1.+ 4.*mu*( 1. + rho*( nbr_species - 1. ) ) /
                (K*(rplus-rminus)) ) ) / ( 2.*( 1. + rho*(nbr_species-1.) ) ) )
    return int(det_mean)

def mfpt_a2b(dstbn, a, b, mu):
    """
    From distribution, get the mfpt <T_{b}(a)>, a<b
    """
    K = 50.; rminus = 1.; rplus = 2.; S = 30
    mfpt = 0.0
    for i in np.arange(a,b):
        mfpt += ( np.sum(dstbn[:i+1] ) ) / ( ( rplus*i + mu )*dstbn[i] )
    return mfpt

def mfpt_b2a(dstbn, a, b, mu):
    """
    From distribution, get the mfpt <T_{a}(b)>
    """
    K = 50.; rminus = 1.; rplus = 2.; S = 30
    mfpt = 0
    for i in np.arange(a,b):
        mfpt += ( np.sum(dstbn[i+1:]) ) / ( ( rplus*i + mu ) * dstbn[i] )
        return mfpt

def determine_modality_sim(arr, lines):
    """
    This is a fine art, which I do not truly believe in
    """
    modality_arr = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    for i in np.arange(np.shape(arr)[0]):
        for j in np.arange(np.shape(arr)[1]):
            #smooth_arr = smooth(arr[i,j,:],21)
            smooth_arr = smooth(arr[i,j,:25],7)
            max_idx, _ = find_peaks( smooth_arr )
            #f = plt.figure()
            #plt.plot(savgol_filter(arr[i,j,:],21,3))
            #plt.plot(arr[i,j])
            #plt.show()
            if arr[i,j,0]>arr[i,j,1]:
                modality_arr[i,j] = lines[2]
                if len(max_idx) == 1 and max_idx[0]<5:
                    modality_arr[i,j] = lines[0]
            else:
                modality_arr[i,j] = lines[1]

    return modality_arr

def determine_bimodality_mf(arr):
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

def fig2A(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'):
    """
    Heatmap of richness, with
    """
    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    lines = [2,30]; line_colours = ['firebrick','hotpink']
    imshow_kw = { 'cmap' : 'cividis_r', 'aspect' : None, 'interpolation' : None}

    # plots
    xrange = data[xlabel]; yrange = data[ylabel]
    mean_rich_sim = np.tensordot(data['rich_dist'] ,
                        np.arange(np.shape(data['rich_dist'])[2]) , axes=(2,0))
    mf_rich = np.shape(data['rich_dist'])[2] * ( 1 - data['mf_dist'][:,:,0] )
    rates_rich_sim = data['sim_dist'][:,:,0]/(1.0 - data['sim_dist'][:,:,0])
    mask = 4*np.ones(( len(xrange), len(yrange) ))
    rates_rich_29 = np.ma.masked_where(np.around(1*rates_rich_sim) != 29., mask)
    rates_rich_2 = np.ma.masked_where(np.around(29*rates_rich_sim) != 1., mask)

    im = ax.imshow(mean_rich_sim.T, **imshow_kw)
    MF = ax.contour( mf_rich.T, lines, linestyles='solid'
                        , colors = line_colours, linewidths = 3)
    x, y = np.meshgrid( np.arange(len(xrange)), np.arange(len(yrange)) )
    plt.scatter(x, y, s=rates_rich_29[x,y], marker='x', c=line_colours[0])
    plt.scatter(x, y, s=rates_rich_2[x,y], marker='x', edgecolor='k'
                    , c=line_colours[1])

    POINTS_BETWEEN_X_TICKS = 20; POINTS_BETWEEN_Y_TICKS = 40
    ax.set_xticks([i for i, xval in enumerate(xrange)
                        if i % POINTS_BETWEEN_X_TICKS == 0])
    ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                        for i, xval in enumerate(xrange)
                        if (i % POINTS_BETWEEN_X_TICKS==0)])
    ax.set_yticks([i for i, kval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS == 0])
    ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                        for i, yval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS==0])
    ax.invert_yaxis();
    cb = plt.colorbar(im,ax=ax,cmap=imshow_kw['cmap'])
    cb.ax.hlines(lines, cb.vmin, cb.vmax, colors=line_colours, lw=[3,6])
    #cb.set_label('mean richness simulation')
    cb.ax.set_title('sim.')
    plt.xlabel(VAR_NAME_DICT[xlabel]); plt.ylabel(VAR_NAME_DICT[ylabel])
    fake_legend = [ Line2D([0],[0], color='grey', lw=2, linestyle='solid'
                    , label=r'mean field approx.')
                    , Line2D([0],[0], color='grey', lw=2, linestyle='dotted'
                                    , label=r'flux balance sim.')
                    ]
    ax.legend(handles=fake_legend, loc=4)
    plt.title('Richness')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "richness" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "richness" + '.png');
    else:
        plt.show()

def fig2B(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'):
    """
    Heatmap of modality
    Need to smooth out simulation results. How to compare?
    """
    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')
    xrange = data[xlabel]; yrange = data[ylabel]
    K = 50.; rminus = 1.; rplus = 2.; S = 30

    lines        = [0,1,2];
    line_colours = ['wheat','mediumorchid','cornflowerblue']
    line_names   = ['unimodal\n peak at 0','unimodal\n peak at >0','multimodal']
    bounds       = [-0.5, 0.5, 1.5, 2.5]; center_name = [0, 0, 0]
    lines_center = [ a + b for a,b in zip(lines, center_name)]
    modality_sim = determine_modality_sim( data['sim_dist'], lines )
    modality_mf  = determine_bimodality_mf( data['mf_dist'] )

    mf_meanJ = meanJ_est(data['mf_dist'], (np.shape(data['rich_dist'])[2]-1))
    mf_rich = np.shape(data['rich_dist'])[2] * ( 1 - data['mf_dist'][:,:,0] )
    mf_meanJ = meanJ_est(data['sim_dist'], (np.shape(data['rich_dist'])[2]-1))

    mu  = (xrange*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1]))).T
    rho = (yrange*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])))
    mf_unimodal = (1./mu)*(rminus + (rplus-rminus)*( 1. +
                                rho*( mf_meanJ - 1. ) )/K )
    # fixed point given number of species present
    nstar       = K * (1. + np.sqrt( ( 1. + 4*mu*( 1. + rho*( mf_rich-1. ) ) )
                    /( K*(rplus-rminus) ) ) ) / ( 2*( 1.+rho*( mf_rich-1. ) ) )

    # fixed point given J
    nstar2      = ( K - rho * mf_meanJ - 2.*(1.-rho) + np.lib.scimath.sqrt( (
                    K - rho * mf_meanJ - 2.*(1.-rho) )**2 - 4.*(1.-rho)*(
                    rminus*K + (rplus-rminus)*( 1 + rho*(mf_meanJ-1.) ) - mu*K )
                    /(rplus-rminus) ) ) / ( 2.*( 1. - rho) )
    # From probability distribution +
    nstar3      = ( K - rho * mf_meanJ - 2.*(1.-rho) + np.lib.scimath.sqrt( (
                    K - rho * mf_meanJ - 2.*(1.-rho) )**2 - 4.*(1.-rho)*(
                    rminus*K + (rplus-rminus)*( 1 + rho*(mf_meanJ-1.) ) - K*mu )
                    /(rplus-rminus) ) ) / ( 2.*( 1. - rho) )
    # From probability distribution -
    nstar4      = ( K - rho * mf_meanJ - 2.*(1.-rho) - np.lib.scimath.sqrt( (
                    K - rho * mf_meanJ - 2.*(1.-rho) )**2 - 4.*(1.-rho)*(
                    rminus*K + (rplus-rminus)*( 1 + rho*(mf_meanJ-1.) ) - mu*K )
                    /(rplus-rminus) ) ) / ( 2.*( 1. - rho) )

    nstar5      = (K - rho*mf_meanJ)/(1.-rho)/2. + np.lib.scimath.sqrt(
                    (K-rho*mf_meanJ)**2/(1-rho)**2
                    +4*K*( mu-rplus )/(1.-rho) )/2.

    # death rate = birth rate of 1
    mf_decay    = ((rplus-rminus)/(K*mu))*nstar*( 1. +
                                rho*( mf_meanJ - 1. ) - K ) + rplus/mu
    # death rate of 1 = immigration rate of S-R
    mf_decay1   = np.divide( rminus + (rplus-rminus)*( 1. +
                                rho*( mf_meanJ - 1. ) )/K , (30.-mf_rich)*mu )
    # death rate (n+1) = birth rate (n)
    n = nstar3.real
    mf_decay2   = ( ( mu  + rplus * (n) ) / ( n + 1. )
                    - ( rplus - rminus )*( 1. + rho * ( mf_meanJ - 1) ) / K )\
                    / rminus
    n = nstar4.real
    mf_decay3   = ( ( mu  + rplus * (n) ) / ( n + 1. )
                    - ( rplus - rminus )*( 1. - rho + rho * mf_meanJ ) / K )\
                    / rminus


    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()

    # plots
    cmap = colors.ListedColormap( line_colours )
    norm = colors.BoundaryNorm( bounds, cmap.N )
    im = plt.imshow(modality_sim.T, cmap=cmap, norm=norm)

    MF = ax.contour( (data['mf_dist'][:,:,0]/data['mf_dist'][:,:,1]).T, [1.]
                        , linestyles='solid', colors = 'k', linewidths = 3)
    MF = ax.contour( modality_mf.T, [0.5], linestyles='solid'
                        , colors = 'k', linewidths = 3)
    #MF2 = ax.contour( mf_unimodal.T, [1.], linestyles='dotted'
    #                    , colors = 'mediumseagreen', linewidths = 3)
    MF3 = ax.contour( mf_decay2.T, [1.], linestyles='dotted'
                        , colors = 'mediumseagreen', linewidths = 3)
    MF4 = ax.contour( (nstar3.imag).T, [0.], linestyles='dotted'
                        , colors = 'lightblue', linewidths = 3)

    POINTS_BETWEEN_X_TICKS = 20; POINTS_BETWEEN_Y_TICKS = 40
    ax.set_xticks([i for i, xval in enumerate(xrange)
                        if i % POINTS_BETWEEN_X_TICKS == 0])
    ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                        for i, xval in enumerate(xrange)
                        if (i % POINTS_BETWEEN_X_TICKS==0)])
    ax.set_yticks([i for i, kval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS == 0])
    ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                        for i, yval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS==0])
    ax.invert_yaxis();
    cb = plt.colorbar(im, ax=ax, cmap=cmap, ticks=lines_center)
    cb.ax.set_title('sim.')
    cb.ax.tick_params(width=0,length=0)
    cb.ax.set_yticklabels(line_names)#,rotation=270)
    fake_legend = [ Line2D([0],[0], color='k', lw=2, linestyle='solid'
                    , label=r'mean field approx.')
                    , Line2D([0],[0], color='mediumseagreen', lw=2
                                    , linestyle='dotted'
                                    , label=r'detailed balance bound')
                    ]
    ax.legend(handles=fake_legend, loc=3)

    plt.xlabel(VAR_NAME_DICT[xlabel]); plt.ylabel(VAR_NAME_DICT[ylabel])
    plt.title('Modality')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "modality" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "modality" + '.png');
    else:
        plt.show()
    return 0

def fig3A(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'):
    """
    Turnover rate
    """

    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')
    xrange = data[xlabel]; yrange = data[ylabel]
    K = 50.; rminus = 1.; rplus = 2.; S = 30

    mu  = (xrange*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1]))).T
    rho = (yrange*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])))

    # Nava
    nbr = 0
    dist = 'sim_dist' # sim or mf
    meanJ = meanJ_est(data[dist], (np.shape(data['rich_dist'])[2]-1))
    turnover_nava = 1. / (data[dist][:,:,nbr]*mu)
    turnover_calc = ( 1. + data[dist][:,:,nbr] ) / ( data[dist][:,:,nbr]
                        * ( rplus*nbr + mu + nbr*( rminus + ( rplus-rminus )
                        * ( ( 1. - rho ) * nbr + rho * meanJ )/K  ) ) )
    turnover = turnover_calc

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('cividis_r')) # copy the default cmap
    my_cmap.set_bad((0,0,0))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    im = plt.imshow( turnover.T, **imshow_kw)

    POINTS_BETWEEN_X_TICKS = 20; POINTS_BETWEEN_Y_TICKS = 40
    ax.set_xticks([i for i, xval in enumerate(xrange)
                        if i % POINTS_BETWEEN_X_TICKS == 0])
    ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                        for i, xval in enumerate(xrange)
                        if (i % POINTS_BETWEEN_X_TICKS==0)])
    ax.set_yticks([i for i, kval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS == 0])
    ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                        for i, yval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS==0])
    ax.invert_yaxis();

    plt.xlabel(VAR_NAME_DICT[xlabel]); plt.ylabel(VAR_NAME_DICT[ylabel])
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T_0(0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "turnover" + '.png');
    else:
        plt.show()

    return 0

def fig3B(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'):
    """
    Taylors Law
    """
    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')
    xrange = data[xlabel]; yrange = data[ylabel]
    K = 50.; rminus = 1.; rplus = 2.; S = 30

    dist = 'mf_dist'
    start = 1
    prob = data[dist][:,:,start:]\
            /np.sum(data[dist][:,:,start:],axis=2)[:,:,np.newaxis]
    print(np.shape(prob))
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

def fig3CDE(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'):
    """
    Time to extinction, time to dominance
    """
    data = np.load(filename); plt.style.use('custom_heatmap.mplstyle')
    xrange = data[xlabel]; yrange = data[ylabel]
    K = 50.; rminus = 1.; rplus = 2.; S = 30

    mu  = (xrange*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1]))).T
    rho = (yrange*np.ones( (np.shape(data['sim_dist'])[0]
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
                print("AAH")
                nbr_species = np.dot(data['rich_dist'][i,j]
                                    , np.arange(len(data['rich_dist'][i,j])) )
                nbr = deterministic_mean(nbr_species, mu[i,j],rho[i,j])
                # Or something else
            else:
                nbr = max_idx[-1]
            max_arr[i,j] = nbr
            dom_turnover[i,j] = ( ( 1. + arr[i,j,nbr] ) / ( arr[i,j,nbr]
                    * ( rplus*nbr + mu[i,j] + nbr*( rminus + ( rplus-rminus )
                    * ( ( 1. - rho[i,j] ) * nbr + rho[i,j] * meanJ[i,j] )/K ))))
            fpt_dominance[i,j] = mfpt_a2b(arr[i,j], 0, nbr, mu[i,j])
            fpt_submission[i,j] = mfpt_b2a(arr[i,j], 0, nbr, mu[i,j])

    ## FIGURE 3C
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('cividis_r')) # copy the default cmap
    my_cmap.set_bad((0,0,0))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    im = plt.imshow( dom_turnover.T, **imshow_kw)

    POINTS_BETWEEN_X_TICKS = 20; POINTS_BETWEEN_Y_TICKS = 40
    ax.set_xticks([i for i, xval in enumerate(xrange)
                        if i % POINTS_BETWEEN_X_TICKS == 0])
    ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                        for i, xval in enumerate(xrange)
                        if (i % POINTS_BETWEEN_X_TICKS==0)])
    ax.set_yticks([i for i, kval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS == 0])
    ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                        for i, yval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS==0])
    ax.invert_yaxis();

    plt.xlabel(VAR_NAME_DICT[xlabel]); plt.ylabel(VAR_NAME_DICT[ylabel])
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T_{n*}(n*) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "dominance_turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "dominance_turnover" + '.png');
    else:
        plt.show()

    ## FIGURE 3D
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('cividis_r')) # copy the default cmap
    my_cmap.set_bad((0,0,0))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    im = plt.imshow( fpt_dominance.T, **imshow_kw)

    POINTS_BETWEEN_X_TICKS = 20; POINTS_BETWEEN_Y_TICKS = 40
    ax.set_xticks([i for i, xval in enumerate(xrange)
                        if i % POINTS_BETWEEN_X_TICKS == 0])
    ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                        for i, xval in enumerate(xrange)
                        if (i % POINTS_BETWEEN_X_TICKS==0)])
    ax.set_yticks([i for i, kval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS == 0])
    ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                        for i, yval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS==0])
    ax.invert_yaxis();

    plt.xlabel(VAR_NAME_DICT[xlabel]); plt.ylabel(VAR_NAME_DICT[ylabel])
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
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
    im = plt.imshow( fpt_submission.T, **imshow_kw)

    POINTS_BETWEEN_X_TICKS = 20; POINTS_BETWEEN_Y_TICKS = 40
    ax.set_xticks([i for i, xval in enumerate(xrange)
                        if i % POINTS_BETWEEN_X_TICKS == 0])
    ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                        for i, xval in enumerate(xrange)
                        if (i % POINTS_BETWEEN_X_TICKS==0)])
    ax.set_yticks([i for i, kval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS == 0])
    ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                        for i, yval in enumerate(yrange)
                        if i % POINTS_BETWEEN_Y_TICKS==0])
    ax.invert_yaxis();

    plt.xlabel(VAR_NAME_DICT[xlabel]); plt.ylabel(VAR_NAME_DICT[ylabel])
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T_{0}(n*) \rangle$')
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
    xrange = data[xlabel]; yrange = data[ylabel]
    K = 50.; rminus = 1.; rplus = 2.; S = 30
    mu  = (xrange*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1]))).T
    rho = (yrange*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])))

    nbr_species = S # Or something different?
    det_mean = K*( ( 1. + np.sqrt( 1.+ 4.*mu*( 1. + rho*( nbr_species - 1. ) ) /
                (K*(rplus-rminus)) ) ) / ( 2.*( 1. + rho*(nbr_species-1.) ) ) )

if __name__ == "__main__":
    sim = 'multiLV45'; save = True
    sim_dir     = RESULTS_DIR + os.sep + sim
    #mlv_consolidate_sim_results( sim_dir )
    npz_file    = sim_dir + os.sep + NPZ_SHORT_FILE
    #fig2A(npz_file, save)
    #fig2B(npz_file, save)
    #fig3A(npz_file, save)
    fig3B(npz_file, save)
    #fig3CDE(npz_file, save)
