import os
import numpy as np
from scipy.signal import savgol_filter, find_peaks, argrelextrema
from src.manual_revision import DICT_REVISION
from src.settings import VAR_NAME_DICT
import matplotlib.pyplot as plt; from matplotlib.colors import LogNorm

from src.settings import VAR_NAME_DICT, COLOURS, IMSHOW_KW, NPZ_SHORT_FILE\
                    , VAR_SYM_DICT, MANU_FIG_DIR

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

def plot_prob(probability, i, j, colour, SRA=False, approx_dstbn={}
                            , nbrSpecies=30):
    DIST_DIR = MANU_FIG_DIR + os.sep + 'distributions'
    os.makedirs(DIST_DIR, exist_ok=True);
    size = 7
    lstyle = ['solid', 'dashed', 'dotted', 'dashdot']
    maxn = 150
    if not SRA:
        fsize = (1.75,1.25); cols = 1
    else:
        fsize = (3.5,1.25); cols = 2
    fig = plt.figure(figsize=fsize)
    gs = fig.add_gridspec(1,cols)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(probability[:maxn], color=colour, linewidth=2, zorder=10
                    , label='sim.')
    for k, key in enumerate(approx_dstbn):
        ax.plot(approx_dstbn[key][:maxn], color='k', linewidth=2, zorder=10
                        ,linestyle=lstyle[k], label=key)
    ax.set_xticks(list(np.arange(0,maxn+1,50)))
    ax.axes.labelsize = size
    ax.set_xlabel(r'n',fontsize=size, labelpad=2);
    ax.set_ylabel(r'P(n)', fontsize=size, labelpad=2);
    ax.set_xlim([0,maxn]); plt.yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=size)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l)
    #ax.tick_params(axis='both', which='minor', labelsize=6)
    if SRA:
        ax2 = fig.add_subplot(gs[0,1])
        rank, SRA = SAD_to_SRA(np.arange(0,maxn),nbrSpecies*probability[:maxn])
        ax2.plot(rank, SRA, color=colour, linewidth=2, zorder=10)
        for k, key in enumerate(approx_dstbn):
            rank, SRA = SAD_to_SRA(np.arange(0,maxn)
                                , nbrSpecies*approx_dstbn[key][:maxn])
            ax2.plot(rank, SRA, color='k', linewidth=2, zorder=10
                            ,linestyle=lstyle[k], label=key)
        ax2.axes.labelsize = size
        ax2.set_xlabel(r'rank',fontsize=size, labelpad=2);
        ax2.set_ylabel(r'abundance', fontsize=size, labelpad=2);
        ax2.set_xlim([0,nbrSpecies]); plt.yscale('log')
        ax2.tick_params(axis='both', which='major', labelsize=size)
    fig.savefig(DIST_DIR + os.sep + "dstbn_" + str(i) + '_' + str(j)+'.pdf')
    plt.close(fig)

    return 0

def SAD_to_SRA(n, SAD):
    rank = np.flip(np.cumsum(SAD))
    SRA = n
    return rank, SRA


def determine_modality( arr, plot=True, revisionmanual=None, sampleP0 = True
                                , nbrSpecies=30, approx_dict={}, SRA=True ):
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
                """
                elif len(max_idx) > 2:
                    if revisionmanual == None:
                        pass
                    else:
                        modality_arr[i,j] = line_names.index('multimodal')
                """
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
                approx_dict_ij = {}
                for key in approx_dict:
                    approx_dict_ij[key] = approx_dict[key][i,j,:]
                plot_prob(arr[i,j,:], i, j, line_colours[int(modality_arr[i,j])]
                                , SRA, approx_dict_ij)
                #plot_prob(arr[i,j,:], i, j, 'red')
    return modality_arr, line_names, line_colours

def determine_bimodality_mf( arr ):
    modality_arr = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    for i in np.arange(np.shape(arr)[0]):
        for j in np.arange(np.shape(arr)[1]):
            max_idx, _ = find_peaks( arr[i,j] )
            if max_idx.tolist() != []:
                modality_arr[i,j] = 1.
    return modality_arr


def heatmap(xrange, yrange, arr, xlabel, ylabel, title, dir, pbtx=10, pbty=20
            , save=False, xlog=True, ylog=True, xdatalim=None, ydatalim=None):
    # TODO : CHANGE AXES

    plt.style.use('src/custom_heatmap.mplstyle')
    #plt.style.use('src/custom.mplstyle')

    vmin = - np.max(np.abs(arr)); vmax = np.max(np.abs(arr))

    imshow_kw = {'cmap': 'YlGnBu', 'aspect': None }
    imshow_kw = {'cmap': 'YlGnBu', 'aspect': None, 'norm' : LogNorm() }
    #imshow_kw = {'cmap': 'RdBu', 'aspect': None, 'vmin' : vmin, 'vmax' : vmax }

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    im = ax.imshow(arr, interpolation='none', **imshow_kw)

    # labels and ticks
    set_axis(ax, plt, pbtx, pbty, xrange, yrange, xlog, ylog
                                        , xdatalim, ydatalim, xlabel, ylabel)

    #plt.xscale('log'); plt.yscale('log')
    plt.colorbar(im,ax=ax,cmap=imshow_kw['cmap'])

    plt.title(title)
    if save:
        fname = ((title.replace(" ","")).replace('/','_')).replace("$","")
        plt.savefig(dir + os.sep + fname + '.pdf');
        #plt.savefig(DIR_OUTPUT + os.sep + fname + '.eps');
        plt.savefig(dir + os.sep + fname + '.png')
    else:
        plt.show()

    plt.close()

    return 0

def convert_npz_mat(filename):
    data = np.load(filename)
    data_dict = {}
    for key in list(data.keys()):
        data_dict[key] = data[key]

    sio.savemat(filename[:-3] + 'mat', mdict=data_dict)
    print('done', filename[:-3] + 'mat')

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
