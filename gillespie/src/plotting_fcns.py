import os
import numpy as np
from scipy.signal import savgol_filter, find_peaks, argrelextrema
from src.manual_revision import DICT_REVISION
from src.settings import VAR_NAME_DICT
import matplotlib.pyplot as plt; from matplotlib.colors import LogNorm

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
    if title == 'Local maxima':
        cbar = plt.colorbar(im,ax=ax,cmap=imshow_kw['cmap']
                        , boundaries=[-0.5,0.5,1.5,2.5]
                        , ticks=[-0.5,0.5,1.5,2.5])
        cbar.ax.set_yticklabels(['Max 0', 'Max n', '2 Maxs', ''])
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
