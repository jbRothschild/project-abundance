import numpy as np
import matplotlib.pyplot as plt; from matplotlib import colors, ticker
import argparse, os, pathlib, sys
from scipy.signal import savgol_filter
from src.consolidate import make_ordered_2Darray

from default import DATA_FOLDER

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

    plt.xlabel(xlabel); plt.ylabel(ylabel)

    return ax, plt

def plot_abundance(param, colour='r',show=True):
    # TODO : just pass param, then plot what you need. saving params in title of
    #       figure could be useful.
    f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()
    binCentres = param['binCentres']; countsNorm = param['countsNorm']
    plt.hist(binCentres, weights=countsNorm, bins=len(binCentres),color=colour)
    windowSize = 4*int(len(binCentres)/10)+3;
    polynomial = 5
    smoothedFcn = savgol_filter(countsNorm, windowSize, polynomial)
    #plt.plot(binCentres, smoothedFcn, 'r')
    #sns.histplot(trajectory, bins=nbins, stat='probability', kde=True)
    plt.yscale('log'); #plt.xlim((0.0,trajMax))
    plt.title(r'$\rho$ : ' + str(param['compOverlap']) + r', $\mu$ : '
                    + str(param['immiRate']) )
    if show: plt.show()

    return f

def plot_modality( dir, parameter1, parameter2, show=True ):
    results = make_ordered_2Darray( dir, parameter1, parameter2)

    # array that will hold 1, 2, 3
    modality = np.zeros(( len(results[parameter1]), len(results[parameter2]) ))

    for i in np.arange( len(results[parameter1]) ):
        for j in np.arange( len(results[parameter2]) ):
            abundance = results['countsNorm'][i][j]
            if abundance[0] >= abundance[1]:
                modality[i,j] = 1.5
                if np.argmax(abundance[5:]) > 5:
                    modality[i,j] += 1.0
            else:
                modality[i,j] = 0.5

    bounds      = [0,1,2,3]
    lineNames   = ['bimodal','unimodal neutral','unimodal niche']
    lineColours = ['slateblue','khaki','mediumturquoise']
    lineCentres = [0.5, 1.5, 2.5]

    f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()

    # plots
    cmap = colors.ListedColormap( lineColours )
    norm = colors.BoundaryNorm( bounds, cmap.N )

    im = plt.imshow(modality.T, cmap=cmap, norm=norm, aspect='auto')

    POINTS_BETWEEN_X_TICKS = 20; POINTS_BETWEEN_Y_TICKS = 20
    xlog = True; ylog = True;
    set_axis(ax, plt, POINTS_BETWEEN_X_TICKS, POINTS_BETWEEN_Y_TICKS
                    , results[parameter1], results[parameter2]
                    , xlog, ylog, None, None, parameter1, parameter2)

    if show: plt.show()

    return f


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        description = "Plotting results from a simulation.")
    parser.add_argument('-d', '--directory', type = str, default = 'default'
                        , nargs = '?', help = "Directory to save sim in.")
    parser.add_argument('--id', help='identification')
    parser.add_argument('-s','--save', dest='save', action='store_true'
                        , default=False, required=False, help = "Use to save.")

    args = parser.parse_args()

    simDir = DATA_FOLDER + os.sep + args.directory

    if args.id is None:
        plot_modality( simDir, 'immiRate', 'compOverlap')
    if args.id is not None:
        file = simDir + os.sep + 'sim' + str(args.id) + "_results" + '.npy'
        param = np.load(file, allow_pickle=True)[()]
        plot_abundance(param)
