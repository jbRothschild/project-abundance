import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import meshgrid
import numpy as np
import os
import matplotlib.ticker as ticker
from decimal import Decimal
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('parameters.mplstyle')

FS = 20
POINTS_BETWEEN_X_TICKS = 33
POINTS_BETWEEN_Y_TICKS = 25


def fixed_points(alpha, numspecies, K, r, mu):
    return K * ( 1 + np.sqrt( 1 + 4*( alpha*( numspecies-1 ) + 1  )*mu/(K*r) ) ) / ( 2*( alpha*( numspecies-1 ) + 1 ) )

def eigen_ratio(alpha, numspecies, K, r, mu):
    lambda1 = (K - 2 * fixed_points(alpha, numspecies, K, r, mu) * ( 1 + alpha*(numspecies - 1 ) ) )
    lambdai = ( K - fixed_points(alpha, numspecies, K, r, mu) * ( 2 + alpha*(numspecies - 2 ) ) )
    return  lambda1/lambdai , lambda1, lambdai

def plot_heatmap(arr, xrange, yrange, fname, label, show=True, save=True, **kwargs):
    """
    xrange: range of values for y
    yrange: range of x values
    fname: file saved as pdf and eps format with this name
    show: shows plots
    save: saves plots
    kwargs: a variety of keyword args are used below. They are mostly used for contour plot lines if I understand correctly. Using Duncan's default ones mostly, but we can talk about it.
    """

    if 'vmin' in kwargs.keys(): vmin = kwargs['vmin']
    else: vmin = np.min(arr)

    if 'vmax' in kwargs.keys(): vmax = kwargs['vmax']
    else: vmax = np.max(arr)

    if 'fmt' in kwargs.keys(): fmt = kwargs['fmt']
    else: fmt = ticker.LogFormatterMathtext()

    imshow_kw = {'cmap': 'YlGnBu', 'aspect': None, 'vmin': vmin, 'vmax': vmax}
    imshow_kw = {'cmap': 'YlGnBu', 'aspect': None, 'vmin': vmin, 'vmax': vmax, 'norm': mpl.colors.LogNorm(vmin,vmax)}

    # TODO change colour scheme, see https://matplotlib.org/examples/color/colormaps_reference.html
    """
    Colours viridis, YlGnBu, terrain, plasma
    """
    #print 'arr limits:', np.min(arr), np.max(arr)
    # plot setup
    f = plt.figure()
    im = plt.imshow(arr, interpolation='spline36', **imshow_kw)

    # axes setup
    fig = plt.gcf(); ax = plt.gca()

    # method 1
    ax.set_xticks([i for i, xval in enumerate(xrange) if i % POINTS_BETWEEN_X_TICKS == 0])
    ax.set_yticks([i for i, kval in enumerate(yrange) if i % POINTS_BETWEEN_Y_TICKS == 0])
    ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval) for i, xval in enumerate(xrange) if i % POINTS_BETWEEN_X_TICKS==0], fontsize=FS)
    ax.set_yticklabels([str(int(yval)) for i, yval in enumerate(yrange) if i % POINTS_BETWEEN_Y_TICKS==0], fontsize=FS)

    ax.invert_yaxis()
    ax.set_xlabel(label[0], fontsize=FS); ax.set_ylabel(label[1], fontsize=FS)

    # create colorbar
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)# Make colorbar fixed size
    cbar = fig.colorbar(im, cax=cax)
    #cbar.locator = ticker.LogLocator(base=10)
    cbar.ax.set_ylabel(label[2], rotation=-90, va="bottom", fontsize=FS, labelpad=20); cbar.ax.tick_params(labelsize=FS)
    cbar.ax.minorticks_off();
    # UNCOMMENT THIS ONLY WHEN TICKS DON'T APPEAR
    #cbar.set_ticks([round(vmin,3)+0.001,round(vmax,3)-0.001])
    cbar.update_ticks()

    # save
    if save == True:
        plt.savefig(os.getcwd() + os.sep + "figures" + os.sep + fname + '.pdf', transparent=True)
        #plt.savefig(os.getcwd() + os.sep + dict_dir['name'] + '_phase_diagram_' + str(numspecies) + 'species_alpha' + str(alpha) + '.eps')
    if show:
        plt.show()
    plt.close()

    return fig, ax

if __name__ == '__main__':

    if not os.path.isdir(os.getcwd() + os.sep + "figures"):
        os.makedirs(os.getcwd() + os.sep + "figures")

    # FIGURE 1 FIXED POINTS
    K = 100; r = 10; mu = 0.25;
    alpha =  np.logspace(-3,0,100); S = np.linspace(1,101,101);
    X,Y = meshgrid(alpha,S)

    Z1 = fixed_points(X, Y, K, r, mu)

    #plot_heatmap(Z, alpha, S, 'fixed_points', [r'$\alpha$',r'S',r'Fixed point'], show=True, save=True)

    #FIGURE 2 EIGENVALUE RATIO
    Z2, l1, l2 = eigen_ratio(X, Y, K, r, mu)
    print(l1, l2)
    plot_heatmap(Z2, alpha, S, 'ratio_eigenvalues', [r'$\alpha$',r'S',r'$\lambda_1 / \lambda_i$'], show=True, save=True)
