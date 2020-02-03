import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # histgram animation
import matplotlib.path as path # histogram animation
from matplotlib import animation # animation
#import seaborn as sns

plt.style.use('parameters.mplstyle')

def plot1(fig, ax, x, **params):
    y = x*0 +1
    ax.plot(x,y)
    return 0

def plot2(fig, ax, x, **params):
    return 0

def abundance_trajectory_animation(trajectory, save_dir):
    """
    Make a gif of the abundance of species along one trajectory

    Returns:
        gif of histogram.
    """

    # Makes limits for histogram plot
    maximum = np.max(trajectory)+int(0.1*np.max(trajectory))
    bins = np.linspace(0, maximum+1, num=maximum+2);
    left = np.array(bins[:-1]); right = np.array(bins[1:]);
    bottom = np.zeros(len(left)); top = bottom + len(trajectory[0,:]); nrects = len(left);

    # magic mumbo jumbo
    nverts = nrects * (1 + 3 + 1); verts = np.zeros( (nverts, 2) ); codes =  np.ones(nverts,int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO; codes[4::5] = path.Path.CLOSEPOLY;
    verts[0::5,0] = left; verts[0::5,1] = bottom; verts[1::5,0] = left; verts[1::5,1] = top;
    verts[2::5,0] = right; verts[2::5,1] = top; verts[3::5,0] = right; verts[3::5,1] = bottom;

    patch = None; # need to use a patch to be able to update histogram

    def animate(i, maximum, trajectory):
        bins = np.linspace(0, maximum+1, num=maximum+2);
        n, bins = np.histogram(trajectory[i, :], bins=bins)
        print(n)
        top = bottom + n
        verts[1::5,1] = top; verts[2::5,1] = top;
        return [patch, ]

    fig, ax = plt.subplots();
    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(barpath, facecolor='gray', edgecolor='black', alpha=0.5)
    ax.add_patch(patch)
    ax.set_xlim(left[0], right[-1]); ax.set_ylim(bottom.min(), top.max()/4);
    ani = animation.FuncAnimation(fig, animate, fargs=(maximum,trajectory), frames=600, repeat=False, blit=True)
    anim.save(save_dir + os.sep + 'abundance.gif', writer='imagemagick', fps=60)
    plt.close()

def abundance_steady_trajectory(trajectory, times, params, kwargs):
    """

    """
    # create bins
    maximum = np.max(trajectory) + int(0.1*np.max(trajectory) ); hist_bins = np.linspace(0, maximum+1, maximum+2)
    #find index of time for which steady state s achieved
    for i, t in enumerate(times):
        if i == len(times)-1:
            print('Never reached steady state: should run for longer or change range of acceptance.')
            exit()
        j = len(times)-(i+1)
        if np.sum( np.var(trajectory[j:,:], axis=0) ) > 0.5 * np.shape(trajectory)[1]: # some measure of steady state, weird
            if j + 10*np.shape(trajectory)[1] > len(times): # too close to end of simulation
                print('Not enough data points for steady state: should run for longer or change range of acceptance')
            steady_idx = j
            break

    total_time = times[-1]-times[steady_idx]; # total time in steady state
    time_in_state = times[steady_idx+1:] - times[steady_idx:-1] # array of times in each state

    #  need to weight the histogram by the amount of time they spend in each state (not a simple average)
    steady_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=hist_bins)[0], 1, trajectory[steady_idx:-1,:]) # tricky function that essentially creates an array of histograms (histograms for each moment in time)
    weighted_hist = steady_hist * time_in_state[:,None] /total_time # weight each of these histograms by amount of time spent

    fig, ax = plt.subplots()

    ax.bar(hist_bins[:-1], np.sum(weighted_hist,axis=0), width = 1, color='gray', edgecolor='black', alpha=0.5)
    if kwargs:
        for i in kwargs:
            kwargs[i](fig, ax, hist_bins, **params)
    else:
        pass

    plt.xlim(hist_bins[0],hist_bins[-1])
    plt.savefig(params['sim_dir'] + os.sep + 'figures' + os.sep + 'ss_abundance' + '.pdf', transparent=True)
    plt.savefig(params['sim_dir'] + os.sep + 'figures' + os.sep + 'ss_abundance' + '.eps')
    plt.show()
