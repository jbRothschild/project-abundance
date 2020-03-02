import numpy as np; import csv; import os
import matplotlib.pyplot as plt
from gillespie_models import RESULTS_DIR
import matplotlib.patches as patches # histgram animation
import matplotlib.path as path # histogram animation
from matplotlib import animation # animation
from itertools import combinations
from scipy.special import gamma
#import seaborn as sns

plt.style.use('parameters.mplstyle')

######################### THEORY PLOTS ########################################

def anton_abund(fig, ax, x, total_indiv, num_species, **params):
    """
    Using Anton's derivation, Nava corrected the abundance distribution we should get for Moran model with mutations... is that similar to our situation with immigration?
    """
    immi_rate = params['immi_rate']
    birth_rate = params['birth_rate']
    a = 2*immi_rate/birth_rate
    Norm = gamma(num_species*a + 1 + a)/( gamma(num_species*a)*gamma(a) )
    y = (x/total_indiv)**(-1+a)*np.exp(-a*x)/Norm
    ax.plot(x, y, c='c', label='Moran')
    return 0

def haeg_abund_1(fig, ax, x, total_indiv, num_species, **params):
    """
    Plot of Haegeman and Loreau results for comp_overlap = 1.0
    """
    immi_rate = params['immi_rate']; K = params['K']; birth_rate = params['birth_rate']; death_rate = params['death_rate']; comp_overlap = params['comp_overlap']
    N = K # TODO this is WRONG

    # TODO really unclear how they calculate all this
    """
    def species_combinations(J, S):
        for c in combinations(range(J+S-1),S-1):
            yield tuple( b - a - 1 for a, b in zip((-1,) + c, c + (n + m -1,)))
    """
    sum_all_Nj = 0;

    Norm = gamma(num_species*2*immi_rate + 1 + 2*immi_rate)/( gamma(num_species*2*immi_rate)*gamma(2*immi_rate) )
    y = 0*x
    ax.plot(x, y, c='r', label='Haeg. \& Lor.')
    return 0

def Nava_sol_abund(fig, ax, x, total_indiv, num_species, **params):
    """
    Nava has solutions for a variety of plots. Need to calculate these and plot them.
    """
    return 0

def det_bal_abund(fig, ax, x, total_indiv, num_species, **params):
    """
    Plot of the detailed balance scheme described by Sid, for now, fixed number of individuals (roughly K)
    """
    immi_rate = params['immi_rate']; K = params['K']; birth_rate = params['birth_rate']; death_rate = params['death_rate']; comp_overlap = params['comp_overlap']

    N = K # TODO this is WRONG

    def c_k_partial(k, immi_rate, birth_rate, death_rate, N, K, comp_overlap ):
        value = 1.0
        if k == 0.0:
            return value
        else:
            for i in np.arange(1,k+1):
                value = value*( ( immi_rate + birth_rate*(i-1))/( i*(death_rate + (birth_rate-death_rate)*(i*(1-comp_overlap)+comp_overlap*N)/K) ) )
            return value

    Norm = N/(np.sum([ i*c_k_partial(i, immi_rate, birth_rate, death_rate, N, K, comp_overlap) for i in np.arange(1,N+1)]))

    y = [ c_k_partial(i, immi_rate, birth_rate, death_rate, N, K, comp_overlap)*Norm for i in x]

    ax.plot(x, y, c='b', label='Kolmogorov eq.')

    return 0

################## TYPES OF PLOTS TO USE ###################################

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
        top = bottom + n
        verts[1::5,1] = top; verts[2::5,1] = top;
        return [patch, ]

    fig, ax = plt.subplots();
    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(barpath, facecolor='gray', edgecolor='black', alpha=0.5)
    ax.add_patch(patch)
    ax.set_xlim(left[0], right[-1]); ax.set_yscale('log'); # ax.set_ylim(bottom.min(), top.max()/4);
    # TODO only plot certain frames
    ani = animation.FuncAnimation(fig, animate, fargs=(maximum,trajectory), frames=600, repeat=False, blit=True)
    anim.save(save_dir + os.sep + 'abundance.gif', writer='imagemagick', fps=60)
    plt.close()

def abundance_steady_trajectory(trajectory, times, params, plots):
    """
    function that plots the steady state abundance distribution with various theoretical curves overlapped
    """
    # create bins
    maximum = int(np.max(trajectory)) + int(0.1*np.max(trajectory) ); hist_bins = np.linspace(0, maximum+1, maximum+2)
    #find index of time for which steady state s achieved
    """
    for i, t in enumerate(times):
        if i == len(times)-1:
            print('Never reached steady state: should run for longer or change range of acceptance.')
            exit()
        j = len(times)-(i+1)
        if np.sum( np.var(trajectory[j:,:], axis=0) ) > 1.0 * np.shape(trajectory)[1]: # some measure of steady state, weird
            if j + 10*np.shape(trajectory)[1] > len(times): # too close to end of simulation
                print('Not enough data points for steady state: should run for longer or change range of acceptance')
            steady_idx = j
            break
    """
    steady_idx=500000
    print(steady_idx, len(times))
    total_time = times[-1]-times[steady_idx]; # total time in steady state
    time_in_state = times[steady_idx+1:] - times[steady_idx:-1] # array of times in each state

    #  need to weight the histogram by the amount of time they spend in each state (not a simple average)
    steady_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=hist_bins)[0], 1, trajectory[steady_idx:-1,:]) # tricky function that essentially creates an array of histograms (histograms for each moment in time)
    weighted_hist = steady_hist * time_in_state[:,None] /total_time # weight each of these histograms by amount of time spent
    total_indiv = float(int( np.sum(weighted_hist) )); num_species = float(np.shape(trajectory)[1])

    fig, ax = plt.subplots()

    ax.bar(hist_bins[:-1], np.sum(weighted_hist,axis=0), width = 1, color='gray', edgecolor='black', alpha=0.5)
    ax.set_yscale('log'); ax.set_ylabel(r'species abundance, $A(k)$'); ax.set_xlabel(r'number of individuals, $k$')
    ax.set_ylim(bottom=0.001)
    if plots:
        for i in plots:
            plots[i](fig, ax, hist_bins, total_indiv, num_species, **params)
    else:
        pass

    plt.xlim(hist_bins[0],hist_bins[-1])
    ax.legend()
    plt.savefig(params['sim_dir'] + os.sep + 'figures' + os.sep + 'ss_abundance' + '.pdf', transparent=True)
    plt.savefig(params['sim_dir'] + os.sep + 'figures' + os.sep + 'ss_abundance' + '.eps')
    plt.show()

def probability_steady_traj(trajectory, times, params, plots):

    return 0


if __name__ == "__main__":

    sim_dir = "multiLV1"
    dict_sim = {}
    # get parameters of simulation
    with open(os.getcwd() + os.sep + RESULTS_DIR + os.sep + sim_dir + os.sep + "params.csv", newline="") as paramfile:
        reader = csv.reader(paramfile)
        dict_sim = dict(reader)

    # turn certain strings to numbers
    for i in dict_sim:
        if i != 'sim_dir':
            dict_sim[i] = float(dict_sim[i])

    traj = 0

    if traj != None:
        trajectory = np.loadtxt( dict_sim['sim_dir'] + os.sep + 'trajectory_%s.txt' %(traj))
        time = np.loadtxt( dict_sim['sim_dir'] + os.sep + 'trajectory_%s_time.txt' %(traj))

        #abundance_trajectory_animation(trajectory, self.sim_dir )
        abundance_steady_trajectory( trajectory, time, dict_sim, {'1' : anton_abund, '2': haeg_abund_1, '3' : det_bal_abund} )

        #abund_average_trajectory_animation()
        #abund_average_steady_trajectory()
