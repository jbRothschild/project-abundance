import matplotlib as mpl
#mpl.use('Agg') # necessary for creating plots on scinet
import numpy as np; import csv; import os
import matplotlib.pyplot as plt
from gillespie_models import RESULTS_DIR
import matplotlib.patches as patches # histgram animation
import matplotlib.path as path # histogram animation
from matplotlib import animation # animation
from itertools import combinations
from scipy.special import gamma, poch, factorial
import seaborn as sns
import itertools

plt.style.use('parameters.mplstyle')

######################### THEORY FUNCTIONS ########################################

def WKB_trajectory():

    return 0

def WKB_mte():

    return 0

def markov_inverse_trajectory():

    return 0

def markov_inverse_mte():

    return 0

######################### THEORY PLOTS ########################################

def trajectory(fig, ax, **params):

    return 0


################## TYPES OF PLOTS TO USE ###################################

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
    #steady_idx=250000
    print(steady_idx, len(times))
    total_time = times[-1]-times[steady_idx]; # total time in steady state
    time_in_state = times[steady_idx+1:] - times[steady_idx:-1] # array of times in each state

    #  need to weight the histogram by the amount of time they spend in each state (not a simple average)
    steady_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=hist_bins)[0], 1, trajectory[steady_idx:-1,:]) # tricky function that essentially creates an array of histograms (histograms for each moment in time)
    weighted_hist = steady_hist * time_in_state[:,None] /total_time # weight each of these histograms by amount of time spent
    total_indiv = float( int( np.sum(weighted_hist) )); num_species = float(np.shape(trajectory)[1])
    # For now tot_indiv = mean of trajectory
    total_indiv = float( int( np.mean(np.sum(trajectory[-steady_idx:],axis=1)) ) )
    print(total_indiv)

    fig, ax = plt.subplots(figsize=(10,4))

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
    plt.savefig(params['sim_subdir'] +  os.sep + 'ss_abundance' + '.pdf', transparent=True)
    plt.savefig(params['sim_subdir'] +  os.sep + 'ss_abundance' + '.eps')
    plt.show()


def probability_steady_traj(trajectory, times, params, plots):

    return 0

def plot_traj(trajectory, times, params, range, species_list=None, sea_colours = True):
    """
    Plots many ends of trajectories from end to -range

    """
    palette = itertools.cycle(sns.color_palette())
    fig, ax = plt.subplots(figsize=(9,5))

    if species_list == None:
        list = np.arange(0, len(trajectory[0]), 1)
    else:
        list = species_list
    for i in list:
        if sea_colours == True:
            plt.plot(times[-range:],trajectory[-range:,i], color=next(palette))
        else:
            print('hi')
            plt.plot(times[-range:],trajectory[-range:,i])

    ax.set_ylabel(r'Population count'); ax.set_xlabel(r'Time')
    ax.set_xlim(left=times[-range], right=times[-1])

    #plt.xlim(hist_bins[0],hist_bins[-1])
    plt.savefig(params['sim_subdir'] +  os.sep + 'end_trajectory_' + str(range) + '.pdf', transparent=True)
    #plt.savefig(params['sim_subdir'] +  os.sep + 'ss_abundance' + '.eps')
    plt.show()
    #plt.close()

def plot_total_indiv_time(trajectory, times, params, range):
    """
    Plot distribution of J

    Input :
        trajectory : 1 trajectory (nbr_species x nbr_time_points)
        times      : UNNECESSARY
        params     : UNNECESSARY
        range      : What range of late times are we using (since it migth take some
                     time to get to steady state)
    """
    tot_traj = np.sum(trajectory[-range:],axis=1)
    len(tot_traj)
    fig, ax = plt.subplots(figsize=(8,5))
    plt.hist(tot_traj, bins=25, density=True, color='gray', edgecolor='black')
    #ax.set_yscale('log');
    ax.set_ylabel(r'Probability'); ax.set_xlabel(r'Total number of individuals, $J$')
    #ax.set_ylim(bottom=0.001)

    #plt.xlim(hist_bins[0],hist_bins[-1])

    plt.savefig(params['sim_subdir'] +  os.sep + 'dist_J' + '.pdf', transparent=True)
    #plt.savefig(params['sim_subdir'] +  os.sep + 'ss_abundance' + '.eps')
    plt.show()
    #plt.close()

def first_passage_time(trajectory, times, params):
    """
    Plots the distribution of first passage times, from 0 to 0.

    Input :
        trajectory : Using 1 trajectory only
        times      : times from 1 trajectory
    """
    num_species = np.shape(trajectory)[1]
    num_times = np.shape(trajectory)[0]
    fpt_dist = []
    for i in range( num_species ):
        zero = 0; t1 = 0; t2 = 0
        for j in range( num_times ):
            if zero == 0 and trajectory[j,i] == 0:
                zero = 1
                t1 = times[j]
            if zero == 1 and trajectory[j,i] != 0:
                zero = 0; t2 = times[j]
                fpt_dist.append(t2-t1)
    fig, ax = plt.subplots(figsize=(8,5))
    plt.hist(fpt_dist, bins=25, density=True, color='green', edgecolor='black')
    ax.set_yscale('log');
    ax.set_ylabel(r'Probability Distribution'); ax.set_xlabel(r'First Passage Time')
    #ax.set_ylim(bottom=0.001)

    #plt.xlim(hist_bins[0],hist_bins[-1])

    plt.savefig(params['sim_subdir'] + os.sep + 'FPT_distribution' + '.pdf', transparent=True)
    #plt.savefig(params['sim_subdir'] +  os.sep + 'ss_abundance' + '.eps')
    plt.show()

if __name__ == "__main__":

    if CONSOLIDATE:
        if MANY_SIMS:
            if
                function_get_abundance_quantities_of_interest_and_save()

        else:
            sim_num = input("Which number simulation to select from? ") # select directory to use

            while not os.path.exists(os.getcwd() + os.sep + RESULTS_DIR + os.sep + "multiLV" + sim_num + os.sep): # Check if valid simulation
                sim_num = input("Not a valid simulation directory number. Which number simulation to select from? ")




    elif ERGODIC:

        sim_subdir = "multiLV1"
        dict_sim = {}
        # get parameters of simulation
        with open(os.getcwd() + os.sep + RESULTS_DIR + os.sep + sim_subdir + os.sep + "params.csv", newline="") as paramfile:
            reader = csv.reader(paramfile)
            dict_sim = dict(reader)

        # turn certain strings to numbers
        for i in dict_sim:
            if i != 'sim_subdir':
                dict_sim[i] = float(dict_sim[i])

        trajectory = np.loadtxt( dict_sim['sim_subdir'] + os.sep + 'trajectory_%s.txt' %(traj))
        time = np.loadtxt( dict_sim['sim_subdir'] + os.sep + 'trajectory_%s_time.txt' %(traj))

        #abundance_trajectory_animation(trajectory, self.sim_subdir )
        #abundance_steady_trajectory( trajectory, time, dict_sim, {'1' : draw_anton_abund, '2': draw_haeg_abund_1, '3' : draw_det_bal_abund} )
        range_plot = 100000
        #plot_traj(trajectory, time, dict_sim, range_plot)
        #plot_total_indiv_time(trajectory, time, dict_sim, range_plot)
        #abundance_steady_trajectory( trajectory, time, dict_sim, {'1' : draw_anton_abund} )
        #abundance_steady_trajectory( trajectory, time, dict_sim, {'1' : draw_anton_abund, '2': det_bal_abund_mult } )
        abundance_steady_trajectory( trajectory, time, dict_sim, {'1' : draw_det_bal_abund_mult} )
        first_passage_time(trajectory, time, dict_sim)

        #abund_average_trajectory_animation()
        #abund_average_steady_trajectory()
