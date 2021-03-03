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

#plt.style.use('parameters.mplstyle')
plt.style.use('custom.mplstyle')

######################### THEORY FUNCTIONS ########################################

def moranton_abund_old(x, num_species, birth_rate, immi_rate):
    a = 2*immi_rate/birth_rat
    Norm = gamma(num_species*a + 1 + a)/( gamma(num_species*a)*gamma(a)*num_species )
    return (x/total_indiv)**(-1+a)*np.exp(-a*x)/Norm

def moranton_abund(x, num_species, birth_rate, immi_rate):
    a = total_indiv*immi_rate/(birth_rate*total_indiv+immi_rate*num_species)
    #a = immi_rate/birth_rate
    #total_indiv = 1
    Norm = gamma(num_species*a)*gamma(a)/( gamma((num_species+1)*a+1) )
    return (x/total_indiv)**(-1+a)*np.exp(-a*x)/Norm

def sid_abund(x, immi_rate, birth_rate, death_rate, N, K, comp_overlap):
    N = total_indiv # TODO this is WRONG
    print(total_indiv,num_species)

    def c_k_partial( k, immi_rate, birth_rate, death_rate, N, K, comp_overlap ):
        value = 1.0
        if k == 0.0:
            return value
        else:
            """ previous calc.
            for i in np.arange(1,k+1):
                value = value*( ( immi_rate + birth_rate*(i-1))/( i*(death_rate + (birth_rate-death_rate)*(i*(1-comp_overlap)+comp_overlap*N)/K) ) )
            """
            c = ( death_rate*K/((birth_rate-death_rate)) + comp_overlap*N ) / ( 1-comp_overlap )
            value = ( birth_rate*K / ( (birth_rate - death_rate)*(1-comp_overlap) ) ) ** k * poch( immi_rate/birth_rate, k ) / ( factorial(k)*poch( c + 1, k ) )
            return value

    Norm = N/(np.sum([ i*c_k_partial(i, immi_rate, birth_rate, death_rate, N, K, comp_overlap) for i in np.arange(1,N+1)]))

    return [ c_k_partial(i, immi_rate, birth_rate, death_rate, N, K, comp_overlap)*Norm for i in x]

def mte_1d(n, immi_rate, birth_rate, death_rate, N, K, comp_overlap):
    """
    1d mean time to extinction, which is relevant if the flutations of N (J) are negligeable
    """

    def birth(i, immi_rate, birth_rate, death_rate, N, K, comp_overlap):
        quadratic = 0.0
        return i * ( birth_rate - quadratic*(i + comp_overlap*N)/K ) + immi_rate

    def death(i, immi_rate, birth_rate, death_rate, N, K, comp_overlap):
        quadratic = 0.0
        emmi_rate = 0.0
        return i * ( death_rate + emmi_rate + ( birth_rate - death_rate )*( 1.0 - quadratic )*(i + comp_overlap*N)/K )

    def R(i, immi_rate, birth_rate, death_rate, N, K, comp_overlap):
        if i == 1:
            return 1.0
        else:
            return ( birth_rate*K )**(i-1) * poch( immi_rate/birth_rate + 1 , i - 1 ) / ( fact(i-1)*poch(death_rate*K + comp_overlap*N + 1, i-1) )

    def T(i, immi_rate, birth_rate, death_rate, N, K, comp_overlap):
        return death(1)*R(i+1, immi_rate, birth_rate, death_rate, N, K, comp_overlap)/birth(i, immi_rate, birth_rate, death_rate, N, K, comp_overlap)

    return np.array( [ ( 1/death(1) )*np.sum( [ ( 1/R(j) )*np.sum( [T(k) for k in np.arange(j, N+1) ] ) for j in np.arange(1,i+1) ] ) for i in n ] )


######################### THEORY PLOTS ########################################

def draw_anton_abund_old(fig, ax, x, total_indiv, num_species, **params):
    """
    Using Anton's derivation, Nava corrected the abundance distribution we should get for Moran model with mutations... is that similar to our situation with immigration?
    """
    immi_rate = params['immi_rate']
    birth_rate = params['birth_rate']
    y = moranton_abund_old(x, num_species, birth_rate, immi_rate)
    ax.plot(x, y, c='c', label='Moran')
    return 0

def draw_anton_abund(fig, ax, x, total_indiv, num_species, **params):
    """
    Using Anton's derivation, Nava corrected the abundance distribution we should get for Moran model with mutations... is that similar to our situation with immigration?
    """
    immi_rate = params['immi_rate']
    birth_rate = params['birth_rate']
    y = moranton_abund(x, num_species, birth_rate, immi_rate)
    ax.plot(x, y, c='c', label='Moran')
    return 0

def draw_haeg_abund_1(fig, ax, x, total_indiv, num_species, **params):
    """
    Plot of Haegeman and Loreau results for comp_overlap = 1.0. Deprecated
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


def draw_det_bal_abund(fig, ax, x, total_indiv, num_species, **params):
    """
    Plot of the detailed balance scheme described by Sid, for now, fixed number of individuals (roughly K)
    """
    immi_rate = params['immi_rate']; K = params['K']; birth_rate = params['birth_rate']; death_rate = params['death_rate']; comp_overlap = params['comp_overlap']

    N = total_indiv # TODO this is WRONG

    y = sid_abund(x, immi_rate, birth_rate, death_rate, N, K, comp_overlap)

    ax.plot(x, y, c='b', label='Kolmogorov eq.')

    return 0

def draw_det_bal_abund_mult(fig, ax, x, total_indiv, num_species, **params):
    """
    Plot of the detailed balance scheme described by Sid, for now, fixed number of individuals
    """
    immi_rate = params['immi_rate']; K = params['K']; birth_rate = params['birth_rate']; death_rate = params['death_rate']; comp_overlap = params['comp_overlap']

    N = total_indiv # TODO this is WRONG
    print(total_indiv,num_species)

    P = np.zeros(len(x))

    for i, prob in enumerate(P):
        if i==0:
            P[i] = 1.0
        else:
            P[i] = ( birth_rate*(i-1) + immi_rate )*P[i-1] / ( i * (death_rate + (birth_rate-death_rate)*(1-comp_overlap)*i/K + comp_overlap*(birth_rate-death_rate)*N/K ) )


    #Norm = num_species/(np.sum( [P[i] for i in np.arange(0,len(x))] ))
    Norm = N/(np.sum( [i*P[i] for i in np.arange(0,len(x))] ))

    y = P*Norm

    ax.plot(x, y, c='b', label='Kolmogorov eq.')

    return 0

def draw_MTE_1D(fig, ax, n, total_indiv, num_species, **params):
    """
    Plot of the mean time to extinction as a function of
    """
    immi_rate = params['immi_rate']; K = params['K']; birth_rate = params['birth_rate']; death_rate = params['death_rate']; comp_overlap = params['comp_overlap']

    N = total_indiv # TODO this is WRONG
    print(total_indiv,num_species)

    for i, prob in enumerate(P):



    #Norm = num_species/(np.sum( [P[i] for i in np.arange(0,len(x))] ))
    Norm = N/(np.sum( [i*P[i] for i in np.arange(0,len(x))] ))

    y = P*Norm

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
    Plots the end of a trajectory from end to -range
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

    ax.set_ylabel(r'Population count, $n$'); ax.set_xlabel(r'Time')
    ax.set_xlim(left=times[-range], right=times[-1])

    #plt.xlim(hist_bins[0],hist_bins[-1])
    plt.savefig(params['sim_subdir'] +  os.sep + 'end_trajectory_' + str(range) + '.pdf', transparent=True)
    #plt.savefig(params['sim_subdir'] +  os.sep + 'ss_abundance' + '.eps')
    plt.show()
    #plt.close()

def plot_total_indiv_time(trajectory, times, params, range):
    """
    Plot distribution of J
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
    Plots the distribution of first passage times.
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

    CONSOLIDATE = True
    MANY_SIMS = False
    ERGODIC = True

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
