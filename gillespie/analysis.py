import os, glob, csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import argrelextrema

from gillespie_models import RESULTS_DIR, MultiLV, SIR
import theory_equations as theqs
from settings import VAR_NAME_DICT, COLOURS, IMSHOW_KW

SIM_FIG_DIR = 'figures' + os.sep + 'simulations'
while not os.path.exists( os.getcwd() + os.sep + SIM_FIG_DIR ):
    os.makedirs(os.getcwd() + os.sep + SIM_FIG_DIR);

import pickle

plt.style.use('custom.mplstyle')

def consolidate_trajectories(sim_dir, save_file=False, FORCE_NUMBER=3000):
    """
    Put all trajectories into 1 huge array. Hopefully easier to manipulate then.

    Input :
        sim_dir (string) : diretory to go through all the time files
        plot (binary)    : whether or not to plot this Distribution
        FORCE_NUMBER     : Maximum number of trajectories

    Return :
        all_traj (array) : all trajectories ( num_traj x length_traj
                                              x num_species )
        times (array)    : array of times (note that some early times might
                           be zero)
    """

    num_traj = 0
    for subdir, dirs, files in os.walk(sim_dir):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith("time.txt"):
                num_traj += 1

    # TODO : Change params-csv to loading pickle
    # with open('results_%s.pickle' % i, 'rb') as handle:
    #    param_dict = pickle.load(handle)
    #

    file = sim_dir + os.sep + 'params.csv'
    reader = csv.reader(open(file))
    param_dict = {}
    for row in reader:
        key = row[0]
        if key in param_dict:
            # implement your duplicate row handling here
            pass
        param_dict[key] = row[1:]

    if num_traj > FORCE_NUMBER:
        num_traj = FORCE_NUMBER
    # Load all the trajectories into one array
    times = np.zeros( (num_traj, int(param_dict['max_gen_save'][0])) )
    all_traj = np.zeros( (num_traj, int(param_dict['max_gen_save'][0]),
                         int(param_dict['nbr_states'][0])) ) ; i = 0;

    for subdir, dirs, files in os.walk(sim_dir):
        for filename in files:
            filepath = subdir + os.sep + filename
            if ( ('trajectory' in filepath) and
                 ( not filepath.endswith("time.txt") ) and i<FORCE_NUMBER):
                all_traj[i, :, :] = np.loadtxt(filepath)
                times[i,:] = np.loadtxt(filepath[:-4]+'_time.txt')
                i += 1;

    return all_traj, times

def fpt_distribution(sim_dir, plot = True):
    """
    Assuming the trajectories all end with a fpt event, this will output all the
    fpt

    Input :
        sim_dir (string) : diretory to go through all the time files
        plot (binary)    : whether or not to plot this distribution

    Return :
        fpt (array) : array of first passage times

    """
    # count number of trajectories to use in all subfolders
    num_traj = 0
    for subdir, dirs, files in os.walk(sim_dir):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith("time.txt"):
                num_traj += 1

    fpt = np.zeros(num_traj); i = 0;
    for subdir, dirs, files in os.walk(sim_dir):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith("time.txt"):
                with open(filepath, 'r') as f:
                    lines = f.read().splitlines()
                    fpt[i] = lines[-1]; i += 1;

    if plot:
        kwargs = dict( alpha=0.3, density=True, histtype='stepfilled',
                       color='steelblue');
        # TODO : np.histogram(x,nbins)
        plt.hist( fpt, bins=100, **kwargs );
        plt.axvline( fpt.mean(), color='k', linestyle='dashed', linewidth=1 );
        plt.xlim(left=0.0)
        min_ylim, max_ylim = plt.ylim()
        min_xlim, max_xlim = plt.xlim()
        plt.text( fpt.mean()*1.1, max_ylim*0.9,
                  'Meanrichness: {:.2f}'.format(fpt.mean()) )
        plt.yscale('log')
        plt.ylabel(r'probability')
        plt.xlabel(r'time (gen.)')
        plt.show()

    return fpt

### MULTILV

def mlv_extract_results_sim(dir, sim_nbr=1):
    """
    Analyze information collected in many results_(sim_nbr).pickle, output to
    other functions as arrays.

    Input :
        dir     : directory that we're extracting
        sim_nbr : simulation number (subdir sim%i %(sim_nbr))

    Output :
        param_dict          :
        ss_dist             :
        richness_dist       :
        time_btwn_ext       :
        mean_pop            :
        mean_rich           :
        mean_time_present   :
        P0                  :
        nbr_local_max       :
        H                   :
        GS                  :

    """
    # TODO QUICK FIX
    while not os.path.exists(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
               'results_0.pickle'):
        sim_nbr += 1

    with open(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
               'results_0.pickle', 'rb') as handle:
        param_dict = pickle.load(handle)

    model = MultiLV(**param_dict)

    # distribution
    if 'ss_distribution' in model.results:
        ss_dist     = model.results['ss_distribution'] \
                              / np.sum(model.results['ss_distribution'])
        mean_pop    = np.dot(ss_dist, np.arange(len(ss_dist)))
        P0          = ss_dist[0]
        nbr_local_max = np.min([len( argrelextrema(ss_dist, np.greater) ),2])
        H           = -np.dot(ss_dist[ss_dist>0.0],np.log(ss_dist[ss_dist>0.0]))
        GS          = 1.0 - np.dot(ss_dist,ss_dist)
    else: ss_dist, mean_pop, P0, nbr_local_max, H, GS = None, None, None, None \
                                                          , None, None

    # richness
    if 'richness' in model.results:
        richness_dist =  model.results['richness']
        mean_rich     = np.dot(model.results['richness']
                           , np.arange(len(model.results['richness'])))
    else: richness_dist, mean_rich = None, None

    # time
    if 'time_btwn_ext' in model.results:
        time_btwn_ext = model.results['time_btwn_ext'];
        if time_btwn_ext != []:
            mean_time_present = np.mean(model.results['time_btwn_ext']);
        else:
            mean_time_present = np.nan
    else: time_btwn_ext, mean_time_present = None, None

    # Change to dictionary
    return param_dict, ss_dist, richness_dist, time_btwn_ext, mean_pop\
                     , mean_rich, mean_time_present, P0, nbr_local_max, H, GS\
                     , param_dict['nbr_species']



def mlv_plot_single_sim_results(dir, sim_nbr = 1):
    """
    Plot information collected in a single results_(sim_nbr).pickle

    Input :
        dir     : directory that we're plotting from
        sim_nbr : simulation number (subdir sim%i %(sim_nbr))

    Output :
        Plots of a single simulation
    """
    # replace with a dict
    param_dict, ss_dist_sim, richness_sim, time_present_sim, mean_pop_sim\
              , mean_rich_sim, mean_time_present_sim, _, _, _, _, _  \
              = mlv_extract_results_sim(dir, sim_nbr=sim_nbr)

    print(param_dict)

    # theory equations
    theory_models   = theqs.Model_MultiLVim(**param_dict)
    ss_dist_thry, _ = theory_models.abund_1spec_MSLV()
    #theory_dist, theory_abund = theory_models.abund_sid()

    fig = plt.figure()
    plt.scatter(np.arange(len(richness_sim)), richness_sim, color='b')
    plt.ylabel(r"probability of richness")
    plt.xlabel(r'richness')
    plt.axvline( mean_rich_sim, color='k' , linestyle='dashed', linewidth=1)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.show()

    ## dstbn present
    if time_present_sim != []:
        nbins   = 100
        logbins = np.logspace(np.log10(np.min(time_present_sim))
                              , np.log10(np.max(time_present_sim)), nbins)
        counts, bin_edges = np.histogram(time_present_sim, density=True
        #                                , bins=logbins)
                                         , bins = nbins)
        fig  = plt.figure()
        axes = plt.gca()
        plt.scatter((bin_edges[1:]+bin_edges[:-1])/2,
                 counts, color='g')
        plt.axvline( mean_time_present_sim, color='k', linestyle='dashed'
                    , linewidth=1 ) # mean
        plt.ylabel(r"probability of time present")
        plt.xlabel(r'time present between extinction')
        plt.yscale('log')
        axes.set_ylim([np.min(counts[counts!=0.0]),2*np.max(counts)])
        #plt.xscale('log')
        plt.show()

    ## ss_dstbn (compare with deterministic mean)
    fig  = plt.figure()
    axes = plt.gca()

    plt.scatter(np.arange(len(ss_dist_sim)),ss_dist_sim,label='simulation')
    plt.plot(np.arange(len(ss_dist_thry)),ss_dist_thry,label='theory')
    plt.ylabel(r"probability distribution function")
    plt.xlabel(r'n')
    plt.axvline( mean_pop_sim, color='k' , linestyle='dashed'
                , linewidth=1 ) #mean
    #plt.axvline( theory_models.deterministic_mean(), color='k' ,
    #            linestyle='dashdot', linewidth=1 ) #mean
    plt.yscale('log')
    axes.set_ylim([np.min(ss_dist_sim[ss_dist_sim!=0.0]),2*np.max(ss_dist_sim)])
    axes.set_xlim([0.0,np.max(np.nonzero(ss_dist_sim))])
    plt.legend(loc='best')
    #plt.xscale('log')
    plt.show()

    return 0

def mlv_consolidate_sim_results(dir, parameter1, parameter2=None):
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

    filename =  dir + os.sep + 'consolidated_results.npz'

    # count number of subdirectories
    nbr_sims = len( next( os.walk(dir) )[1] )

    # initialize the
    param1      = np.zeros(nbr_sims); mean_pop          = np.zeros(nbr_sims)
    mean_rich   = np.zeros(nbr_sims); mean_time_present = np.zeros(nbr_sims)
    P0          = np.zeros(nbr_sims); nbr_local_max     = np.zeros(nbr_sims)
    H           = np.zeros(nbr_sims); GS                = np.zeros(nbr_sims)
    nbr_species = np.zeros(nbr_sims); ss_dist_vary      = []

    # if 2 parameters vary in the simulation
    if parameter2 != None: param2 = np.zeros(nbr_sims)

    #
    for i in np.arange(nbr_sims):
        param_dict, ss_dist_sim, _, _, mean_pop[i], mean_rich[i], mean_time_present[i]\
                  , P0[i], nbr_local_max[i], H[i], GS[i], nbr_species[i]\
                  = mlv_extract_results_sim(dir, sim_nbr = i+1)
        # sims might not have same distribution length
        ss_dist_vary.append(np.array(ss_dist_sim))

        # Value of parameters
        param1[i] = param_dict[parameter1]
        if parameter2 != None: param2[i] = param_dict[parameter2]

    # making all sims have same distribution length
    length_longest_dstbn = len(max(ss_dist,key=len))
    ss_dist = np.zeros((nbr_sims,length_longest_dstbn))
    for i in np.arange(nbr_sims):
        ss_dist[i,:len(ss_dist_vary)] = ss_dist_vary[i]

    # Single parameter changing
    if parameter2 == None:
        if len(np.unique(param1)) == 1:
            print('Warning : ' + parameter1 + ' parameter does not vary!')
            raise SystemExit
        elif len(np.unique(param1)) != len(param1):
            print('Warning : ' + parameter1 +
                    ' is repeated here, hence there might be another param!')



        # save arrays in a npz file, dict here
        dict_arrays = {  parameter1 : param1, 'mean_pop'  : mean_pop
                                        , 'mean_rich'     : mean_rich
                                        , 'mean_time_present' : mean_time_present
                                        , 'P0'            : P0
                                        , 'nbr_local_max' : nbr_local_max
                                        , 'entropy'       : H
                                        , 'gs_idx'        : GS
                                        , 'nbr_species'   : nbr_species
                                        , 'ss_dist'       : ss_dist
                                        }

    # For heatmap stuff
    else:
        param1_2D = np.unique(param1); param2_2D = np.unique(param2)
        dim_1     = len(param1_2D)   ; dim_2      = len(param2_2D)

        if dim_1 == 1:
            print('Warning : Parameter ' + parameter1 + ' does not vary!')
            print('       > This will make for a very boring heatmap...')

        elif dim_2 == 1:
            print('Warning : Parameter ' + parameter2 + ' does not vary!')
            print('       > This will make for a very boring heatmap...')

        # initialize
        mean_pop2D          = np.zeros((dim_1,dim_2))
        mean_rich2D         = np.zeros((dim_1,dim_2))
        mean_time_present2D = np.zeros((dim_1,dim_2))
        P02D                = np.zeros((dim_1,dim_2))
        nbr_local_max2D     = np.zeros((dim_1,dim_2))
        H2D                 = np.zeros((dim_1,dim_2))
        GS2D                = np.zeros((dim_1,dim_2))
        nbr_species2D       = np.zeros((dim_1,dim_2))
        ss_dist2D           = np.zeros((dim_1,dim_2,length_longest_dstbn))

        # put into a 2d array all the previous results
        for sim in np.arange(nbr_sims):
            i = np.where(param1_2D==param1[sim])[0][0]
            j = np.where(param2_2D==param2[sim])[0][0]
            mean_pop2D[i,j]          = mean_pop[sim]
            mean_rich2D[i,j]         = mean_rich[sim]
            mean_time_present2D[i,j] = mean_time_present[sim]
            P02D[i,j]                = P0[sim]
            nbr_local_max2D[i,j]     = nbr_local_max[sim]
            H2D[i,j]                 = H[sim]
            GS2D[i,j]                = GS[sim]
            nbr_species2D[i,j]       = nbr_species[sim]
            ss_dist2D[i,j]           = ss_dist[sim]

        # arrange into a dictionary to save
        dict_arrays = {  parameter1 : param1_2D, parameter2  : param2_2D
                                           , 'mean_pop'      : mean_pop2D
                                           , 'mean_rich'     : mean_rich2D
                                           , 'mean_time_present' : mean_time_present2D
                                           , 'P0'            : P02D
                                           , 'nbr_local_max' : nbr_local_max2D
                                           , 'entropy'       : H2D
                                           , 'gs_idx'        : GS2D
                                           , 'nbr_species'   : nbr_species2D
                                           , 'ss_dist'       : ss_dist2D
                                           }
    # save results in a npz file
    np.savez(filename, **dict_arrays)

    return 0

def mlv_plot_sim_results(dir, parameter1):
    """
    Plot results from file consolidated results. If it doesn't exist,
    creates it here.
    """
    filename =  dir + os.sep + 'consolidated_results.npz'

    if not os.path.exists(filename):
        mlv_consolidate_sim_results(dir, parameter1)

    with np.load(filename) as f:
        param1    = f[parameter1] ; mean_pop          = f['mean_pop']
        mean_rich = f['mean_rich']; mean_time_present = f['mean_time_present']
        P0        = f['P0']       ; nbr_local_max     = f['nbr_local_max']
        H         = f['entropy']  ; GS                = f['gs_idx']
        nbr_spec  = f['nbr_species']

    labelx = VAR_NAME_DICT[parameter1]

    # richness
    fig = plt.figure()
    plt.scatter(param1, (1.0-P0)*nbr_spec, marker='+', label=r'$(1-P(0))$')
    plt.scatter(param1, mean_rich, marker='x', label='Gill. mean rich.')
    plt.ylabel(r"richness")
    plt.xlabel(labelx)
    #plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.show()

    # mean
    fig = plt.figure()
    plt.scatter(param1, mean_pop, color='r')
    plt.ylabel(r'$\langle n_i \rangle$')
    plt.xlabel(labelx)
    #plt.yscale('log')
    plt.xscale('log')
    plt.show()

    # Entropy
    fig = plt.figure()
    plt.scatter(param1, H, color='b')
    plt.ylabel(r"entropy")
    plt.xlabel(labelx)
    #plt.yscale('log')
    plt.xscale('log')
    plt.show()

    # Gini-Simpson index
    fig = plt.figure()
    plt.scatter(param1, GS, color='b')
    plt.ylabel(r"gini-simpson index")
    plt.xlabel(labelx)
    #plt.yscale('log')
    plt.xscale('log')
    plt.show()

    # mean_time_present
    fig = plt.figure()
    plt.scatter(param1, mean_time_present, color='g')
    plt.ylabel(r"$\langle t_{present} \rangle$")
    plt.xlabel(labelx)
    #plt.yscale('log')
    plt.xscale('log')
    plt.show()

    return 0

def mlv_sim2theory_results(dir, parameter1):
    """
    Plot results from file consolidated results. If it doesn't exist,
    creates it here.
    """

    return 0

def heatmap(xrange, yrange, arr, xlabel, ylabel, title, pbtx=18, pbty=9
            , save=False):

    plt.style.use('custom_heatmap.mplstyle')

    if title not in IMSHOW_KW:
        imshow_kw = {'cmap': 'YlGnBu', 'aspect': None }
    else:
        imshow_kw = IMSHOW_KW[title]

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    im = ax.imshow(arr, interpolation='none', **imshow_kw)

    POINTS_BETWEEN_X_TICKS = pbtx
    POINTS_BETWEEN_Y_TICKS = pbty
    # labels and ticks
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
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    ax.invert_yaxis()
    #plt.xscale('log'); plt.yscale('log')
    plt.colorbar(im,ax=ax)
    plt.title(title)
    if save:
        fname = ((title.replace(" ","")).replace('/','_')).replace("$","")
        plt.savefig(SIM_FIG_DIR + os.sep + fname + '.pdf');
        #plt.savefig(DIR_OUTPUT + os.sep + fname + '.eps');
        plt.savefig(SIM_FIG_DIR + os.sep + fname + '.png')
    else:
        plt.show()

    plt.close()

    return 0

def mlv_plot_sim_results_heatmaps(dir, parameter1, parameter2, save=False):
    """
    Plot results from file consolidated results. If it doesn't exist,
    creates it here.
    """
    filename =  dir + os.sep + 'consolidated_results.npz'

    if not os.path.exists(filename):
        mlv_consolidate_sim_results(dir, parameter1)

    with np.load(filename) as f:
        param1_2D   = f[parameter1]  ; mean_pop2D          = f['mean_pop']
        mean_rich2D = f['mean_rich'] ; mean_time_present2D = f['mean_time_present']
        P02D        = f['P0']        ; nbr_local_max2D     = f['nbr_local_max']
        H2D         = f['entropy']   ; GS2D                = f['gs_idx']
        nbr_spec2D  = f['nbr_species']; param2_2D   = f[parameter2]

    labelx = VAR_NAME_DICT[parameter1]; labely = VAR_NAME_DICT[parameter2]

    ## Entropy 2D
    #heatmap(xrange, yrange, arr, xlabel, ylabel, title)
    heatmap(param1_2D, param2_2D, H2D.T, labelx, labely, 'entropy', save=save)

    ## Gini-Simpson
    heatmap(param1_2D, param2_2D, GS2D.T, labelx, labely, 'Gini-Simpson index'
                    , save=save)

    ## Richness ( divide nbr_species*(1-P0) by mean_pop )
    heatmap(param1_2D, param2_2D, nbr_spec2D.T*(1.0-P02D).T, labelx, labely
            , r'$S(1-P(0))$', save=save)
    heatmap(param1_2D, param2_2D, mean_rich2D.T, labelx, labely
            , r'$\langle S \rangle$', save=save)

    #heatmap(param1_2D, param2_2D, np.divide(nbr_spec2D*(1.0-P02D), mean_rich2D).T
    #        , labelx, labely, r'$S(1-P(0))/\langle S \rangle$', save=save)


    heatmap(param1_2D, param2_2D, np.divide(nbr_spec2D*(1.0-P02D), mean_rich2D).T
            , labelx, labely, r'$S(1-P(0))/\langle S \rangle$', save=save)

    ## Mean time present



    return 0

def mlv_sim2theory_results_heatmaps(dir, parameter1, parameter2, save=False):
    """
    Plot results from file consolidated results. If it doesn't exist,
    creates it here.
    """
    theory_fname = theqs.THRY_FIG_DIR + os.sep + 'metrics2.npz'
    simulation_fname = dir + os.sep + 'consolidated_results.npz'

    if not os.path.exists(simulation_fname):
        mlv_consolidate_sim_results(dir, parameter1, parameter2)

    if not os.path.exists(theory_fname):
        print('Warning : ' + theory_fname + " doesn't exist!")
        raise SystemExit

    with np.load(simulation_fname) as f:
        param1_2D   = f[parameter1]   ; mean_pop2D          = f['mean_pop']
        mean_rich_sim = f['mean_rich']; mean_time_present2D = f['mean_time_present']
        P02D        = f['P0']         ; nbr_local_max2D     = f['nbr_local_max']
        H2D         = f['entropy']    ; GS2D                = f['gs_idx']
        nbr_spec2D  = f['nbr_species']; param2_2D   = f[parameter2]
        dist_sim    = f['ss_dist']

    labelx = VAR_NAME_DICT[parameter1]; labely = VAR_NAME_DICT[parameter2]

    with np.load(simulation_fname) as f:
        dist_thry   = f['approx_dist']; richness_thry = f['richness']

    heatmap(param1_2D, param2_2D, self.model.JS_divergence(dist_sim, dist_thry).T
            , labelx, labely, r'Jensen-Shannon Divergence', save=save)

    return 0



### SIR

def sir_dstbn_fp(all_traj, plot = True):
    """
    Plots distribution of Susceptible when infected get to zero
    Input :
        all_traj (array) : all trajectories ( num_traj x length_traj
                                              x num_species )

    """

    FORCE_NUMBER = 1000

    final_pstn = [];

    for i in range(0,FORCE_NUMBER):
        final_pstn.append(all_traj[i,-1,0])

    kwargs = dict( alpha=0.3, density=True, histtype='stepfilled',
                   color='steelblue');
    plt.hist( final_pstn, bins=100, **kwargs );
    plt.xlim(left=0.0)
    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    plt.yscale('log')
    plt.ylabel(r'probability')
    plt.xlabel(r'Susceptible at fpt')
    plt.show()


    return 0

# TODO Somethimes should be able to just
def sir_mean_trajectory(sim_dir, plot = True):
    """
    Assuming the trajectories all end at the same point (extinction of infected)
    , this will
    recreate the mean trajectory to extinction

    Input :gi
        sim_dir (string) : diretory to go through all the time files
        plot (binary)    : whether or not to plot this distribution

    Return :
        mean_traj (array) : array of first passage times

    """
    FN = 500;

    all_traj, times = consolidate_trajectories(sim_dir, FORCE_NUMBER=FN);

    fpt = np.mean(times[:,-1])
    for i in range(0,np.shape(times)[1]):
        times[:,i] = times[:,i] - times[:,-1]

    mean_traj_times, step_size = np.linspace( -2*fpt, np.max(times),
    #mean_traj_times, step_size = np.linspace( np.min(times), np.max(times),
                                              FN, retstep=True );
    mean_traj = np.zeros( ( len(mean_traj_times), np.shape(all_traj)[2]) );
    normalization = np.zeros( len( mean_traj_times ) );

    # Populate the mean position at each time point
    for traj_idx in range(0, np.shape(all_traj)[0]):
        for time_idx in range(1, np.shape(all_traj)[1]):
            # check that it's not zero time
            if ( times[traj_idx,time_idx]-times[traj_idx,time_idx-1] != 0.0
                 and -int(np.floor(times[traj_idx,time_idx]/step_size))<FN):
                mean_traj[-1+int(np.floor(times[traj_idx,time_idx]
                                         /step_size)),:] \
                                         += all_traj[traj_idx,time_idx,:]
                normalization[-1+int(np.floor(times[traj_idx,time_idx]
                                                            /step_size) )]+= 1

    mean_traj = mean_traj / normalization[:,np.newaxis]

    #sir_dstbn_fp(all_traj, plot) # TODO : somewhere else

    if plot:
        fig = plt.figure(); ax = plt.gca()
        for i in range(0, len(mean_traj_times) - 1 ):

            ax.plot(mean_traj[i:i+2,0], mean_traj[i:i+2,1],
                    color=plt.cm.plasma(int(255*i/len(mean_traj_times))))
        plt.ylim(bottom=0.0); #plt.xlim(left=0.0)
        min_ylim, max_ylim = plt.ylim()
        min_xlim, max_xlim = plt.xlim()
        plt.xlabel(r'number susceptible ($S$)')
        plt.ylabel(r'number infected ($I$)')
        plt.show()

    return mean_traj

if __name__ == "__main__":

    sim_dir = RESULTS_DIR + os.sep + 'multiLV2'

    #mlv_consolidate_sim_results(sim_dir, 'comp_overlap', 'immi_rate')
    mlv_sim2theory_results_heatmaps(sim_dir, 'comp_overlap', 'immi_rate', save=False)
    #mlv_plot_sim_results_heatmaps(sim_dir, 'comp_overlap', 'immi_rate', save=False)

    #mlv_plot_single_sim_results(sim_dir, sim_nbr = 401)
    #mlv_plot_single_sim_results(sim_dir, sim_nbr = 361)
    #mlv_plot_single_sim_results(sim_dir, sim_nbr = 381)
    #mlv_plot_single_sim_results(sim_dir, sim_nbr = 122)
    #mlv_plot_sim_results(sim_dir, 'comp_overlap')

    #sim_dir = RESULTS_DIR + os.sep + 'sir0'
    #fpt_distribution(sim_dir)
    #sir_mean_trajectory(sim_dir)
