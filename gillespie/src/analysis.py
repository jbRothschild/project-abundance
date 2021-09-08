import os, glob, csv, copy, io

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

from scipy.signal import argrelextrema

from src.gillespie_models import RESULTS_DIR, MultiLV, SIR
import src.theory_equations as theqs
from src.settings import VAR_NAME_DICT, COLOURS, IMSHOW_KW

SIM_FIG_DIR = 'figures' + os.sep + 'simulations'
while not os.path.exists( os.getcwd() + os.sep + SIM_FIG_DIR ):
    os.makedirs(os.getcwd() + os.sep + SIM_FIG_DIR);

import pickle

plt.style.use('src/custom.mplstyle')

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
    for subdir, dirs, fmean_popiles in os.walk(sim_dir):
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
    theory_model = theqs.Model_MultiLVim(**param_dict)

    # distribution
    if 'ss_distribution' in model.results:
        ss_dist     = model.results['ss_distribution'] \
                              / np.sum(model.results['ss_distribution'])
        mean_pop    = np.dot(ss_dist, np.arange(len(ss_dist)))
        P0          = ss_dist[0]
        nbr_local_max = np.min([len( argrelextrema(ss_dist, np.greater) ),2])
        H           = -np.dot(ss_dist[ss_dist>0.0],np.log(ss_dist[ss_dist>0.0]))
        GS          = 1.0 - np.dot(ss_dist,ss_dist)
        setattr(theory_model,'nbr_species',int( (1.0-P0)*param_dict['nbr_species']))
        det_mean_present = theory_model.deterministic_mean()
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

    if 'corr_ni_nj' in model.results:
        correlation = model.results['corr_ni_nj']
    else:
        correlation = None

    if 'conditional' in model.results:
        conditional = model.results['conditional']
    else:
        conditional = None

    if 'av_J' in model.results:
        av_J = model.results['av_J']
    else:
        av_J = None

    # TODO : change to dictionary
    return param_dict, ss_dist, richness_dist, time_btwn_ext, mean_pop\
                     , mean_rich, mean_time_present, P0, nbr_local_max, H, GS\
                     , param_dict['nbr_species'], det_mean_present, correlation\
                     , conditional, av_J


def mlv_consolidate_sim_results(dir, parameter1=None, parameter2=None):
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
    #print(nbr_sims)

    # initialize the
    mean_pop    = np.zeros(nbr_sims); param1 = np.zeros(nbr_sims);
    mean_rich   = np.zeros(nbr_sims); mean_time_present = np.zeros(nbr_sims)
    P0          = np.zeros(nbr_sims); nbr_local_max     = np.zeros(nbr_sims)
    H           = np.zeros(nbr_sims); GS                = np.zeros(nbr_sims)
    nbr_species = np.zeros(nbr_sims); ss_dist_vary      = []
    rich_dist_vary = []             ; det_mean_present  = np.zeros(nbr_sims)
    correlation = np.zeros(nbr_sims); cond_vary      = []
    av_J        = np.zeros(nbr_sims);

    # if 2 parameters vary in the simulation
    if parameter2 != None: param2 = np.zeros(nbr_sims)

    # TODO change to dictionary
    for i in np.arange(nbr_sims):
        param_dict, ss_dist_sim, richness_dist,  _, mean_pop[i], mean_rich[i]\
                    , mean_time_present[i], P0[i], nbr_local_max[i], H[i]\
                    , GS[i], nbr_species[i], det_mean_present[i],correlation[i]\
                    , conditional, av_J[i]\
                  = mlv_extract_results_sim(dir, sim_nbr = i+1)
        # sims might not have same distribution length
        rich_dist_vary.append( np.array( richness_dist ) )
        ss_dist_vary.append( np.array( ss_dist_sim ) )
        cond_vary.append( np.array(conditional) )

        # Value of parameters
        param1[i] = param_dict[parameter1]
        if parameter2 != None: param2[i] = param_dict[parameter2]

    # making all sims have same distribution length
    length_longest_dstbn = len(max(ss_dist_vary,key=len))
    length_longest_rich  = len(max(rich_dist_vary,key=len))
    length_longest_cond  = len(max(cond_vary,key=np.shape))
    ss_dist = np.zeros((nbr_sims,length_longest_dstbn))
    rich_dist = np.zeros((nbr_sims,length_longest_rich))
    cond_dist = np.zeros((nbr_sims,length_longest_cond,length_longest_cond))
    for i in np.arange(nbr_sims):
        ss_dist[i,:len(ss_dist_vary[i])] = ss_dist_vary[i]
        rich_dist[i,:len(rich_dist_vary[i])] = rich_dist_vary[i]
        cond_dist[i,:np.shape(cond_vary[i])[0],:np.shape(cond_vary[i])[1]] =\
                                    cond_vary[i]

    # Single parameter changing
    if parameter2 == None:
        if len(np.unique(param1)) == 1:
            print('Warning : ' + parameter1 + ' parameter does not vary!')
            #raise SystemExit
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
                                        , 'det_mean_present' : det_mean_present
                                        , 'rich_dist'       : rich_dist
                                        , 'correlation'     : correlation
                                        , 'conditional'     : cond_dist
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
        rich_dist2D         = np.zeros((dim_1,dim_2,length_longest_rich))
        det_mean_present2D  = np.zeros((dim_1,dim_2))
        correlation2D       = np.zeros((dim_1,dim_2))
        cond_2D             = np.zeros((dim_1,dim_2,length_longest_cond
                                                    ,length_longest_cond))
        av_J2D              = np.zeros((dim_1,dim_2))

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
            rich_dist2D[i,j]         = rich_dist[sim]
            det_mean_present2D[i,j]  = det_mean_present[sim]
            correlation2D[i,j]       = correlation[sim]
            cond_2D[i,j]             = cond_dist[sim]
            av_J2D[i,j]              = av_J[sim]

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
                                           , 'det_mean_present' : det_mean_present2D
                                           , 'rich_dist'        : rich_dist2D
                                           , 'correlation'      : correlation2D
                                           #, 'conditional'      : cond_2D
                                           , 'av_J2D'           : av_J2D
                                           }
    # save results in a npz file
    np.savez(filename, **dict_arrays)
    sio.savemat(filename[:-4]+'.mat',mdict=dict_arrays)

    return filename, dict_arrays

def mlv_multiple_folder_consolidate(list_dir, consol_name_dir, parameter1=None
                                            , parameter2=None):
    """
    Takes simulations from multiple folders and combines them by averaging their
    results all together. The only reason I do this is to average the
    distributions
    """
    dict_arr = []
    for dir in list_dir:
         _, dict_temp = mlv_consolidate_sim_results(dir, parameter1, parameter2)
         dict_arr.append(dict_temp)
         del dict_temp

    df = pd.DataFrame(dict_arr); mean_dict = {}
    for column in df:
        mean_dict[column] = df[column].mean()

    while not os.path.exists( consol_name_dir ):
        os.makedirs( consol_name_dir );

    filename =  consol_name_dir + os.sep + 'consolidated_results.npz'

    # save results in a npz file
    np.savez(filename, **mean_dict)
    sio.savemat(filename[:-4]+'.mat',mdict=mean_dict)

    return 0

def mlv_plot_average_sim_results(dir,parameter='comp_overlap'):
    """
    Plot the average of a many results_(sim_nbr).pickle

    Input :
        dir     : directory that we're plotting from
        sim_nbr : simulation number (subdir sim%i %(sim_nbr))

    Output :
        Plots of a single simul
    """
    filename =  dir + os.sep + 'consolidated_results.npz'

    if not os.path.exists(filename):
        mlv_consolidate_sim_results(dir, parameter)

    with np.load(filename) as f:
        dist_sim    = f['ss_dist'];
        cond_dist   = f['conditional'];

    param_dict, ss_dist, richness_dist, time_btwn_ext, mean_pop, mean_rich\
        , mean_time_present, P0, nbr_local_max, H, GS, nbr_species\
        , det_mean_present, correlation, conditional\
        = mlv_extract_results_sim(dir, 1)

    ss_dist_sim = np.mean(dist_sim, axis=0)
    mean_cond   = np.mean(cond_dist,axis=0)

    fig  = plt.figure()
    axes = plt.gca()
    r = np.random.randint(np.shape(dist_sim)[0], size=3)
    plt.plot(np.arange(len(ss_dist_sim)),ss_dist_sim,label='mean simulation')
    plt.scatter(np.arange(len(dist_sim[r[0]])),dist_sim[r[0]],label='simulation i')
    plt.scatter(np.arange(len(dist_sim[r[1]])),dist_sim[r[1]],label='simulation j')
    plt.scatter(np.arange(len(dist_sim[r[2]])),dist_sim[r[2]],label='simulation k')
    plt.ylabel(r"probability distribution function")
    plt.xlabel(r'n')
    plt.yscale('log')
    axes.set_ylim([np.min(ss_dist_sim[ss_dist_sim!=0.0]),2*np.max(ss_dist_sim)])
    title = r'$\rho=$' + str(param_dict['comp_overlap']) + r', $\mu=$' \
            + str(param_dict['immi_rate']) + r', $S=$' + str(param_dict['nbr_species'])
    plt.title(title)
    axes.set_xlim([0.0,np.max(np.nonzero(ss_dist_sim))])
    plt.legend()
    fname = 'av_distribution'
    plt.savefig(dir + os.sep + fname + '.pdf');
    #plt.xscale('log')
    #plt.show()

    fig  = plt.figure()
    axes = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('PuBu'))
    my_cmap.set_bad((0,0,0))
    #print(np.sum( mean_cond[:200,:200].T + mean_cond[:200,:200] ,axis=1))
    axis_show = int(param_dict['carry_capacity'] * 1.5)
    plt.imshow( mean_cond[:axis_show,:axis_show].T
                , norm=mpl.colors.LogNorm(), cmap=my_cmap
                , interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.ylabel(r"i")
    plt.xlabel(r'j')
    title = r'$\rho=$' + str(param_dict['comp_overlap']) + r', $\mu=$' \
            + str(param_dict['immi_rate']) + r', $S=$' + str(param_dict['nbr_species'])
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(r'$P(i|j)$')
    #plt.xscale('log')
    fname = 'conditional'
    plt.savefig(dir + os.sep + fname + '.pdf');

    param_dict['conditional'] = mean_cond
    model = theqs.Model_MultiLVim(**param_dict)
    probability = model.abund_jer( approx='simulation' )

    fig  = plt.figure()
    axes = plt.gca()
    r = np.random.randint(np.shape(dist_sim)[0], size=3)
    plt.plot(np.arange(len(ss_dist_sim)),ss_dist_sim,label='mean trajectories')
    plt.plot(np.arange(len(probability)),probability,label=r'from $P(n_i|n_j)$')
    plt.ylabel(r"probability distribution function")
    plt.xlabel(r'n')
    plt.yscale('log')
    axes.set_ylim([np.min(ss_dist_sim[ss_dist_sim!=0.0]),2*np.max(ss_dist_sim)])
    title = r'$\rho=$' + str(param_dict['comp_overlap']) + r', $\mu=$' \
            + str(param_dict['immi_rate']) + r', $S=$' + str(param_dict['nbr_species'])
    plt.title(title)
    axes.set_xlim([0.0,np.max(np.nonzero(ss_dist_sim))])
    plt.legend()
    #plt.xscale('log')
    fname = 'check_steady_state'
    plt.savefig(dir + os.sep + fname + '.pdf');
    #plt.show()

    return 0

def mlv_plot_single_sim_results(dir, sim_nbr = 1):
    """
    Plot information collected in a single results_(sim_nbr).pickle

    Input :
        dir     : directory that we're plotting from
        sim_nbr : simulation number (subdir sim%i %(sim_nbr))

    Output :
        Plots of a single simulation
    """
    # TODO : replace with a dict
    param_dict, ss_dist_sim, richness_sim, time_present_sim, mean_pop_sim\
                  , mean_rich_sim, mean_time_present_sim, _, _, _, _, _ , _, _\
                  , conditional\
                  = mlv_extract_results_sim(dir, sim_nbr=sim_nbr)

    # theory equations
    theory_models   = theqs.Model_MultiLVim(**param_dict)
    conv_dist, _ = theory_models.abund_1spec_MSLV()
    mf_dist, mf_abund = theory_models.abund_sid()
    title = r'$\rho=$' + str(param_dict['comp_overlap']) + r', $\mu=$' \
            + str(param_dict['immi_rate']) + r', $S=$' + str(param_dict['nbr_species'])

    fig = plt.figure()
    plt.scatter(np.arange(len(richness_sim)), richness_sim, color='b')
    plt.ylabel(r"probability of richness")
    plt.xlabel(r'richness')
    plt.axvline( mean_rich_sim, color='k' , linestyle='dashed', linewidth=1)
    plt.title(title)
    fname = 'richness' + 'sim' + str(sim_nbr)
    plt.savefig(dir + os.sep + fname + '.pdf');
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.show()

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
        plt.title(title)
        fname = 'time_present' + 'sim' + str(sim_nbr)
        plt.savefig(dir + os.sep + fname + '.pdf');
        #plt.xscale('log')
        #plt.show()

    ## ss_dstbn (compare with deterministic mean)
    fig  = plt.figure()
    axes = plt.gca()

    plt.scatter(np.arange(len(ss_dist_sim)),ss_dist_sim,label='simulation')
    plt.plot(np.arange(len(conv_dist)),conv_dist,label='convolution approx.')
    plt.plot(np.arange(len(mf_dist)),mf_dist,label='mean field approx.')
    plt.ylabel(r"probability distribution function")
    plt.xlabel(r'n')
    plt.axvline( mean_pop_sim, color='r' , linestyle='dashed'
                , linewidth=1 ) #mean
    plt.axvline( theory_models.deterministic_mean(), color='k' ,
                linestyle='dashdot', linewidth=1 ) #mean
    setattr(theory_models,'nbr_species',int(mean_rich_sim))
    plt.axvline( theory_models.deterministic_mean(), color='b' ,
                linestyle='-', linewidth=1 ) #mean
    plt.yscale('log')
    axes.set_ylim([np.min(ss_dist_sim[ss_dist_sim!=0.0]),2*np.max(ss_dist_sim)])
    title = r'$\rho=$' + str(param_dict['comp_overlap']) + r', $\mu=$' \
            + str(param_dict['immi_rate']) + r', $S=$' + str(param_dict['nbr_species'])
    plt.title(title)
    axes.set_xlim([0.0,np.max(np.nonzero(ss_dist_sim))])
    plt.legend(loc='best')
    fname = 'distribution' + 'sim' + str(sim_nbr)
    plt.savefig(dir + os.sep + fname + '.pdf');
    #plt.xscale('log')
    #plt.show()


    fig  = plt.figure()
    axes = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('PuBu'))
    my_cmap.set_bad((0,0,0))
    print(np.sum(conditional,axis=0),np.sum(conditional,axis=1))
    plt.imshow( conditional[:2*param_dict['carry_capacity']
                ,:2*param_dict['carry_capacity']].T
                , norm=mpl.colors.LogNorm(), cmap=my_cmap
                , interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.ylabel(r"i")
    plt.xlabel(r'j')
    title = r'$\rho=$' + str(param_dict['comp_overlap']) + r', $\mu=$' \
            + str(param_dict['immi_rate']) + r', $S=$' + str(param_dict['nbr_species'])
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(r'$P(i|j)$')
    fname = 'conditional' + 'sim' + str(sim_nbr)
    plt.savefig(dir + os.sep + fname + '.pdf');
    #plt.xscale('log')
    #plt.show()

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

def heatmap(xrange, yrange, arr, xlabel, ylabel, title, pbtx=10, pbty=20
            , save=False):
    # TODO : CHANGE AXES

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
    plt.colorbar(im,ax=ax,cmap=imshow_kw['cmap'])
    if title == 'Local maxima':
        cbar = plt.colorbar(im,ax=ax,cmap=imshow_kw['cmap']
                        , boundaries=[-0.5,0.5,1.5,2.5]
                        , ticks=[-0.5,0.5,1.5,2.5])
        cbar.ax.set_yticklabels(['Max 0', 'Max n', '2 Maxs', ''])
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
        mlv_consolidate_sim_results(dir, parameter1, parameter2)

    with np.load(filename) as f:
        param1_2D   = f[parameter1]  ; mean_pop2D          = f['mean_pop']
        mean_rich2D = f['mean_rich'] ; mean_time_present2D = f['mean_time_present']
        P02D        = f['P0']        ; nbr_local_max2D     = f['nbr_local_max']
        H2D         = f['entropy']   ; GS2D                = f['gs_idx']
        nbr_spec2D  = f['nbr_species']; param2_2D          = f[parameter2]
        det_mean_present2D = f['det_mean_present'];
        rich_dist2D = f['rich_dist']
        correlation2D = f['correlation']

    labelx = VAR_NAME_DICT[parameter1]; labely = VAR_NAME_DICT[parameter2]

    ## Entropy 2D
    #heatmap(xrange, yrange, arr, xlabel, ylabel, title)
    heatmap(param1_2D, param2_2D, H2D.T, labelx, labely, 'entropy', save=save)

    ## Gini-Simpson
    heatmap(param1_2D, param2_2D, GS2D.T, labelx, labely, 'Gini-Simpson index'
                    , save=save)

    ## Richness ( divide nbr_species*(1-P0) by mean_pop )
    heatmap(param1_2D, param2_2D, (nbr_spec2D*(1.0-P02D)), labelx, labely
            , r'$S(1-P(0))$', save=save)
    heatmap(param1_2D, param2_2D, mean_rich2D.T, labelx, labely
            , r'$\langle S \rangle$', save=save)

    #heatmap(param1_2D, param2_2D, np.divide(nbr_spec2D*(1.0-P02D),mean_rich2D).T
    #        , labelx, labely, r'$S(1-P(0))/\langle S \rangle$', save=save)

    ## mean_n
    heatmap(param1_2D, param2_2D, mean_pop2D.T, labelx, labely
            , r'$\langle n \rangle$', save=save)

    ## det_mean_n_present
    heatmap(param1_2D, param2_2D, det_mean_present2D.T, labelx, labely
            , r'Lotka Voltera steady state with $S(1-P(0))$', save=save)

    # COrrelation need not be flipped....????
    heatmap(param2_2D, param1_2D, correlation2D, labelx, labely
            , r'$\rho_{Pears}(n_i,n_j)$', save=save)

    ## diversity distribution
    binom_approx    = np.zeros( rich_dist2D.shape )
    JS_rich         = np.zeros( P02D.shape )
    mean_rich_sim   = np.zeros( P02D.shape )
    mean_rich_binom = np.zeros( P02D.shape )
    var_rich_sim   = np.zeros( P02D.shape )
    var_rich_binom = np.zeros( P02D.shape )

    for i in np.arange(len(param1_2D)):
        for j in np.arange(len(param2_2D)):
            binom_approx[i,j,:] =\
                            theqs.Model_MultiLVim().binomial_diversity_dstbn(
                                    P02D[i,j] , nbr_spec2D[i,j])
            JS_rich[i,j] = theqs.Model_MultiLVim().JS_divergence(
                                                            binom_approx[i,j,:]
                                                            , rich_dist2D[i,j,:]
                                                            )

    mean_rich_sim = np.tensordot(np.arange(0,31), rich_dist2D, axes=([0],[2]))
    mean_rich_binom = np.tensordot(np.arange(0,31), binom_approx,axes=([0],[2]))

    var_rich_sim = (np.tensordot(np.arange(0,31)**2, rich_dist2D,axes=([0],[2]))
                                    - mean_rich_sim**2)
    var_rich_binom = np.tensordot(np.arange(0,31)**2, binom_approx
                                    , axes=([0],[2])) - mean_rich_binom**2


    # Somehow none of these need to be flipped... weird.
    heatmap(param1_2D, param2_2D, mean_rich_sim, labelx, labely
                    , r'mean richness (sim.)', save=save) # mean diveristy
    heatmap(param1_2D, param2_2D, mean_rich_binom, labelx, labely
                    , r'mean richness (binom.)', save=save) # av div. binonmia;
    heatmap(param1_2D, param2_2D, (mean_rich_sim/mean_rich_binom)
                    , labelx, labely, r'mean richness (sim./binom.)', save=save)
                # mean/mean diveristy

    heatmap(param1_2D, param2_2D, JS_rich, labelx, labely
                , r'Jensen-Shannon divergence (sim./binom.)', save=save)
                # JS divergenced


    heatmap(param1_2D, param2_2D, var_rich_sim, labelx, labely
                , r'variance richness (sim.)', save=save) # variance
    heatmap(param1_2D, param1_2D, var_rich_binom, labelx, labely
                , r'variance richness (binom.)', save=save) # variance


    heatmap(param1_2D, param2_2D, (var_rich_sim/var_rich_binom), labelx
                , labely, r'var richness (sim./binom.)', save=save)
                # variance/variance


    #plot() # many distributions
    f = plt.figure();


    return 0

def mlv_sim2theory_results_heatmaps(dir, parameter1, parameter2, save=False):
    """
    Plot results from file consolidated results. If it doesn't exist,
    creates it here.
    THIS IS AN AWEFUL FUNCTION THAT NEED METRICS AND CONSOLIDATED TO BE THE SAME
    LENGTH I HATE IT
    """
    theory_fname = theqs.THRY_FIG_DIR + os.sep + 'metric45.npz'
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
        dist_sim    = f['ss_dist']    ; det_mean_present2D = f['det_mean_present']

    labelx = VAR_NAME_DICT[parameter1]; labely = VAR_NAME_DICT[parameter2]

    with np.load(theory_fname) as f:
        dist_thry2   = f['approx_dist_nava']; richness_thry = f['richness']
        det_mean    = f['det_mean']        ; dist_thry = f['approx_dist_sid']

    ## J-S divergence
    JS = np.zeros( ( len(param1_2D) , len(param2_2D) ) )
    for i in range(np.shape(dist_sim)[0]):
        for j in range(np.shape(dist_sim)[1]):
            JS[i,j]=theqs.Model_MultiLVim().JS_divergence(dist_sim[i,j], dist_thry[i,j])

    heatmap(param1_2D, param2_2D, JS.T, labelx, labely
                , r'Jensen-Shannon Divergence', save=save)

    ## mean richness
    heatmap(param1_2D, param2_2D, (30*richness_thry).T
            , labelx, labely, r'Method 2 richness', save=save)
    heatmap(param1_2D, param2_2D, np.divide(30*richness_thry, mean_rich_sim)
            , labelx, labely, r'Method 2 richness / richness simulation', save=save)

    ## mean deterministic vs mean simulation
    heatmap(param1_2D, param2_2D, (np.divide(det_mean, mean_pop2D))
            , labelx, labely, r'LV mean / $\langle n \rangle_{sim}$', save=save)

    ## mean deterministic with S(1-P(0)) vs mean simulation
    heatmap(param1_2D, param2_2D, (np.divide(det_mean_present2D, mean_pop2D))
            , labelx, labely, r'LV mean $S(1-P(0))$ / $\langle n \rangle_{sim}$', save=save)

    heatmap(param1_2D, param2_2D, (np.divide(det_mean_present2D, det_mean))
            , labelx, labely, r'LV mean $(S(1-P(0)))$ / LV mean $S$', save=save)

    ## Number of peaks
    peaks = np.zeros( ( len(param1_2D) , len(param2_2D) ) )
    for i in range(np.shape(dist_sim)[0]):
        for j in range(np.shape(dist_sim)[1]):
            peakhigh = 0; peak0 = 0
            if np.argmax(dist_sim[i,j,1:]) > 1:
                peakhigh = 1;
                peak0 = int( dist_sim[i,j,0] > dist_sim[i,j,1] )

            peaks[i,j] = peak0 + peakhigh

    heatmap(param1_2D, param2_2D, (np.divide(det_mean_present2D, det_mean)).T
            , labelx, labely, r'Local maxima', save=save)


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

    #mlv_plot_average_sim_results(sim_dir,'comp_overlap')


    sim_dir = RESULTS_DIR + os.sep + 'multiLV45'
    sim_dir = RESULTS_DIR + os.sep + 'multiLV71'
    sim_dir = RESULTS_DIR + os.sep + 'multiLVNavaJ'

    mlv_consolidate_sim_results(sim_dir, 'immi_rate', 'comp_overlap')
    """
    mlv_consolidate_sim_results(dir, 'immi_rate', 'comp_overlap')

    mlv_plot_single_sim_results(sim_dir, sim_nbr = 1500)

    mlv_plot_sim_results_heatmaps(sim_dir, 'immi_rate', 'comp_overlap'
                                    , save=True)
    mlv_sim2theory_results_heatmaps(sim_dir, 'immi_rate', 'comp_overlap'
                                        , save=True)
    """
    mult_fold = [RESULTS_DIR + os.sep + 'multiLV71'\
                , RESULTS_DIR + os.sep + 'multiLV72'\
                , RESULTS_DIR + os.sep + 'multiLV73']
    many_dir = RESULTS_DIR + os.sep + 'many_folder1'
    mlv_multiple_folder_consolidate(mult_fold, many_dir,  'immi_rate'
                                            , 'comp_overlap')


    #mlv_plot_sim_results(sim_dir, 'comp_overlap')

    #sim_dir = RESULTS_DIR + os.sep + 'sir0'
    #fpt_distribution(sim_dir)
    #sir_mean_trajectory(sim_dir)
