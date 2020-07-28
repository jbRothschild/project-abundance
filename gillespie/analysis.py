import os, glob, csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from gillespie_models import RESULTS_DIR, MultiLV, SIR
import theory_equations as theqs

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

def mlv_single_sim_results(dir, parameter, sim_nbr = 1, plot = True):
    """
    Analyze information collected in results_(sim_nbr).pickle

    Input :
        dir       : directory that we're plotting from
        parameter : the parameter that changes between the different simulations
                    (string)
    Output :

    """

    with open(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
               'results_0.pickle', 'rb') as handle:
        param_dict = pickle.load(handle)

    model = MultiLV(**param_dict)

    # TODO : Why isn't this normalized? I think because of total time...
    ss_dist = model.results['ss_distribution']/np.sum(model.results['ss_distribution'])
    mean_richness = np.dot(model.results['richness']
                           , np.arange(len(model.results['richness'])))
    mean_time_btwn_exit = np.mean(model.results['time_btwn_ext'])
    mean_nbr_indvdl = np.dot(ss_dist, np.arange(len(ss_dist)))

    # theory equations
    theory_models = theqs.Model_MultiLVim(**param_dict)
    theory_dist, theory_abund = theory_models.abund_1spec_MSLV()

    if plot:
        fig = plt.figure()
        plt.scatter(np.arange(len(model.results['richness'])),
                 model.results['richness'], color='b')
        plt.ylabel(r"probability of richness")
        plt.xlabel(r'richness')
        plt.axvline( mean_richness, color='k' , linestyle='dashed', linewidth=1)
        #plt.yscale('log')
        #plt.xscale('log')
        plt.show()

        ## dstbn present
        if model.results['time_btwn_ext'] != []:
            nbins = 100
            logbins = np.logspace(np.log10(np.min(model.results['time_btwn_ext']))
                                  , np.log10(np.max(model.results['time_btwn_ext']))
                                  , nbins)
            counts, bin_edges = np.histogram(model.results['time_btwn_ext']
                                             , density=True
            #                                 , bins=logbins)
                                             , bins = nbins)
            fig = plt.figure()
            axes = plt.gca()
            plt.scatter((bin_edges[1:]+bin_edges[:-1])/2,
                     counts, color='g')
            plt.axvline( mean_time_btwn_exit, color='k', linestyle='dashed'
                        , linewidth=1 ) # mean
            plt.ylabel(r"probability of time present")
            plt.xlabel(r'time present between extinction')
            plt.yscale('log')
            axes.set_ylim([np.min(counts[counts!=0.0]),2*np.max(counts)])
            #plt.xscale('log')
            plt.show()

        ## ss_dstbn (compare with deterministic mean)
        fig = plt.figure()
        axes = plt.gca()

        # deterministic mean
        det_mean = theory_models.deterministic_mean()

        plt.scatter(np.arange(len(ss_dist)),ss_dist,label='simulation')
        plt.plot(np.arange(len(theory_dist)),theory_dist,label='theory')
        plt.ylabel(r"probability distribution function")
        plt.xlabel(r'n')
        plt.axvline( mean_nbr_indvdl, color='k' , linestyle='dashed'
                    , linewidth=1 ) #mean
        plt.axvline( theory_models.deterministic_mean(), color='k' ,
                    linestyle='dashdot', linewidth=1 ) #mean
        plt.text( det_mean*1.1, 2*np.max(ss_dist)*0.9,
                  'deterministic mean: {:.2f}'.format(det_mean) )
        plt.yscale('log')
        axes.set_ylim([np.min(ss_dist[ss_dist!=0.0]),2*np.max(ss_dist)])
        axes.set_xlim([0.0,np.max(np.nonzero(ss_dist))])
        plt.legend(loc='best')
        #plt.xscale('log')
        plt.show()

    # or should I return model? more things there
    return param_dict, mean_richness, mean_time_btwn_exit, mean_nbr_indvdl \
           , ss_dist[0]

def mlv_results(dir, parameter):
    """
    Analyze how the results form different simulations differ for varying
    (parameter)

    Input :
        dir       : directory that we're plotting from
        parameter : the parameter that changes between the different simulations
                    (string)
    """
    for i in np.arange(20):
        param_dict, mean_richness, mean_time_btwn_exit, mean_nbr_indvdl \
           , ss_dist[0] = mlv_single_sim_results(dir, parameter, sim_nbr = i
                                                 , plot = False)

    # mean richness vs (1-P(0)*nbr_species) vs equation

    # mean_time_btwn_exit vs mean extinciton time

    # mean vs deterministic mean

def consolidate_simulation_results():

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

    sim_dir = RESULTS_DIR + os.sep + 'multiLV0'

    mlv_single_sim_results(sim_dir, 'comp_overlap', sim_nbr = 25, plot = True)

    #sim_dir = RESULTS_DIR + os.sep + 'sir0'
    #fpt_distribution(sim_dir)
    #sir_mean_trajectory(sim_dir)
