import os, glob, csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from gillespie_models import RESULTS_DIR

plt.style.use('custom.mplstyle')

def consolidate_trajectories(sim_dir, save_file=False, FORCE_NUMBER=3000):
    """
    Put all trajectories into 1 huge array. Hopefully easier to manipulate then.

    Input :
        sim_dir (string) : diretory to go through all the time files
        plot (binary)    : whether or not to plot this distribution

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
        plt.hist( fpt, bins=100, **kwargs );
        plt.axvline( fpt.mean(), color='k', linestyle='dashed', linewidth=1 );
        plt.xlim(left=0.0)
        min_ylim, max_ylim = plt.ylim()
        min_xlim, max_xlim = plt.xlim()
        plt.text( fpt.mean()*1.1, max_ylim*0.9,
                  'Mean: {:.2f}'.format(fpt.mean()) )
        plt.yscale('log')
        plt.ylabel(r'probability')
        plt.xlabel(r'time (gen.)')
        plt.show()

    return fpt

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
    FN = 500

    all_traj, times = consolidate_trajectories(sim_dir, FORCE_NUMBER=FN);

    fpt = np.mean(times[:,-1])
    for i in range(0,np.shape(times)[1]):
        times[:,i] = times[:,i] - times[:,-1]


    mean_traj_times, step_size = np.linspace( -2*fpt, np.max(times),
    #mean_traj_times, step_size = np.linspace( np.min(times), np.max(times),
                                              FN, retstep=True )
    mean_traj = np.zeros( ( len(mean_traj_times), np.shape(all_traj)[2]) )
    normalization = np.zeros( len( mean_traj_times ) )

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

    if plot:
        fig = plt.figure(); ax = plt.gca()
        for i in range(0, len(mean_traj_times) - 1 ):
            print(mean_traj[i:i+2,0],mean_traj[i:i+2,1])
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

    sim_dir = RESULTS_DIR + os.sep + 'sir0'

    #fpt_distribution(sim_dir)
    sir_mean_trajectory(sim_dir)
