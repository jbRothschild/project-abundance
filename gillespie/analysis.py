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
        plot (binary) : whether or not to plot this distribution

    Return :
        all_traj (array) : all trajectories ( num_traj x length_traj x num_species )
        times (array) : array of times (note that some early times might be zero)
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
    all_traj = np.zeros( (num_traj, int(param_dict['max_gen_save'][0]), int(param_dict['num_species'][0])) ) ; i = 0;

    for subdir, dirs, files in os.walk(sim_dir):
        for filename in files:
            filepath = subdir + os.sep + filename
            if ('trajectory' in filepath) and ( not filepath.endswith("time.txt") ) and i<FORCE_NUMBER:
                all_traj[i, :, :] = np.loadtxt(filepath)
                times[i,:] = np.loadtxt(filepath[:-4]+'_time.txt')
                i += 1;

    return all_traj, times

def fpt_distribution(sim_dir, plot = True):
    """
    Assuming the trajectories all end with a fpt event, this will output all the fpt

    Input :
        sim_dir (string) : diretory to go through all the time files
        plot (binary) : whether or not to plot this distribution

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
        kwargs = dict(alpha=0.3, density=True,
             histtype='stepfilled', color='steelblue');
        plt.hist(fpt, bins=100, **kwargs );
        plt.axvline(fpt.mean(), color='k', linestyle='dashed', linewidth=1);
        min_ylim, max_ylim = plt.ylim()
        plt.text(fpt.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(fpt.mean()))
        plt.yscale('log')
        plt.ylabel(r'probability')
        plt.xlabel(r'time (gen.)')
        plt.show()

    return fpt

def sir_mean_trajectory(sim_dir, plot = True):
    """
    Assuming the trajectories all end at the same point (extinction of infected), this Will
    recreate the mean trajectory to extinction

    Input :
        sim_dir (string) : diretory to go through all the time files
        plot (binary) : whether or not to plot this distribution

    Return :
        mean_traj (array) : array of first passage times

    """

    all_traj, times = consolidate_trajectories(sim_dir, FORCE_NUMBER=100);

    print(np.shape(times)[0])

    for i in range(0,np.shape(times)[1]):
        times[:,i] = times[:,i] - times[:,-1];

    mean_traj_times = np.linspace( min(times), np.max(times), 1000 )
    mean_traj = np.zeros( ( len(mean_traj_times), np.shape(all_traj)[2]) )
    normalization = zeros( len( mean_traj_times ) )

    return mean_traj

if __name__ == "__main__":

    sim_dir = RESULTS_DIR + os.sep + 'sir'

    fpt_distribution(sim_dir)
    sir_mean_trajectory(sim_dir)
