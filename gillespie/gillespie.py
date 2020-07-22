#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jeremy Rothschild

Other info...

Usage :

Example usage :

"""

# TODO : num_states is poorly named, should be number of species or something.
#        Even deleted if if must.

import numpy as np
import random, datetime, argparse
from gillespie_models import MODELS
#import gillespie_analysis as ga unnecessary
import os
import gillespie_models as gm

def sample_discrete(probs, r2):
    """
    Randomly sample an reaction from probability given by probs.
    Returns i, the index of the reaction that will occur.

    Input
        probs : array of probabilities of each reaction
        r2    : uniformly drawn random number between (0,1)

    Returns:
        i-1 : index of reaction
    """
    i = 0; p_sum = 0.0
    while p_sum < r2: #find index
        p_sum += probs[i]; i += 1
    return i - 1

def gillespie_draw(Model, current_state):
    """
    Randomly sample a time and the reaction (from propensities) that takes place
    at that time.

    Returns:
        propensity   : array of propensities
        reaction_idx : which reaction happens
        time         : time that passes, dt
    """
    # draw random number_runs
    r1, r2 = np.random.random(2)

    # compute propensities
    propensity = Model.propensity( current_state );
    prob_propensity = propensity/sum( propensity )

    # compute time
    time = - np.log( r1 )/np.sum( propensity )

    # draw reaction from this distribution
    reaction_index = sample_discrete(prob_propensity, r2)

    return reaction_index, time

def gillespie(Model, traj):
    """
    Running 1 trajectory

    Input:
        Model (Class)   : Model object that has functions to get propensities,
                          updates, etc.
        traj            : trajectory index we are following

    Returns:
        simulation : the whole simulated trajectory
        times      : when each reaction happened along the trajectory
    """
    init_state = Model.initialize( )

    # create output arrays output
    simulation = np.zeros( ( Model.max_gen_save, len(init_state) ) )
    times = np.zeros( ( Model.max_gen_save ) )

    # Initialize and perform simulation
    i = 1
    simulation[0,:] = init_state.copy()
    current_state = simulation[0,:].copy()

    while not ( ( Model.stop_condition(current_state) ) or
                  Model.generation_time_exceed( times[(i-1)%Model.max_gen_save],
                                                i-1) ) :
        print(i)
        # draw the event and time step
        reaction_idx, dt = gillespie_draw(Model, current_state)

        Model.update_results(current_state, dt)

        # Update the system
        # TODO what if system size changes? Going to have to rethink this...
        simulation[i%Model.max_gen_save,:] = Model.update( current_state,
                                                           reaction_idx );
        times[i%Model.max_gen_save] = times[(i-1)%Model.max_gen_save] + dt;
        current_state = simulation[i%Model.max_gen_save,:].copy()

        i += 1

    Model.save_trajectory(simulation, times, traj)

    return simulation, times

class StoreDictKeyPair(argparse.Action):
    """
    Allows one to store from command line dictionary keys and values
    using 'KEY=VAL' after the -p key
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest,
              nargs=nargs, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split('=')
            my_dict[k] = float(v)
        setattr(namespace, self.dest, my_dict)

if __name__ == "__main__":

    np.random.seed( ) # set random seed

    # get commandline arguments
    parser = argparse.ArgumentParser(description = "Simulation of N species")

    #TODO in this new structure, I think number of species is useless... should
    #     be a done differently.
    parser.add_argument('-m', type = str, default = 'multiLV', nargs = '?',
                        help = "Model to use.")
    parser.add_argument('-g', type = int, default = 10**10, nargs = '?',
                        help = "Number of generations (rxns) in total.")
    parser.add_argument('-T', type = int, default = 10**9, nargs = '?',
                        help = "Total time to not exceed.")
    parser.add_argument('-t', type = int, default = 1, nargs = '?',
                        help = "Number of runs/trajectories.")
    parser.add_argument('-n', type = int, default = 0, nargs = '?',
                        help = "Simulation number.")
    parser.add_argument('-p', metavar='KEY=VAL', default= {'nada' : 0.0},
                        dest='my_dict', nargs='*', action=StoreDictKeyPair,
                        required=False,
                        help='Additional parameters to be passed on for the \
                        simulation')
    # TODO : add multiprocessing. Will make it a lot better. Major changes need to
    #        happen to parallelize all this.

    args = parser.parse_args()
    model = args.m; num_runs = args.t;
    nbr_generations = args.g;
    param_dict = vars(args)['my_dict']
    param_dict['sim_number'] = args.n
    param_dict['nbr_generations'] = args.g
    param_dict['max_time'] = args.T

    # select which class/model we are using
    Model = gm.MODELS[model](**param_dict)

    # make directory to save simulation number, change if already exists
    Model.create_sim_folder(); param_dict['sim_number'] = Model.sim_number

    # run gillespie TODO parallelize. Have a couple savepoints?
    for traj in range( num_runs ):
        Model.__init__(**param_dict)
        gillespie(Model, traj)
