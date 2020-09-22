#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jeremy Rothschild

Runs a gillespie algorithm for the models defined in gillespie_models.

Usage :
    Any new Model needs at least the following functions in gillespie_model
        - init()
        - propensity()
        - update()
        - initialize()

    Additionally, can have
        - update_results()
        - stop_condition()
        - save_trajectory()

    If using tau-leaping, then will need to add a function to gillespie_models:
        - find_time_noncritical_rxns( current_state, critical_rxns )

Example usage :
    python3 gillespie.py -m [MODEL] -g [NBR_GENERATIONS] -t [NBR_TRAJECTORIES]
                            -T [TOTAL_TIME] -n [SIMULATION_NBR]
                            -tau [TAU_LEAPING(bool)] -p [PARAMETER=VALUE]
    python3 gillespie.py -m multiLV -t 1 -g 70000000 -n 0 -p max_gen_save=10000
                            sim_dir=multiLV3

"""

# TODO : num_states is poorly named, should be number of species or something.
#        Even deleted if if must.

import numpy as np
import random, time, argparse
from gillespie_models import MODELS
#import gillespie_analysis as ga unnecessary
import os
import gillespie_models as gm

def gillespie_sample_discrete(probs, r2):
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
    sorted_prob = sorted(probs, reverse=True) # sort list first, might save time
    sorted_idx = sorted(range(len(probs)), key=lambda k: probs[k]
                            , reverse=True)
    while p_sum < r2: #find index
        p_sum += sorted_prob[i]; i += 1
    return sorted_idx[i - 1]

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
    reaction_index = gillespie_sample_discrete(prob_propensity, r2)

    return reaction_index, time

def tau_leaping(Model, simulation, times, current_state, nbr_for_criticality=10.
                    , small_nbr_rxns=10., epsilon=0.03):
    i=1
    while not ( ( Model.stop_condition(current_state,i) ) or
        Model.generation_time_exceed( times[(i-1)%Model.max_gen_save],i-1)):
        #start = time.time()
        updated = False
        # (1) Identify which reactions are critical (number of times they
        # happen before extinction) and a time in which one of these happens (4)
        critical_rxns, tau_pp, propensities = Model.find_critical_rxns(
                                                current_state
                                                , nbr_for_criticality)

        # (2) Find time tau_p in which next non-critical reactions take place,
        # draw the time step
        tau_p = Model.find_time_noncritical_rxns( current_state, critical_rxns
                                                    , propensities, epsilon )

        while not updated: # updated checks if there has been a step taken
            # (3) If tau_p less than some time, do regular SSA for some steps
            # TODO Replace with gillespie function
            if tau_p < small_nbr_rxns/np.sum(propensities):
                j = 1
                while not (  Model.stop_condition(current_state,i) or
                    Model.generation_time_exceed(times[(i-1)%Model.max_gen_save]
                                , i-1) or j > 100):
                    # draw the event and time step
                    reaction_idx, dt = gillespie_draw(Model, current_state)
                    Model.update_results(current_state, dt)

                    # Update the system
                    simulation[i%Model.max_gen_save,:] = current_state + \
                                    Model.update( current_state, reaction_idx );
                    times[i%Model.max_gen_save] =\
                                        times[(i-1)%Model.max_gen_save] + dt;
                    current_state = simulation[i%Model.max_gen_save,:].copy()

                    j += 1; i += 1
                nbr_rxns = 100
                updated = True

            else:
                # (4) Estimate time tau_pp for next critical rxns
                # NOTE : This was done in the previous step (1)

                # (5) Take minimum of tau_p and tau_pp
                if tau_p < tau_pp:
                    tau = tau_p
                    update, nbr_rxns = Model.no_critical_rxns_update(
                                propensities, critical_rxns, current_state, tau)

                else:
                    tau = tau_pp
                    update, nbr_rxns = Model.critical_rxns_update( propensities
                                , critical_rxns, current_state, tau)

                # (6) If negative component, restart at step 3
                next_state = current_state + update
                if np.min(next_state) < 0:
                    tau_p /= 2; updated = False
                else:
                    Model.update_results(current_state, tau)

                    # Update the system
                    # TODO what if system size changes? Going to have to rethink
                    simulation[i%Model.max_gen_save,:] = next_state;
                    times[i%Model.max_gen_save] =\
                                        times[(i-1)%Model.max_gen_save] + tau;
                    current_state = simulation[i%Model.max_gen_save,:].copy()

                    i += 1; updated = True

        """
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(">> Time elapsed : {:0>2}:{:0>2}:{:05.2f}".format( int(hours)
                                                    , int(minutes), seconds) )
        print("     >> Number of reactions : {}".format(nbr_rxns))
        """

def gillespie(Model, simulation, times, current_state):
    i=1
    while not ( ( Model.stop_condition(current_state) ) or
        Model.generation_time_exceed(times[(i-1)%Model.max_gen_save], i-1)):
        start = time.time()
        # draw the event and time step
        reaction_idx, dt = gillespie_draw(Model, current_state)

        Model.update_results(current_state, dt)

        # Update the system
        # TODO what if system size changes? Going to have to rethink this...
        simulation[i%Model.max_gen_save,:] = current_state + Model.update(
                                                current_state, reaction_idx );
        times[i%Model.max_gen_save] = times[(i-1)%Model.max_gen_save] + dt;
        current_state = simulation[i%Model.max_gen_save,:].copy()

        i += 1
        """
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print(end-start)
        print(">> Time elapsed : {:0>2}:{:0>2}:{:05.2f}".format(int(hours)
                    , int(minutes), seconds))
        """


def SSA(Model, traj):
    """
    Running 1 trajectory of Stochastic Simulation Algorithm

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
    simulation[0,:] = init_state.copy()
    current_state = simulation[0,:].copy()

    if not Model.tau: # regular SSM
        gillespie(Model, simulation, times, current_state)

    else: # tau leaping, (Cao, Gillespie, and Petzold, 2006)
        tau_leaping(Model, simulation, times, current_state)

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
            if k == 'sim_dir':
                my_dict[k] = str(v)
            else:
                my_dict[k] = float(v)
        setattr(namespace, self.dest, my_dict)


if __name__ == "__main__":

    np.random.seed( ) # set random seed

    # get commandline arguments
    parser = argparse.ArgumentParser(description = "Simulation of N species")

    #TODO in this new structure, I think number of species is useless... should
    #     be a done differently.
    parser.add_argument('-m', '--model',type = str, default = 'multiLV'
                        , nargs = '?', help = "Model to use.")
    parser.add_argument('-g', '--nbr_rxns', type = int, default = 10**8
                        , nargs = '?'
                        , help = "Number of generations (rxns) in total.")
    parser.add_argument('-T', '--total_time', type = int
                        , default = 10**8, nargs = '?'
                        , help = "Total time to not exceed.")
    parser.add_argument('-t', '--nbr_trajectories', type = int, default = 1
                        , nargs = '?', help = "Number of runs/trajectories.")
    parser.add_argument('-n', '--sim_nbr', type = int, default = 0, nargs = '?'
                        , help = "Simulation number.")
    parser.add_argument('-tau', '--tau_leap', action='store_true'
                        , help = "Use if you want tau-leaping algorithm")
    parser.add_argument('-p', '--params', metavar='KEY=VAL'
                        , default= {'nada' : 0.0}, dest='my_dict', nargs='*'
                        , action=StoreDictKeyPair, required=False
                        ,help='Additional parameters to be passed on for the \
                        simulation')

    # TODO : add multiprocessing. Will make it a lot better. Major changes need to
    #        happen to parallelize all this.


    args = parser.parse_args()
    model = args.model; num_runs = args.nbr_trajectories;
    param_dict = vars(args)['my_dict']
    param_dict['sim_number'] = args.sim_nbr
    param_dict['nbr_generations'] = args.nbr_rxns
    param_dict['max_time'] = args.total_time
    param_dict['tau'] = args.tau_leap


    # select which class/model we are using
    Model = gm.MODELS[model](**param_dict)

    # make directory to save simulation number, change if already exists
    Model.create_sim_folder(); param_dict['sim_number'] = Model.sim_number

    # run gillespie TODO parallelize. Have a couple savepoints?
    for traj in range( num_runs ):
        Model.__init__(**param_dict)
        SSA(Model, traj)
