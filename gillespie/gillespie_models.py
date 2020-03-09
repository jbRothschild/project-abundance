import os; import csv
import numpy as np
from matplotlib import animation # animation

BIRTH_RATE = 20.0
DEATH_RATE = 1.0
IMMIGRATION_RATE = 0.05
EMMIGRATION_RATE = 0.0
CARRYING_CAPACITY = 100
LINEAR_TERM = 0.0
QUADRATIC_TERM = 0.0

RESULTS_DIR = "sim_results"

"""
Need some explanation of how this works.
"""

class MultiLV(object):
    def __init__( self,  sim_dir='multiLV', birth_rate=BIRTH_RATE, death_rate=DEATH_RATE, immi_rate=IMMIGRATION_RATE, emmi_rate=EMMIGRATION_RATE, K=CARRYING_CAPACITY, linear=LINEAR_TERM, quadratic=QUADRATIC_TERM, comp_overlap=0.0, **kwargs):

        self.birth_rate=birth_rate; self.death_rate=death_rate; self.immi_rate=immi_rate; self.emmi_rate=emmi_rate; self.K=K; self.linear=linear; self.quadratic=quadratic; self.sim_dir=sim_dir; self.comp_overlap=comp_overlap;
        self.sim_dir=sim_dir

    def create_sim_folder( self ):
        # create the directory with simulation results
        save_dir =  os.getcwd() + os.sep + RESULTS_DIR + os.sep + self.sim_dir; i = 0;
        while os.path.exists( save_dir + str(i) ): i += 1;
        save_dir = save_dir + str(i)

        self.sim_dir = save_dir; os.makedirs( save_dir )

        # save the parameters of simulation
        dict = self.__dict__
        w = csv.writer( open(self.sim_dir + os.sep + "params.csv", "w"))
        for key, val in dict.items():
            w.writerow([key,val])

        # figures subfolder
        self.figure_dir = self.sim_dir + os.sep + 'figures'
        if not os.path.exists( self.figure_dir):
            os.makedirs(self.figure_dir)

        return 0

    #def unpack( self ):
    #    return self.d, self.d_co, self.time, self.dt, self.dx, self.dy, self.dz

    def propensity( self, current_state ):
        """
        Multi species Lotka-Voltera model. For each species, either the species gives birth/immigrates or dies/emmigrates. This can be altered for a variety of different LV models. Right now we follow a model in which the competitive overlap is the same for all species.

        Input:
            current_state : The state that the population is in before the reaction takes place
        Output:
            prop : array of propensities
        """
        prop = np.zeros( len(current_state)*2 )

        for i in np.arange(0,len(current_state)):
            prop[i*2] = current_state[i] * ( self.birth_rate - self.quadratic*(current_state[i] + self.comp_overlap*np.sum(np.delete(current_state,i)))/self.K ) + self.immi_rate # birth + immigration
            prop[i*2+1] = current_state[i] * ( self.death_rate + self.emmi_rate + ( self.birth_rate - self.death_rate )*( 1 - self.quadratic )*(current_state[i] + self.comp_overlap*np.sum(np.delete(current_state,i)))/self.K )  # death + emmigration

        return prop

    def update( self, current_state, idx_reaction):
        """
        When the index of the reaction is chosen, rules for update the population
        Input:
            current_state : The state that the population is in before the reaction takes place
            idx_reaction : index of the reaction to occur
        Output:
            New state of the population
        """
        update = np.zeros( len(current_state) )
        if idx_reaction % 2 == 0:
            update[int(np.floor(idx_reaction/2))] = 1
        else:
            update[int(np.floor(idx_reaction/2))] = -1

        return current_state + update

    def initialize( self, num_states ):
        """
        Inital state of our simulation. Here close to the steady state solution
        """
        initial_state = np.zeros( (num_states) ) #necessary, everything 0
        initial_state += int(self.K*( 1 + np.sqrt( 1 + 4*self.immi_rate*( self.comp_overlap*( num_states - 1 ) + 1 )/( self.K*(self.birth_rate-self.death_rate) ) ) )/( 2*( self.comp_overlap*( num_states - 1 ) + 1 ) ))
        return initial_state

    def save_trajectory( self, simulation, times, traj ):
        """
        For now simply saves each trajectory.
        """
        np.savetxt(self.sim_dir + os.sep + 'trajectory_%s.txt' %(traj), simulation )
        np.savetxt(self.sim_dir + os.sep + 'trajectory_%s_time.txt' %(traj), times)
        return 0

############################## BE SURE TO ADD YOUR MODEL HERE #############################

MODELS = {'multiLV' : MultiLV}
