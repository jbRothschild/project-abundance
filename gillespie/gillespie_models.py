import numpy as np
import os

BIRTH_RATE = 0.5
DEATH_RATE = 0.5
IMMIGRATION_RATE = 0.1
EMMIGRATION_RATE = 0.0
CARRYING_CAPACITY = 100
LINEAR_TERM = 0.0
QUADRATIC_TERM = 0.0

"""
Need some explanation of how this works.
"""

class Simple(object):
    def __init__( self,  sim_dir='simple', birth_rate=BIRTH_RATE, death_rate=DEATH_RATE, immi_rate=IMMIGRATION_RATE, emmi_rate=EMMIGRATION_RATE, K=CARRYING_CAPACITY, linear=LINEAR_TERM, quadratic=QUADRATIC_TERM, **kwargs):

        self.birth_rate=birth_rate; self.death_rate=death_rate; self.immi_rate=immi_rate; self.emmi_rate=emmi_rate; self.K=K; self.linear=linear; self.quadratic=quadratic; self.sim_dir=sim_dir

    def save_directory( self ):
        save_dir =  os.getcwd() + os.sep + "results" + os.sep + self.sim_dir
        if not os.path.exists( save_dir ):
            os.makedirs( save_dir )
        return save_dir

    # to build dictionary from a class, simply do Class.__dict__

    #def unpack( self ):
    #    return self.d, self.d_co, self.time, self.dt, self.dx, self.dy, self.dz

    def simple_propensity( self, current_state ):
        """
        Example of how to define propensities. Here we have 4 reactions possible per stateself.

        Input:
            current_state : The state that the population is in before the reaction takes place

        Output:
            prop : array of propensities
        """
        prop = np.zeros( len(current_state)*4 )

        for i in np.arange(0,len(current_state)):
            prop[i*4] = self.birth_rate*current_state[i] # birth
            prop[i*4+1] = self.death_rate*current_state[i]**2 # death
            prop[i*4+2] = self.immi_rate # immigration
            prop[i*4+3] = self.emmi_rate*current_state[i] # emmigration

        return prop

    def simple_update( self, current_state, idx_reaction ):
        """
        When the index of the reaction is chosen, rules for update the population

        Input:
            current_state : The state that the population is in before the reaction takes place
            idx_reaction : index of reaction to take place in prop (see simple_propensity)

        Output:
            Array which is the new state of the population
        """
        update = np.zeros( len(current_state) )
        if idx_reaction % 2 == 0:
            update[int(np.floor(idx_reaction/4))] = 1
        else:
            update[int(np.floor(idx_reaction/4))] = -1

        return current_state + update

    def simple_initialize( num_states, **kwargs):
        """
        When the index of the reaction is chosen, rules for update the population

        Input:
            num_states : number of species

        Output:
            Array which is the new state of the population
        """
        initial_state = np.zeros( (num_states) ) #necessary, everything 0
        initial_state += 10 #make substitutions
        return initial_state







class MultiLV(object):
    def __init__( self,  sim_dir='multiLV', birth_rate=BIRTH_RATE, death_rate=DEATH_RATE, immi_rate=IMMIGRATION_RATE, emmi_rate=EMMIGRATION_RATE, K=CARRYING_CAPACITY, linear=LINEAR_TERM, quadratic=QUADRATIC_TERM, comp_overlap=0.0, **kwargs):

        self.birth_rate=birth_rate; self.death_rate=death_rate; self.immi_rate=immi_rate; self.emmi_rate=emmi_rate; self.K=K; self.linear=linear; self.quadratic=quadratic; self.sim_dir=sim_dir; self.comp_overlap=comp_overlap

        if not os.path.exists( os.getcwd() + os.sep + "results" + os.sep + self.sim_dir ):
            os.makedirs(os.getcwd() + os.sep + "results" + os.sep + self.sim_dir)

    def save_directory( self ):
        save_dir =  os.getcwd() + os.sep + "results" + os.sep + self.sim_dir
        if not os.path.exists( save_dir ):
            os.makedirs( save_dir )
        return save_dir

    # to build dictionary from a class, simply do Class.__dict__

    #def unpack( self ):
    #    return self.d, self.d_co, self.time, self.dt, self.dx, self.dy, self.dz

    def propensity( self, current_state ):
        """
        Multi species Lotka-Voltera model. For each species, either the species gives birth/immigrates or dies/emmigrates. This can be altered for a variety of different LV models. Right now we follow a model in which the competitive overlap is the same for all species.

        Input:
            current_state : The state that the population is in before the reaction takes place
            kwargs :
        Output:
            prop : array of propensities
        """
        prop = np.zeros( len(current_state)*2 )

        for i in np.arange(0,len(current_state)):
            prop[i*2] = current_state[i] * ( self.birth_rate - self.quadratic*(current_state[i] + self.comp_overlap*np.sum(np.delete(current_state,i)))/self.K ) + self.immi_rate # birth + immigration
            prop[i*2+1] = current_state[i] * ( self.death_rate + self.emmi_rate + ( self.birth_rate + self.death_rate )*( 1 - self.quadratic )*(current_state[i] + self.comp_overlap*np.sum(np.delete(current_state,i)))/self.K )  # death + emmigration

        return prop

    def update( self, current_state, idx_reaction):
        """
        When the index of the reaction is chosen, rules for update the population

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
        Inital state of our simulation.
        """
        print(int(self.K*( 1 + np.sqrt( 1 + 4*self.immi_rate*( self.comp_overlap*( num_states - 1 ) + 1 )/self.K ) )/( 2*( self.comp_overlap*( num_states - 1 ) + 1 ) )))
        initial_state = np.zeros( (num_states) )
        initial_state += int(self.K*( 1 + np.sqrt( 1 + 4*self.immi_rate*( self.comp_overlap*( num_states - 1 ) + 1 )/self.K ) )/( 2*( self.comp_overlap*( num_states - 1 ) + 1 ) ))
        return initial_state

############################## BE SURE TO ADD YOUR MODEL HERE #############################

MODELS = {'simple' : Simple, 'multiLV' : MultiLV}
