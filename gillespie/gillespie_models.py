import os; import csv
import numpy as np

RESULTS_DIR = "sim_results"

"""
Need some explanation of how this works.
"""


class Parent(object):
    def __init__(self,  num_generations, max_time, sim_dir='default', sim_number=0):
        self.sim_dir = sim_dir
        self.sim_number = sim_number

    #def unpack( self ):
    #    return self.d, self.d_co, self.time, self.dt, self.dx, self.dy, self.dz

    def create_sim_folder( self ):
        # create the directory with simulation results
        save_dir =  os.getcwd() + os.sep + RESULTS_DIR + os.sep +  self.sim_dir;
        save_subdir =  save_dir + os.sep + 'sim';

        # figures subfolder
        self.figure_dir = save_dir + os.sep + 'figures'
        if not os.path.exists( self.figure_dir ) and self.sim_number == 0:
            os.makedirs(self.figure_dir)

        # simulation number directory
        i = self.sim_number;
        while os.path.exists( save_subdir + str(i) ): i += 1;
        save_subdir = save_subdir + str(i)

        self.sim_dir = save_dir;
        self.sim_subdir = save_subdir; os.makedirs( save_subdir )

        # save the parameters of simulation
        dict = self.__dict__
        w = csv.writer( open(self.sim_subdir + os.sep + "params.csv", "w"))
        for key, val in dict.items():
            w.writerow([key,val])

        return 0

    def generation_time_exceed( self, time, i ):
        if self.num_generations < i:
            print('END OF SIM >>>> Exceeded amount of generations permitted: ' + str(i) + ' generations')
            return True
        elif self.max_time < time:
            print('END OF SIM >>>> Exceeded amount of generations permitted: '  + str(time) + ' time passed')
            return True
        return False

    def save_trajectory( self, simulation, times, traj ):
        """
        For now simply saves each trajectory.
        """
        idx_sort = np.argsort(times)
        np.savetxt(self.sim_subdir + os.sep + 'trajectory_%s.txt' %(traj), simulation[idx_sort,:] )
        np.savetxt(self.sim_subdir + os.sep + 'trajectory_%s_time.txt' %(traj), times[idx_sort])
        return 0


############################## MULTISPECIES LOTKA-VOLTERA MODEL #############################

class MultiLV(Parent):
    def __init__( self, num_generations, max_time, sim_dir='multiLV', birth_rate=20.0, death_rate=1.0, immi_rate=0.05, emmi_rate=0.0, K=100, linear=0.0, quadratic=0.0, comp_overlap=0.5, sim_number=0, **kwargs):

        self.birth_rate=birth_rate; self.death_rate=death_rate; self.immi_rate=immi_rate; self.emmi_rate=emmi_rate; self.K=K; self.linear=linear; self.quadratic=quadratic; self.sim_dir=sim_dir; self.comp_overlap=comp_overlap; self.num_generations = num_generations; self.max_gen_save = num_generations; self.max_time = max_time
        self.sim_dir=sim_dir; self.sim_number = sim_number

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

    def stop_condition( self, current_state, i ):
        """
        Function that returns True if Gillespie should stop
        """
        return False

    def initialize( self, num_states ):
        """
        Inital state of our simulation. Here close to the steady state solution
        """
        initial_state = np.zeros( (num_states) ) #necessary, everything 0
        initial_state += int(self.K*( 1 + np.sqrt( 1 + 4*self.immi_rate*( self.comp_overlap*( num_states - 1 ) + 1 )/( self.K*(self.birth_rate-self.death_rate) ) ) )/( 2*( self.comp_overlap*( num_states - 1 ) + 1 ) ))
        return initial_state

############################## SIR MODEL #############################
class SIR(Parent):
    """
    stochastic SIR model described by Kamenev and Meerson
    """
    def __init__( self,  num_generations, max_time, max_gen_save = 10000, sim_dir='sir', renewal_rate=1.0, infected_death_rate=10.0, total_population=200, beta_rate=20.0, sim_number=0, **kwargs):

        self.renewal_rate = renewal_rate; self.infected_death_rate = infected_death_rate; self.total_population = total_population; self.beta_rate = beta_rate; self.num_generations = num_generations; self.max_gen_save = max_gen_save; self.max_time = max_time
        self.sim_dir=sim_dir; self.sim_number = sim_number

    def propensity( self, current_state ):
        """
        SIR model, in which Susceptible become Infected.

        Input:
            current_state : 0 -> Susceptible, 1 -> Infected
        Output:
            prop : array of propensities
        """
        prop = np.zeros( len(current_state)*2 )

        prop[0] = current_state[0] * self.renewal_rate
        prop[1] = current_state[1] *  self.infected_death_rate
        prop[2] = self.renewal_rate *  self.total_population
        prop[3] = ( self.beta_rate / self.total_population) * current_state[1] * current_state[0]

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
        if idx_reaction == 0:
            update[0] = -1
        elif idx_reaction == 1:
            update[1] = -1
        elif idx_reaction == 2:
            update[0] = 1
        elif idx_reaction == 3:
            update[0] = -1; update[1] = 1;
        else:
            print('invalid reaction')
            raise SystemExit

        return current_state + update

    def stop_condition( self, current_state, i ):
        """
        Function that returns True if Gillespie should stop
        """
        if current_state[1] == 0:
            return True
        else:
            return False

    def initialize( self, num_states ):
        """
        Inital state of our simulation. Here close to the steady state solution, see Kamenev and Meerson
        """
        initial_state = np.zeros( 2, dtype=int ) #necessary, everything 0
        initial_state[0] = np.int( (self.infected_death_rate/self.beta_rate) * self.total_population )
        initial_state[1] = np.int( self.renewal_rate * self.total_population * ( self.beta_rate - self.infected_death_rate ) / (self.beta_rate * self.infected_death_rate) )
        return initial_state


############################## BE SURE TO ADD YOUR MODEL HERE #############################

MODELS = {'multiLV' : MultiLV, 'sir' : SIR}
