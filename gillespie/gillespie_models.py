import os; import csv
import numpy as np
from theory_equations import Model_MultiLVim
import random
import pickle

random.seed(42)

RESULTS_DIR = "sim_results"

"""
TODO : Need some explanation of how this works. Give details and such. Working
       examples perheaps, since that might help the reader
"""

class Parent(object):
    def __init__(self,  nbr_generations, max_time, sim_dir
                 , sim_number=0):
        self.sim_number = sim_number; self.sim_dir = sim_dir;
        self.max_time = max_time; self.nbr_generations = nbr_generations;

    #def unpack( self ):
    #    return self.d, self.d_co, self.time, self.dt, self.dx, self.dy, self.dz

    def create_sim_folder( self ):
        # create the directory with simulation results
        save_dir =  os.getcwd() + os.sep + RESULTS_DIR + os.sep +  self.sim_dir;
        save_subdir =  save_dir + os.sep + 'sim';

        # simulation number directory
        i = self.sim_number;
        while os.path.exists( save_subdir + str(i) ): i += 1;
        save_subdir = save_subdir + str(i)

        self.sim_dir = save_dir;
        self.sim_number = i
        self.sim_subdir = save_subdir; os.makedirs( save_subdir )

        return 0

    def generation_time_exceed( self, time, i ):
        if self.nbr_generations < i:
            print('END OF SIM >>>> Exceeded amount of generations permitted: '
                            + str(i) + ' generations')
            return True
        elif self.max_time < time:
            print('END OF SIM >>>> Exceeded amount of generations permitted: '
                            + str(time) + ' time passed')
            return True

        return False

    def save_trajectory( self, simulation, times, traj ):
        """
        For now simply saves each trajectory and some results
        """
        idx_sort = np.argsort(times)
        np.savetxt(self.sim_subdir + os.sep + 'trajectory_%s.txt' %(traj)
                                , simulation[idx_sort,:] )
        np.savetxt(self.sim_subdir + os.sep + 'trajectory_%s_time.txt' %(traj)
                                , times[idx_sort])

        # Save state and some results of the simulation
        del self.results['temp_time']
        self.results['time_btwn_ext'] = np.array(self.results['time_btwn_ext'])
        with open(self.sim_subdir + os.sep + 'results_%s.pickle' %(traj),
                  'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0


###################### MULTISPECIES LOTKA-VOLTERA MODEL ########################

class MultiLV(Parent):
    def __init__( self, nbr_generations=10**6, max_time=10**6, sim_dir='multiLV',
                  birth_rate=20.0, death_rate=1.0, immi_rate=0.05,
                  emmi_rate=0.0, K=100, linear=0.0, quadratic=0.0,
                  comp_overlap=0.5, sim_number=0, nbr_species=30,
                  **kwargs):
        super(MultiLV, self).__init__(nbr_generations, max_time, sim_dir,
                                      sim_number)
        self.birth_rate=birth_rate; self.death_rate=death_rate;
        self.immi_rate=immi_rate; self.emmi_rate=emmi_rate;
        self.carry_capacity=K; self.linear=linear; self.quadratic=quadratic;
        self.comp_overlap=comp_overlap; self.max_gen_save=nbr_generations;
        self.nbr_species=nbr_species

        if 'max_gen_save' in kwargs.keys():
            self.max_gen_save = int(kwargs['max_gen_save'])
        else:
            self.max_gen_save = nbr_generations

        if 'results' in kwargs.keys():
            self.results = kwargs['results']
        else:
            # TODO : Need to calculate things
            #        - distribution
            #        -
            #
            #
            self.results = {'ss_distribution' : np.zeros(self.carry_capacity*4)
                            , 'richness' : np.zeros(self.nbr_species+1)
                            , 'time_btwn_ext' : []
                            , 'temp_time' : np.zeros(self.nbr_species)
                            # 'J' : [],
                            # '' : ,
                            }

    def propensity( self, current_state ):
        """
        Multi species Lotka-Voltera model. For each species, either the species
        gives birth/immigrates or dies/emmigrates. This can be altered for a
        variety of different LV models. Right now we follow a model in which the
        competitive overlap is the same for all species.

        Input:
            current_state : The state that the population is in before the
                            reaction takes place

        Output:
            prop          : array of propensities
        """
        prop = np.zeros( len(current_state)*2 )

        for i in np.arange(0,len(current_state)):
            prop[i*2] = ( current_state[i] * ( self.birth_rate
                        - self.quadratic * ( current_state[i]
                        + self.comp_overlap*np.sum(
                        np.delete(current_state,i)))/self.carry_capacity )
                        + self.immi_rate)
                        # birth + immigration
            prop[i*2+1] = (current_state[i] * ( self.death_rate + self.emmi_rate
                          + ( self.birth_rate - self.death_rate )*( 1
                          - self.quadratic )*(current_state[i]
                          + self.comp_overlap*np.sum(
                          np.delete(current_state,i)))/self.carry_capacity ) )
                          # death + emmigration
        return prop

    def update( self, current_state, idx_reaction):
        """
        When the index of the reaction is chosen, rules for update the
        population

        Input:
            current_state : The state that the population is in before the
                            reaction takes place
            idx_reaction  : index of the reaction to occur

        Output:
            New state of the population
        """
        update = np.zeros( len(current_state) )
        if idx_reaction % 2 == 0:
            update[int(np.floor(idx_reaction/2))] = 1
        else:
            update[int(np.floor(idx_reaction/2))] = -1

        next_state = current_state + update

        return next_state

    def stop_condition( self, current_state ):
        """
        Function that returns True if Gillespie should stop
        """
        return False

    def initialize( self ):
        """
        Inital state of our simulation. Here close to the steady state solution
        """
        # importing equation for steady state
        theory_model = Model_MultiLVim( **(self.__dict__) )
        ss_probability, _ = theory_model.abund_1spec_MSLV()
        ss_cum = np.cumsum(ss_probability)

        # initialize species state
        initial_state = np.zeros( (self.nbr_species) )

        # Create an initial steady state abundance
        for i in np.arange(self.nbr_species):
            sample = random.random()
            j = 0
            while sample > ss_cum[j]:
                j += 1
            initial_state[i] = j

        return initial_state

    def update_results(self, current_state, dt):
        """

        """
        # normalize steady state distribution
        for i in current_state:
            self.results['ss_distribution'][int(i)] += dt

        # normalize richness distribution
        current_richness = np.count_nonzero( current_state )
        self.results['richness'][current_richness] += dt

        # times that species is present. Note temp_time is keeping track of the
        # time a particular species has been present in the system

        # If species just went extinct, add total time they were present
        for i in np.where( np.logical_and(current_state==0,
                           self.results['temp_time']!=0.0)==True)[0]:
            self.results['time_btwn_ext'] += [ self.results['temp_time'][i] ]
            self.results['temp_time'][i] = 0.0

        # add time they are still present in the system
        self.results['temp_time'] += (current_state != 0)*dt

        return 0

    def save_trajectory(self, simulation, times, traj):
        """

        """
        # normalize the distribution
        self.results['ss_distribution'] /= times[-1]

        # normalize the richness
        self.results['richness'] /= times[-1]



        super(MultiLV, self).save_trajectory(simulation, times, traj)

        # with open('filename.pickle', 'rb') as handle:
        #    b = pickle.load(handle)

        return 0


############################## SIR MODEL #############################
class SIR(Parent):
    """
    Stochastic SIR model described by Kamenev and Meerson
    """
    def __init__( self,  nbr_generations, max_time, max_gen_save = 10000,
                  sim_dir='sir', renewal_rate=1.0,
                  infected_death_rate=10.0, total_population=200,
                  beta_rate=20.0, sim_number=0, **kwargs):

        super(SIR, self).__init__( nbr_generations, max_time, sim_dir,
                                  sim_number)
        self.renewal_rate = renewal_rate;
        self.infected_death_rate = infected_death_rate;
        self.total_population = total_population; self.beta_rate = beta_rate;
        self.nbr_generations = nbr_generations;
        self.max_gen_save = max_gen_save; self.max_time = max_time
        self.sim_dir=sim_dir; self.sim_number = sim_number

    def propensity( self, current_state ):
        """
        SIR model, in which Susceptible become Infected.

        Input:
            current_state : 0 -> Susceptible, 1 -> Infected
        Output:
            prop          : array of propensities
        """
        prop = np.zeros( len(current_state)*2 )

        prop[0] = self.renewal_rate * current_state[0] # death of susceptible
        prop[1] = current_state[1] *  self.infected_death_rate # death infected
        prop[2] = self.renewal_rate *  self.total_population # birth susceptible
        prop[3] = ( ( self.beta_rate / self.total_population) \
                  * current_state[1] * current_state[0] ) #  infection

        return prop

    def update( self, current_state, idx_reaction):
        """
        When the index of the reaction is chosen, rules for update the
        population

        Input:
            current_state : The state that the population is in before the
                            reaction takes place
            idx_reaction  : index of the reaction to occur
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

    def stop_condition( self, current_state ):
        """
        Function that returns True if Gillespie should stop
        """
        if current_state[1] == 0:
            return True
        else:
            return False

    def initialize( self ):
        """
        Inital state of our simulation. Here close to the steady state solution,
        see Kamenev and Meerson
        """
        initial_state = np.zeros( self.nbr_species, dtype=int )
        initial_state[0] = np.int( (self.infected_death_rate/self.beta_rate)
                           * self.total_population )
        initial_state[1] = np.int( self.renewal_rate * self.total_population
                           * ( self.beta_rate - self.infected_death_rate )
                           / (self.beta_rate * self.infected_death_rate) )
        return initial_state

####################### BE SURE TO ADD YOUR MODEL HERE #########################

MODELS = {'multiLV' : MultiLV, 'sir' : SIR}
