import os, csv, sys
import numpy as np
from theory_equations import Model_MultiLVim
import random
import pickle

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
    python3 gillespie.py

"""

RESULTS_DIR = "sim_results"

class Parent(object):
    def __init__(self,  nbr_generations, max_time, sim_dir
                 , sim_number=0):
        self.sim_number = sim_number; self.sim_dir = sim_dir;
        self.max_time = max_time; self.nbr_generations = nbr_generations;

    #def unpack( self ):
    #    return self.d, self.d_co, self.time, self.dt, self.dx, self.dy, self.

    def create_sim_folder( self ):
        # create the directory with simulation results
        tau = ''
        if self.tau: tau='tau'
        save_dir =  os.getcwd() + os.sep + RESULTS_DIR + os.sep +  self.sim_dir\
                    + tau;
        save_subdir =  save_dir + os.sep + 'sim';

        # simulation number directory
        i = self.sim_number;
        while os.path.exists( save_subdir + tau + str(i) ): i += 1;
        save_subdir = save_subdir + tau + str(i)


        self.sim_dir = save_dir;
        self.sim_number = i
        random.seed(i)
        self.sim_subdir = save_subdir; os.makedirs( save_subdir )

        return 0

    def generation_time_exceed( self, time, i ):
        if self.nbr_generations < i:
            print('END OF SIM >>>> Exceeded amount of generations permitted: '
                            + str(i) + ' generations')
            return True
        elif self.max_time < time:
            print('END OF SIM >>>> Exceeded amount of time permitted: '
                            + str(time) + ' time passed')
            return True

        return False

    def update_results(self, current_state, dt):

        if idx == 1:
            print("Warning : Using Parent update_results function.")
            print("          No results tracked and updated")

        return 0

    def stop_condition( self, current_state, idx ):

        if idx == 1:
            print("Warning : Using Parent stop_condition function.")
            print("          No stop condition other than time and generation.")

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
        self.results['time_btwn_ext'] = np.array(self.results['time_btwn_ext'])
        self.results['times'] = times
        self.results['trajectory'] = simulation
        with open(self.sim_subdir + os.sep + 'results_%s.pickle' %(traj),
                  'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0


    def find_critical_rxns( self, current_state, nbr_for_criticality=10.):
        """
        Identify the reactions which are critical, that is to say that if you
        do them (nbr_for criticality) times, they will give you a negative
        amount of a product

        Input :
            current_state       : vector of current products
            nbr_for_criticality : number of times a reactions can happen before
                                  it makes it the product negative
        Output :
            critical_rxns       : vector of length # of rxns. 1 if reaction is
                                  critical, 0 if not.
        """
        propensities = self.propensity(current_state)
        critical_rxns = np.zeros(len(propensities))

        for idx in [i for i, x in enumerate(propensities) if x>0]:
            if (0 > np.min( nbr_for_criticality*self.update( current_state, idx)
                                + current_state )):
                critical_rxns[idx] = 1

        # Generate a random time in which the next critical reaction happens
        r1 = np.random.random(1);
        time = - np.log( r1 )/np.dot( propensities, critical_rxns )

        return critical_rxns, time, propensities

    def no_critical_rxns_update( self, propensities, critical_rxns
                                    , current_state, tau):
        """
        Make an update vector for reactions that are not critical.

        Input :
            propensities    : list of propensities
            critical_rxns   : list of critical_rxns
            current_state   : current state
            tau             : time it takes for all the reactions to happen

        Output :
            update          : update to current state
        """
        non_critical_rxns = 1 - critical_rxns
        nbr_rxns = 0

        # select only the non-critical reactions
        for idx in [i for i, x in enumerate(non_critical_rxns) if x==1]:
            # multiplicity of all each non critical reaction ( drawn from a
            # poisson distribution )
            multiplicity_rxn = np.random.poisson( propensities[idx]*tau, 1 )[0]
            # update the reaction
            update += multiplicity_rxn * self.update( current_state, idx )
            nbr_rxns += multiplicity_rxn

        return update, nbr_rxns

    def critical_rxns_update( self, propensities, critical_rxns, current_state
                                , tau):
        """
        Make an update vector for one critical reaction and all the other
        reactions that happen in that time too.

        Input :
            propensities    : list of propensities
            critical_rxns   : list of critical_rxns
            current_state   : current state
            tau             : time it takes for all the reactions to happen

        Output :
            update          : update to current state
        """
        # probabilities of critical reactions (other reactions set to 0)
        probs = ( propensities*critical_rxns ) / np.dot( propensities
                                                        , critical_rxns)
        i = 0; p_sum = 0.0
        sorted_prob = sorted(probs, reverse=True) # sort list first, saves time
        sorted_idx = sorted(range(len(probs)), key=lambda k: probs[k]
                                , reverse=True)

        # pick the critical reaction to happen
        r1 = np.random.random(1);
        while p_sum < r1: #find index
            p_sum += sorted_prob[i]; i += 1

        # make this critical reaction happen, find update rule
        update = self.update( current_state, sorted_idx[i - 1])
        nbr_rxns = 1

        # update rules of all the non-critical reactions to happen
        update_temp, nbr_rxns_temp = self.no_critical_rxns_update(propensities
                                            , critical_rxns, current_state, tau)
        update += update_temp; nbr_rxns += nbr_rxns_temp

        return update, nbr_rxns

###################### MULTISPECIES LOTKA-VOLTERA MODEL ########################

class MultiLV(Parent):
    def __init__( self, nbr_generations=10**6, max_time=10**6, sim_dir='multiLV'
                  , tau=False, birth_rate=2.0, death_rate=1.0, immi_rate=0.1
                  , emmi_rate=0.0, carry_capacity=100, linear=0.0, quadratic=0.0
                  , comp_overlap=0.2, sim_number=0, nbr_species=30
                  , **kwargs):
        super(MultiLV, self).__init__(nbr_generations, max_time, sim_dir
                                      ,sim_number)
        self.birth_rate=birth_rate; self.death_rate=death_rate; self.tau=tau
        self.immi_rate=immi_rate; self.emmi_rate=emmi_rate;
        self.carry_capacity=int(carry_capacity); self.linear=linear;
        self.quadratic=quadratic; self.comp_overlap=comp_overlap;
        self.max_gen_save=nbr_generations; self.nbr_species=int(nbr_species);

        if 'max_gen_save' in kwargs.keys():
            self.max_gen_save = int(kwargs['max_gen_save'])
        else:
            self.max_gen_save = nbr_generations

        if 'results' in kwargs.keys():
            self.results = kwargs['results']
        else:
            # TODO : Add other results?
            self.results = {'ss_distribution' : np.zeros(self.carry_capacity*5)
                            , 'richness' : np.zeros(self.nbr_species+1)
                            , 'time_btwn_ext' : []
                            , 'temp_time' : np.zeros(self.nbr_species)
                            , 'joint_temp' : np.zeros( (self.carry_capacity*5
                                            ,self.carry_capacity*5) )
                            , 'conditional' : np.zeros( (self.carry_capacity*5
                                            ,self.carry_capacity*5) )
                            , 'av_ni_nj_temp' : np.zeros( ( self.nbr_species,
                                                        self.nbr_species))
                            , 'av_ni_temp' : np.zeros(self.nbr_species)
                            , 'av_ni_sq_temp' : np.zeros(self.nbr_species)
                            , 'corr_ni_nj' : 0.0, 'coeff_ni_nj' : 0.0
                            , 'av_J' : 0.0, 'av_J_sq' : 0.0
                            , 'av_J_n' : np.zeros(self.nbr_species)
                            , 'corr_J_n' : 0.0, 'coeff_J_n' :0.0
                            , 'av_Jminusn' : np.zeros(self.nbr_species)
                            , 'av_Jminusn_sq' : np.zeros(self.nbr_species)
                            , 'av_Jminusn_n' : np.zeros(self.nbr_species)
                            , 'corr_Jminusn_n' : 0.0, 'coeff_Jminusn_n' : 0.0
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

    def update( self, current_state, idx_reaction ):
        """
        When the index of the reaction is chosen, rules for update the
        population

        Input:
            current_state : The state that the population is in before the
                            reaction takes place
            idx_reaction  : index of the reaction to occur

        Output:
            Update rule
        """
        update = np.zeros( len(current_state) )
        if idx_reaction % 2 == 0:
            update[int(np.floor(idx_reaction/2))] = 1
        else:
            update[int(np.floor(idx_reaction/2))] = -1

        return update

    def initialize( self ):
        """
        Inital state of our simulation. Here close to the steady state solution
        """
        # importing equation for steady state
        theory_model = Model_MultiLVim( **(self.__dict__) )
        ss_probability, _ = theory_model.abund_sid()
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
        #sys.exit()
        initial_state[np.argmin(initial_state)] = 0

        return initial_state

    def update_results( self, current_state, dt ):
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

        # Tally the joint distribution P(n_i,n_j) and temporary variables for
        # correlation calculation
        J = np.sum(current_state)
        self.results['av_J'] += dt*J
        self.results['av_J_sq'] += dt*J**2
        self.results['av_J_n'] += dt*J*current_state
        self.results['av_Jminusn'] += dt*(J-current_state)
        self.results['av_Jminusn_sq'] += dt*np.square(J-current_state)
        self.results['av_Jminusn_n'] += dt*(J-current_state)*current_state
        for i, value_i in enumerate(current_state):
            self.results['av_ni_temp'][i] += dt*value_i
            self.results['av_ni_sq_temp'][i] += dt*value_i**2

            for j, value_j in enumerate(current_state[i+1:]):
                self.results['joint_temp'][int(value_i)][int(value_j)] += dt
                self.results['av_ni_nj_temp'][i][i+j+1] += dt*value_i*value_j

        return 0

    def save_trajectory( self, simulation, times, traj ):
        """
        Explain what is going on
        """
        # normalize the distribution
        total_time = np.max(times)
        self.results['ss_distribution'] /= total_time

        # normalize the richness
        self.results['richness'] /= total_time


        # Calculate conditional probability
        self.results['joint_temp'] /= np.sum( self.results['joint_temp'] )
        for i in np.arange(0, np.shape(self.results['joint_temp'])[0]):
            self.results['conditional'][i][i] = \
                                        self.results['joint_temp'][i][i]
                                        # I don't think should be double
            for j in np.arange(i+1,np.shape(self.results['joint_temp'])[1]):
                self.results['conditional'][j][i] = \
                                ( self.results['joint_temp'][i][j]\
                                        + self.results['joint_temp'][j][i] )
                self.results['conditional'][i][j] = \
                                ( self.results['joint_temp'][i][j]\
                                        + self.results['joint_temp'][j][i] )

        self.results['conditional'] /= np.sum(self.results['conditional'])

        for i in np.arange(0, np.shape(self.results['conditional'])[0]):
            if self.results['ss_distribution'][i] != 0:
                self.results['conditional'][:][i] /=\
                                        self.results['ss_distribution'][i]


        # Calculate correlation
        self.results['av_ni_temp'] /= total_time
        self.results['av_ni_sq_temp'] /= total_time
        self.results['av_ni_nj_temp'] /= total_time
        self.results['av_J'] /= total_time
        self.results['av_J_sq'] /= total_time
        self.results['av_J_n'] /= total_time
        self.results['av_Jminusn'] /= total_time
        self.results['av_Jminusn_sq'] /= total_time
        self.results['av_Jminusn_n'] /= total_time
        nbr_correlations = 0

        print( self.results['av_ni_sq_temp'],self.results['av_ni_temp']**2 )

        for i in np.arange(0, self.nbr_species):
            var_J_n = ( ( np.sqrt( self.results['av_ni_sq_temp'][i]
                        - self.results['av_ni_temp'][i]**2 ) ) * (
                        np.sqrt( self.results['av_J_sq']
                        - self.results['av_J']**2 ) ) )
            var_Jminusn_n = ( ( np.sqrt( self.results['av_ni_sq_temp'][i]
                            - self.results['av_ni_temp'][i]**2 ) ) * (
                            np.sqrt( self.results['av_Jminusn_sq'][i]
                            - self.results['av_Jminusn'][i]**2 ) ) )
            # cov(J,n)
            cov_J_n = (self.results['av_J_n'][i] - self.results['av_ni_temp'][i]
                        * self.results['av_J'] )
            # cov(J-n,n)
            cov_Jminusn_n = (self.results['av_Jminusn_n'][i]
                            - self.results['av_ni_temp'][i] *
                            self.results['av_Jminusn'][i] )
            # coefficients of variation
            if self.results['av_J_n'][i] != 0.0:
                self.results['coeff_J_n']+= cov_J_n / self.results['av_J_n'][i]
            if self.results['av_Jminusn_n'][i] != 0.0:
                self.results['coeff_Jminusn_n'] += ( cov_Jminusn_n
                                            / self.results['av_Jminusn_n'][i] )
            # Pearson correlation
            if var_J_n != 0.0:
                self.results['corr_J_n'] += cov_J_n / var_J_n
            if var_Jminusn_n != 0.0:
                self.results['corr_Jminusn_n'] += cov_Jminusn_n / var_Jminusn_n

            for j in np.arange(i+1, self.nbr_species):
                var_nm = ( ( np.sqrt( self.results['av_ni_sq_temp'][i]
                - self.results['av_ni_temp'][i]**2 ) ) * (
                np.sqrt( self.results['av_ni_sq_temp'][j]
                - self.results['av_ni_temp'][j]**2 ) ) )
                # cov(n_i,n_j)
                cov_ni_nj = ( self.results['av_ni_nj_temp'][i][j]
                            - self.results['av_ni_temp'][i] *
                            self.results['av_ni_temp'][j] )
                # coefficients of variation
                if self.results['av_ni_nj_temp'][i][j] != 0.0:
                    self.results['coeff_ni_nj'] += ( cov_ni_nj
                                        / self.results['av_ni_nj_temp'][i][j] )
                # Pearson correlation
                if var_nm != 0.0:
                    self.results['corr_ni_nj'] += cov_ni_nj / var_nm
                    nbr_correlations += 1

        # Taking the average over all species
        self.results['corr_ni_nj'] /= ( self.nbr_species*(self.nbr_species-1)/2)
        self.results['coeff_ni_nj'] /= ( self.nbr_species*(self.nbr_species-1)/2)
        self.results['corr_J_n'] /= ( self.nbr_species)
        self.results['coeff_J_n'] /= ( self.nbr_species)
        self.results['corr_Jminusn_n'] /= ( self.nbr_species)
        self.results['coeff_Jminusn_n'] /= ( self.nbr_species)

        #del self.results['av_ni_temp'], self.results['av_ni_sq_temp']\
        #    , self.results['av_ni_nj_temp'], self.results['joint_temp']

        super(MultiLV, self).save_trajectory(simulation, times, traj)

        # TODO : Check I can get rid of.
        # with open('filename.pickle', 'rb') as handle:
        #    b = pickle.load(handle)
        #print(self.results['corr_ni_nj'])

        return 0

    def find_time_noncritical_rxns( self, current_state, critical_rxns\
                                    , propensities, epsilon):
        """
        Check paper by Gillespie et al. 2006
        """

        nc_propensities = (1-critical_rxns)*propensities
        min_death_change = np.min([self.death_rate, self.comp_overlap*(
                            self.birth_rate - self.death_rate)
                            /self.carry_capacity,( self.birth_rate
                            - self.death_rate)/ self.carry_capacity] )
        tau = 10E10
        for i, pop_nbr in enumerate( current_state ):
            positive_rxn = 2*i; negative_rxn = 2*i+1
            # f_ij for the birth reactions (only terms that will be non zero)
            f_ij_birth = np.zeros( len(propensities) );
            f_ij_birth[positive_rxn] = self.birth_rate;
            f_ij_birth[negative_rxn] = - self.birth_rate;

            # f_ij for the death reactions (only terms that will be non zero)
            f_ij_death = self.comp_overlap*( self.birth_rate - self.death_rate
                        )/self.carry_capacity * np.ones( len(propensities) );
            f_ij_death[positive_rxn] = ( f_ij_death[positive_rxn]
                                        /self.comp_overlap + self.death_rate);
            f_ij_death[negative_rxn] = - ( f_ij_death[positive_rxn]
                                        /self.comp_overlap + self.death_rate);

            # calculate auxilary quantity mu from step (2)
            mu_j_birth = np.abs( np.dot( f_ij_birth, nc_propensities ) )
            mu_j_death = np.abs( np.dot( f_ij_death, nc_propensities ) )

            # calculate auxilary quantity sigma from step (2)
            sigma_birth = np.abs( np.dot( np.square(f_ij_birth)
                                                , nc_propensities) );
            sigma_death = np.abs( np.dot(np.square(f_ij_death)
                                                , nc_propensities) );

            # maximum between epsilon*reaction and the minimum increment of the
            # reaction
            max_birth = np.max( [ epsilon*propensities[positive_rxn]
                                    , self.birth_rate ] )
            max_death = np.max( [ epsilon*propensities[negative_rxn]
                                , min_death_change ] )

            # Minimum time in which significant change happens
            tau = np.min([tau, max_birth/mu_j_birth, max_birth**2/sigma_birth
                            , max_death/mu_j_death, max_death**2/sigma_death])

        return tau


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

        return update

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
