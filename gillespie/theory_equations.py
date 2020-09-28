import numpy as np; import csv, os, glob, csv, time, datetime

import matplotlib as mpl; import matplotlib.pyplot as plt
import matplotlib.patches as patches # histgram animation
import matplotlib.path as path # histogram animation
from matplotlib import animation # animation
from matplotlib.colors import LogNorm

import itertools
from itertools import combinations

from scipy.special import gamma, poch, factorial, hyp1f1 # certain functions
import scipy, scipy.stats
import scipy.io as spo
from scipy.optimize import fsolve as sp_solver # numerical solver, can change to
                                               # a couple others, i.e. broyden1
#import seaborn as sns
from settings import VAR_NAME_DICT

plt.style.use('custom.mplstyle')

# TODO : all these scripts are a bit of a mess. FIgure out a good modular way
#        to do this all.

THRY_FIG_DIR = 'figures' + os.sep + 'theory'
#THEORY_DIR = 'theory_results'

while not os.path.exists( os.getcwd() + os.sep + THRY_FIG_DIR ):
    os.makedirs(os.getcwd() + os.sep + THRY_FIG_DIR);

class Model_MultiLVim(object):
    """
    Various solutions from different models of populations with competition in
    the death rate. Will describe where they come from each.
    """
    def __init__(self, comp_overlap=0.9, birth_rate=2.0, death_rate=1.0,
                 carry_capacity=50, immi_rate=0.01, nbr_species=30,
                 population=np.arange(250), **kwargs ):
        self.comp_overlap = comp_overlap; self.birth_rate = birth_rate;
        self.death_rate = death_rate; self.immi_rate = immi_rate;
        self.carry_capacity = carry_capacity; self.nbr_species = nbr_species;
        self.population = np.arange(10*self.carry_capacity);
        self.total_population = np.size(self.population)*nbr_species;

    def __setattr__(self, name, value):
        """
        Updates population size to
        """
        self.__dict__[name] = value;
        if name == "carry_capacity":
            self.population = np.arange(10*self.carry_capacity);

    def dstbn_n_given_J(self, J):
        """
        Marginal Distribution of n given J.

        Return
            Marginal probability distribution P(n|J)
        """
        a       = self.immi_rate / self.birth_rate;
        b       = 1.0 + self.death_rate * self.carry_capacity / (self.birth_rate
                  - self.death_rate);
        c       = self.birth_rate * self.carry_capacity / (self.birth_rate
                  - self.death_rate);
        b_tilde = 1.0 + ( self.death_rate*self.carry_capacity + (self.birth_rate
                  - self.death_rate ) * self.comp_overlap * J ) / (
                  ( self.birth_rate - self.death_rate) * (1-self.comp_overlap ))
        c_tilde = c / (1-self.comp_overlap)

        # This is complicated and doesn't work (factorial too large!)
        #return ( ( poch(a,self.population) * np.power(c_tilde,self.population))
        #       /( factorial(self.population) * poch(b_tilde+1,self.population)
        #       * hyp1f1(a, b_tilde, c_tilde) ) )

        prob_n_given_J = np.zeros( np.shape(self.population))
        #prob_n_given_J[0] = 1/hyp1f1(a, b_tilde, c_tilde)
        prob_n_given_J[0] = 1.0 #TODO which to use

        for i in np.arange(1,len(prob_n_given_J)):
            prob_n_given_J[i] = prob_n_given_J[i-1]*(self.birth_rate*(i-1)
                               + self.immi_rate ) / ( i*( self.death_rate +
                               (self.birth_rate-self.death_rate)*( i*(1
                               - self.comp_overlap) + self.comp_overlap * J )
                               / self.carry_capacity )  )

        prob_n_given_J = prob_n_given_J/np.sum(prob_n_given_J)

        return prob_n_given_J

    def abund_J(self, technique='Nava'):
        """
        Various approximation for the probability distribution of the total
        population in the system (which we refer to as J)

        Return
            dstbn_tot_pop (array) : Distribution of total population
            mean_tot_pop ( double ): Mean of the distribution
        """
        prob_J = []

        if technique == 'Nava':
            prob_J = np.zeros( np.shape(self.population) );

            for J in self.population:
                prob_n_J = self.dstbn_n_given_J(J);
                convolution = np.copy(prob_n_J)
                for i in np.arange(1,self.nbr_species):
                    convolution = np.convolve(convolution,prob_n_J)
                #prob_J[J] = np.sum(convolution)
                prob_J[J] = convolution[J]
            prob_J = prob_J/np.sum(prob_J)

        elif technique == 'HL':
            def fcn_prob_J(av_n_squared):
                """
                Approximate P(J)
                """
                prob_J = np.zeros( np.shape(self.population) );

                for J in np.arange( 1, len(prob_J)):
                    prob_J_unnormalized[J] = prob_J_unnormalized[previous_J] * (
                        ( self.nbr_species*self.immi_rate + self.birth_rate *
                        previous_J ) / ( ( J*self.death_rate +
                        ( ( self.birth_rate-self.death_rate )*(
                        self.comp_overlap * J**2  + ( 1 - self.comp_overlap )
                        * self.nbr_species * av_n_squared )
                        / self.carry_capacity ) ) ) )
                        #print(prob_J_unnormalized[J])

                P0 = 1.0 / ( np.sum( prob_J_unnormalized ) )

        elif technique == 'Jeremy':
            def fcn_prob_J(av_n_squared):
                """
                If I had av_n_squared, this is how you could calculate prob_J,
                under the assumption Sum n_i^2 \approx nbr_species * <n^2>
                """
                prob_J_unnormalized = np.zeros( 10*len(self.population) );
                prob_J_unnormalized[0] = 1.0;

                for J in np.arange( 1, len(prob_J_unnormalized)):
                    previous_J = J - 1
                    prob_J_unnormalized[J] = prob_J_unnormalized[previous_J] * (
                        ( self.nbr_species*self.immi_rate + self.birth_rate *
                        previous_J ) / ( ( J*self.death_rate +
                        ( ( self.birth_rate-self.death_rate )*(
                        self.comp_overlap * J**2  + ( 1 - self.comp_overlap )
                        * self.nbr_species * av_n_squared )
                        / self.carry_capacity ) ) ) )
                        #print(prob_J_unnormalized[J])

                prob_J = prob_J_unnormalized / ( np.sum( prob_J_unnormalized ) )

                return prob_J[:len(self.population)]

            def equations(vars):
                av_n_squared = vars

                dstbn_J = fcn_prob_J(av_n_squared);

                # How do I want to solve this?
                """
                eqn = -av_n_squared
                for J, prob_J in enumerate(dstbn_J):
                    eqn += np.sum( (self.population**2*self.dstbn_n_given_J(J)
                                    * prob_J ) )
                """

                eqn = - av_n_squared

                prob_n = np.zeros(np.shape(self.population))
                for J, prob_J in enumerate(dstbn_J):
                    prob_n += self.dstbn_n_given_J(J) * prob_J

                # TODO : Does this need some weird normalizing???
                #prob_n = av_n2/np.sum(av_n2)

                eqn += np.dot( prob_n, self.population**2 )

                return eqn

            # TODO Maybe something better?
            initial_guess = (self.carry_capacity/(1 + self.comp_overlap*(
                            self.nbr_species - 1 ) ))**2

            # numerically solve for <n^2>
            av_n_squared_approx = sp_solver(equations, initial_guess)

            print('var',av_n_squared_approx)

            # Probability distribution of J
            prob_J = fcn_prob_J(av_n_squared_approx)

        # mean of J
        mean_tot_pop = np.dot(self.population, prob_J)

        return mean_tot_pop, prob_J


    def abund_moranton(self, const_J = True, dstbn_approx = 'Jeremy'):
        """
        Solving the Moran model for mutation is similar to Moran model for
        immigration. Doing this, we can obtain some

        Input
            const_J (binary) : Approximating J to be fixed or not
            dstbn_tot (string) : Which approximation of distribution of J to use
        """

        mean_J, dstbn_J = self.abund_J(dstbn_approx)

        # parametrization
        a = 2*self.immi_rate/self.birth_rate;
        Norm = gamma(self.nbr_species*a + 1 + a)/(gamma(self.nbr_species*a)*
                                                  gamma(a)*self.nbr_species)

        if const_J :
            abundance = (self.population/mean_J)**(-1+a)*np.exp(
                                    -a*self.population)/Norm
        else:
            # TODO : I don't think this is right. Not simply */ dstbn_J. SUM!
            abundance = np.zeros( np.shape(self.population) )
            for J, prob_J in enumerate(dstbn_J):
                abundance +=  ( (self.population/J)**(-1+a)*np.exp(
                                        -a*self.population)/Norm )*prob_J

        return abundance

    def abund_sid(self):
        """
        Mean field approximation :  Solving Master equation by assuming
                                    <J|n_1>=(S-1)<n>+n_1
        """
        def fcn_prob_n(mean_n):
            prob_n_unnormalized = np.zeros( np.shape(self.population) );
            prob_n_unnormalized[0] = 1.0#1E-250;
            for n in np.arange( 1, len(prob_n_unnormalized)):
                previous_n = n - 1
                prob_n_unnormalized[n] = prob_n_unnormalized[previous_n] * (
                    ( self.immi_rate + self.birth_rate*previous_n)
                    / ( n* (self.death_rate + (1-self.comp_overlap)*n*(
                    self.birth_rate-self.death_rate)/self.carry_capacity
                    + ( (self.nbr_species-1)*mean_n + n )*self.comp_overlap
                    / self.carry_capacity ) ) )

            prob_n = prob_n_unnormalized / ( np.sum( prob_n_unnormalized ) )

            return prob_n

        def equations(vars):
            mean_n = vars

            dstbn_n = fcn_prob_n(mean_n); eqn = 1.0

            eqn = np.dot(dstbn_n,self.population)-mean_n

            return eqn

        # TODO Maybe something better?
        initial_guess = self.carry_capacity / ( 1.0 +
                        self.comp_overlap*(self.nbr_species-1) )

        # numerically solve for <n>
        mean_n_approx = sp_solver(equations, initial_guess)

        # Probability distribution of n
        dstbn_n = fcn_prob_n(mean_n_approx)

        # Abundance of n
        abundance = dstbn_n * self.nbr_species

        return dstbn_n, abundance

    def abund_kolmog(self, const_J = True, dstbn_approx = 'Jeremy'):
        """
        Solving the difference equation for the abundance distribution means
        that at some point, we need to approximate the tot_pop. There are many
        ways to do this, however here I've decided to simply fix it to something
        to simplify. Better ways to do this I'm sure.
        """
        mean_J, dstbn_J = self.abund_J(dstbn_approx)

        def c_k_partial( i, J ):
            value = 1.0
            if i == 0.0:
                return value
            else:
                c = ( self.death_rate*self.carry_capacity/((self.birth_rate -
                    self.death_rate)) + (self.comp_overlap*J)
                    / ( 1-self.comp_overlap ) )
                value = ( self.birth_rate*self.carry_capacity / (
                        (self.birth_rate - self.death_rate) * (1 -
                        self.comp_overlap) ) ) ** i *( poch( self.immi_rate /
                        self.birth_rate,i ) / (factorial(i)*poch( c + 1, i ) ) )
                return value
        if const_J:
            # TODO : Should the sum range from 0 to J+1?
            Norm = mean_J/(np.sum( [ i*c_k_partial(i, mean_J) for i in
                                    self.population]))

            return np.array( [ c_k_partial(i, mean_J)*Norm
                                                 for i in self.population] )

        else :
            abundance = np.zeros( np.shape(self.population) )

            for J, prob_J in enumerate( dstbn_J ):
                Norm = J/(np.sum( [ i*c_k_partial(i, J) for i in
                                        self.population]))
                abundance += np.array( [ c_k_partial(i, J)*Norm for i in
                                  self.population] ) * prob_J

            return abundance

    def abund_jer( self ):
        """
        Approximation of abundance distribution of stochastic Lotka-Volterra
        with immigration. Instead of <n_i|n_j> use <n_i> where <n_i> approx. the
        deterministic solution.
        """

        probability = np.zeros( np.shape(self.population) )

        probability[0] = 1.0
        for i in np.arange(1,len(probability)):
            mean_n_given_i = self.deterministic_mean()
            mean_n_given_i = np.max([ self.carry_capacity-i, 0] ) / ( self.nbr_species-1 )
            probability[i] = probability[i-1] * ( ( self.immi_rate
                + self.birth_rate*(i-1) ) / ( i * ( self.death_rate
                + (self.birth_rate-self.death_rate) * ( i +
                ( ( self.nbr_species -1 ) * mean_n_given_i ) * self.comp_overlap
                )  / self.carry_capacity ) ) )


        probability = probability/np.sum(probability)

        return probability


    def abund_1spec_MSLV(self, dstbn_approx = 'Nava'):
        """
        Abundance distribution of stochastic Lotka-Volterra with immigration.
        Exact solution for comp_overlap = 0 or 1. Approximations involving the
        distribution of the total number of individuals for other comp_overlap
        """

        # different models, rho = 0
        if self.comp_overlap == 0.0:

            probability = np.zeros( np.shape(self.population) )
            #probability[0] = (1.0/hyp1f1(a,b,c))
            probability[0] = 1.0

            for i in np.arange(1,len(probability)):
                probability[i] = probability[i-1]*( self.birth_rate*(i-1)
                                   + self.immi_rate ) / (i*( self.death_rate +
                                   (self.birth_rate-self.death_rate)*( i )
                                   / self.carry_capacity )  )

            probability = probability/np.sum(probability)

        # rho = 1
        elif self.comp_overlap == 1.0:
            """
            probability = np.zeros( np.shape(self.population) )
            #probability[0] = (1.0/hyp1f1(a,b,c))
            probability[0] = 1.0

            for i in np.arange(1,len(probability)):
                probability[i] = probability[i-1]*(self.birth_rate*(i-1)
                                   + self.immi_rate ) / (i*( self.death_rate +
                                   (self.birth_rate-self.death_rate)*( i )
                                   / self.carry_capacity )  )

            probability = probability/np.sum(probability)
            """
            abundance = self.population

        # 0 < rho < 1. TODO : rho > 1 ???
        else:
            """
            This is an approximation! Mean field... sort of?
            """
            mean_J, dstbn_J = self.abund_J(technique=dstbn_approx)

            probability = np.zeros( np.shape(self.population) )

            for J, prob_J in enumerate( dstbn_J ):
                probability += self.dstbn_n_given_J(J) * prob_J

            # Not normalized???
            #probability = probability/(np.sum(probability))

        abundance = self.nbr_species * probability

        return probability, abundance

    def mean_time_extinction(self, n_start, n_end, dstbn=[]):
        """
        Using theory of mean first passage times of 1 specie, here is the equation for the
        mean time it takes to go from n_start to n_end
        """
        if n_end > n_start:
            print('Warning : ' + str(n_end) + " larger than "
                            + str(n_start)  + ", for now can only check opposite!")
            raise SystemExit
        mte = 0.0
        def birth_rate_eqn(n):
            return self.birth_rate*n + self.immi_rate

        if dstbn == []:
            probability = self.abund_1spec_MSLV(self, dstbn_approx = 'Nava')
        else:
            probability = dstbn

        for i in range(n_end, n_start):
            mte += np.sum(probability[i+1:])/(birth_rate_eqn(i)*probability[i])

        return mte

    def deterministic_mean(self):
        """
        Calculates the mean of the LV equations, for com_overlap between 0
        and 1.
        """
        return self.carry_capacity*( ( 1. + np.sqrt( 1.+ 4.*self.immi_rate*
               ( 1. + self.comp_overlap*( self.nbr_species - 1. ) ) /
               (self.carry_capacity*(self.birth_rate-self.death_rate) ) ) )
               / ( 2.*( 1.+self.comp_overlap*( self.nbr_species-1.) ) ) )


    #NONE OF THESE NEED TO BE IN THE MODEL CLASS
    def entropy(self, probability):
        # calculate entropy of a distribution
        return - np.dot(probability[probability>0.0],
                         np.log(probability[probability>0.0]))

    def ginisimpson_idx(self, probability):
        # calculate gini-simpson index of a distribution
        return 1.0 - np.dot(probability,probability)

    def richness(self, probability ):
        # Probability of not being absent in the simulation
        return 1.0 - probability[0]

    def KL_divergence(self, P, Q):
        # 1D kullback-leibler divergence between 2 distributions P and Q
        dimP = len(P); dimQ = len(Q) # TODO change to np.shape

        if dimP != dimQ:
            print('Dimension of 2 distributions not equal!')
            P = P[:]
            P = P[:np.min([dimQ,dimP])]; Q = Q[:np.min([dimQ,dimP])];

        KLpq = np.sum(P[P>0.0]*np.log(P[P>0.0])) \
                            - np.sum(P[Q>0.0]*np.log(Q[Q>0.0]))

        return KLpq

    def JS_divergence(self, P, Q):
        # 1D Jensen-Shannon between 2 distributions P and Q
        dimP = len(P); dimQ = len(Q) # TODO change to np.shape
        if dimP != dimQ:
            print('Dimension of 2 distributions not equal!')
            P = P[:np.min([dimQ,dimP])]; Q = Q[:np.min([dimQ,dimP])];

        #print((self.KL_divergence(P,Q) + self.KL_divergence(Q,P))/2.0)

        return (self.KL_divergence(P,Q) + self.KL_divergence(Q,P))/2.0

    def binomial_diversity_dstbn( self, P0 , nbr_species=None):
        """
        Using a binomial distribution, find the diversity distribution of
        species present R, simply
        from
                        P(R;S,1-P0) = P0^(S-R)*(1-P0)^R
        """
        if nbr_species == None:
            species = np.arange( self.nbr_species + 1)
        else:
            species = np.arange( nbr_species + 1 )
        return scipy.stats.binom.pmf(species, self.nbr_species, 1 - P0)



class CompareModels(object):
    """
    Trying to compare models in certain ways.
    """
    def __init__(self, comp_overlap=np.logspace(-2,-0.01,40),
                 birth_rate=np.logspace(-2,2,40),
                 death_rate=np.logspace(-2,2,40),
                 carry_capacity=(np.logspace(1,3,40)).astype(int),
                 immi_rate=np.logspace(-4,0,40),
                 nbr_species=(np.logspace(0,3,40)).astype(int),
                 model=Model_MultiLVim):
        self.comp_overlap = comp_overlap; self.birth_rate = birth_rate;
        self.death_rate = death_rate; self.immi_rate = immi_rate;
        self.carry_capacity = carry_capacity; self.nbr_species = nbr_species;

        self.model = model()

    def mlv_metric_compare(self, key1):
        """
        Compares along 3 metrics: entropy, richness and gini-simpson index

        Input
            key1 : The variable to vary

        Output
            H : shannon entropy as a function of var (var_array)
            richness : 1 - P(0), probability of being present
            species_richness : average number of species in the system
            GS : Gini-Simpson index
        """

        if load == True:
            H, richness, GS = np.load()

        H = np.zeros( np.shape(getattr(self,key1)) )
        richness = np.zeros( np.shape(getattr(self,key1)) )
        GS = np.zeros( np.shape(getattr(self,key1)) )
        JS = np.zeros( np.shape(getattr(self,key1)) )

        for i, value in enumerate(getattr(self,key1)):
            #setattr(self.model,key1,value)
            #probability, _ = self.model.abund_1spec_MSLV()
            #approximation = self.model.abund_1spec_MSLV
            probability_sid, _  = self.model.abund_sid()
            probability_nava, _ = self.model.abund_1spec_MSLV()
            H[i]           = self.model.entropy(probability_nava)
            GS[i]          = self.model.ginisimpson_idx(probability_nava)
            richness[i]    = self.model.richness(probability_nava)
            JS[i]          = self.model.JS_divergence(probability_nava
                                                        , probability_sid)

            print('>'+str(i))

        ## Entropy 1D
        fig = plt.figure()
        plt.plot(getattr(self,key1),H)
        plt.ylabel("Entropy")
        plt.xlabel(key)
        #plt.yscale('log')
        plt.xscale('log')
        plt.show()

        ## Gini-Simpson index 1D
        fig = plt.figure()
        plt.plot(getattr(self,key1),GS)
        plt.ylabel("Gini-Simspon Index")
        plt.xlabel(key)
        #plt.yscale('log')
        plt.xscale('log')
        plt.show()

        ## Richness 1D
        fig = plt.figure()
        plt.plot(getattr(self,key1),richness)
        plt.ylabel(r"$1-P(0)$")
        plt.xlabel(key)
        #plt.yscale('log')
        plt.xscale('log')
        plt.show()

        ## Species richness 1D
        species_richness = None
        if key1 == 'nbr_species':
            species_richness = richness * getattr(self,key1)
            fig = plt.figure()
            plt.plot(getattr(self,key1),species_richness)
            plt.ylabel(r"$average species richness$")
            plt.xlabel(key)
            #plt.yscale('log')
            plt.xscale('log')
            plt.show()

        return H, GS, richness

    def mlv_metric_compare_heatmap(self, key1, key2, file='metrics3.npz', plot=False
                                    , load_npz=False):
        """
        Compares along 3 metrics: entropy, richness and gini-simpson index

        Input
            key1        : The variable to vary
            key2        : 2nd variable
            files       : Name of file we want to save
            plot        : Whether or not to plot
            load_npz    : Whether or not to load from an npz file

        Output
            H                   : shannon entropy as a function of var
            richness            : 1 - P(0), probability of being present
            species_richness    : average number of species in the system
            GS                  : Gini-Simpson index
        """
        filename = THRY_FIG_DIR + os.sep + file

        # approximation to use
        #approximation = self.model.abund_1spec_MSLV
        #approximation = self.model.abund_sid

        #load
        if load_npz:
            # check it exists
            if not os.path.exists(filename):
                print('No file to load!')
                raise SystemExit
            with np.load(filename) as f:
                H = f['H']; GS = f['GS']; richness = f['richness'];
                JS = f['JS']; xrange = ['xrange']; yrange = ['yrange'];
                approx_dist_sid = f['approx_dist_sid']
                approx_dist_nava = f['approx_dist_nava']

        # Create the npz file
        else:
            # initialize
            H        = np.zeros( ( np.shape(getattr(self,key1))[0],
                                np.shape(getattr(self,key2))[0] ) )
            richness = np.zeros( ( np.shape(getattr(self,key1))[0],
                                np.shape(getattr(self,key2))[0] ) )
            GS       = np.zeros( ( np.shape(getattr(self,key1))[0],
                                np.shape(getattr(self,key2))[0] ) )
            JS       = np.zeros( ( np.shape(getattr(self,key1))[0],
                                np.shape(getattr(self,key2))[0] ) )
            approx_dist_sid = np.zeros( ( np.shape(getattr(self,key1))[0],
                                np.shape(getattr(self,key2))[0],
                                np.shape(getattr(self.model,'population'))[0]))
            approx_dist_nava = np.zeros( ( np.shape(getattr(self,key1))[0],
                                np.shape(getattr(self,key2))[0],
                                np.shape(getattr(self.model,'population'))[0]))
            det_mean = np.zeros( ( np.shape(getattr(self,key1))[0],
                                np.shape(getattr(self,key2))[0] ) )
            mean_time_dominance = np.zeros( ( np.shape(getattr(self,key1))[0],
                                np.shape(getattr(self,key2))[0] ) )
            det_mean_spec_present = np.zeros( ( np.shape(getattr(self,key1))[0],
                                np.shape(getattr(self,key2))[0] ) )
            xrange   = getattr(self,key1)
            yrange   = getattr(self,key2)


            # create heatmap array for metrics
            for i, valuei in enumerate(getattr(self,key1)):
                for j, valuej in enumerate(getattr(self,key2)):
                    setattr(self.model,key1,valuei)
                    setattr(self.model,key2,valuej)
                    #t = time.time()
                    probability_sid, _  = self.model.abund_sid()
                    #print(time.time() - t)

                    #t = time.time()
                    probability_nava, _ = self.model.abund_1spec_MSLV()
                    #print(time.time() - t)
                    approx_dist_sid[i,j] = probability_sid
                    approx_dist_nava[i,j] = probability_nava

                    probability = probability_nava

                    H[i,j]        = self.model.entropy(probability)
                    GS[i,j]       = self.model.ginisimpson_idx(probability)
                    richness[i,j] = self.model.richness(probability)
                    #JS[i,j]         = self.model.JS_divergence(probability_nava
                    #                                       , probability)

                    mean_time_dominance[i,j] = self.model.mean_time_extinction(
                                    int(self.model.nbr_species*(1.-probability[0])), 0, probability)
                    det_mean[i,j] = self.model.deterministic_mean()

                    print('>'+str(j))
                print('>>>'+str(i))

            # save
            metric_dict = {'H' : H, 'GS' : GS, 'richness' : richness, 'JS' : JS
                            , 'approx_dist_sid' : approx_dist_sid
                            , 'approx_dist_nava' : approx_dist_nava
                            , 'det_mean' : det_mean,
                            'xrange' : xrange, 'yrange' : yrange}
            np.savez(filename, **metric_dict)

        if plot:

            # some settings for the heatmaps
            imshow_kw = {'cmap': 'YlGnBu', 'aspect': None
                         #,'vmin': vmin, 'vmax': vmax
                         #,'norm': mpl.colors.LogNorm(vmin,vmax)
                    }
            approx_dist = approx_dist_nava
            # setting of xticks
            POINTS_BETWEEN_X_TICKS = 9
            POINTS_BETWEEN_Y_TICKS = 9

            ## Entropy 2D
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(H, interpolation='none', **imshow_kw)

            # labels and ticks
            ax.set_xticks([i for i, cval in enumerate(xrange)
                                if i % POINTS_BETWEEN_X_TICKS == 0])
            ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                                for i, xval in enumerate(xrange)
                                if (i % POINTS_BETWEEN_X_TICKS==0)])
            ax.set_yticks([i for i, kval in enumerate(yrange)
                                if i % POINTS_BETWEEN_Y_TICKS == 0])
            ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                                for i, yval in enumerate(yrange)
                                if i % POINTS_BETWEEN_Y_TICKS==0])
            plt.xlabel(key1); plt.ylabel(key2)
            ax.invert_yaxis()
            #plt.xscale('log'); plt.yscale('log')
            plt.colorbar(im,ax=ax)
            plt.title('shannon entropy')
            plt.show()

            ## Gini-Simpson 2D
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(GS, interpolation='none', **imshow_kw)

            # labels and ticks
            ax.set_xticks([i for i, cval in enumerate(xrange)
                                if i % POINTS_BETWEEN_X_TICKS == 0])
            ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                                for i, xval in enumerate(xrange)
                                if (i % POINTS_BETWEEN_X_TICKS==0)])
            ax.set_yticks([i for i, kval in enumerate(yrange)
                                if i % POINTS_BETWEEN_Y_TICKS == 0])
            ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                                for i, yval in enumerate(yrange)
                                if (i % POINTS_BETWEEN_Y_TICKS==0)])
            plt.xlabel(key1); plt.ylabel(key2)
            ax.invert_yaxis()
            plt.title(r'Gini-Simpson index')
            #plt.xscale('log'); plt.yscale('log')

            # colorbar
            plt.colorbar(im,ax=ax)
            plt.show()

            ## Richness 2D
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(richness, interpolation='none', **imshow_kw)

            # labels and ticks
            ax.set_xticks([i for i, cval in enumerate(xrange)
                                if i % POINTS_BETWEEN_X_TICKS == 0])
            ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                                for i, xval in enumerate(xrange)
                                if (i % POINTS_BETWEEN_X_TICKS==0)])
            ax.set_yticks([i for i, kval in enumerate(yrange)
                                if i % POINTS_BETWEEN_Y_TICKS == 0])
            ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                                for i, yval in enumerate(yrange)
                                if i % POINTS_BETWEEN_Y_TICKS==0])
            plt.xlabel(key1); plt.ylabel(key2)
            ax.invert_yaxis()
            #plt.xscale('log'); plt.yscale('log')
            plt.colorbar(im,ax=ax)
            plt.title(r'$1-P(0)$')
            plt.show()


            ## Jensen-shannon
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(JS, interpolation='none', **imshow_kw)

            # labels and ticks
            ax.set_xticks([i for i, cval in enumerate(xrange)
                                if i % POINTS_BETWEEN_X_TICKS == 0])
            ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                                for i, xval in enumerate(xrange)
                                if (i % POINTS_BETWEEN_X_TICKS==0)])
            ax.set_yticks([i for i, kval in enumerate(yrange)
                                if i % POINTS_BETWEEN_Y_TICKS == 0])
            ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                                for i, yval in enumerate(yrange)
                                if i % POINTS_BETWEEN_Y_TICKS==0])
            plt.xlabel(key1); plt.ylabel(key2)
            ax.invert_yaxis()
            #plt.xscale('log'); plt.yscale('log')
            plt.colorbar(im,ax=ax)
            plt.title(r'$Jensen-Shannon divergence$')
            plt.show()

            ## Mean time extinction
            plt.style.use('custom_heatmap.mplstyle')
            imshow_kw ={'cmap': 'viridis', 'aspect': None }
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(mean_time_dominance, interpolation='none'
                                , norm=LogNorm(vmin=np.min(mean_time_dominance)
                                            ,vmax=np.max(mean_time_dominance))
                                ,**imshow_kw)

            # labels and ticks
            ax.set_xticks([i for i, cval in enumerate(xrange)
                                if i % POINTS_BETWEEN_X_TICKS == 0])
            ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                                for i, xval in enumerate(xrange)
                                if (i % POINTS_BETWEEN_X_TICKS==0)])
            ax.set_yticks([i for i, kval in enumerate(yrange)
                                if i % POINTS_BETWEEN_Y_TICKS == 0])
            ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                                for i, yval in enumerate(yrange)
                                if i % POINTS_BETWEEN_Y_TICKS==0])
            plt.xlabel(VAR_NAME_DICT[key1]); plt.ylabel(VAR_NAME_DICT[key2])
            ax.invert_yaxis()
            #plt.xscale('log'); plt.yscale('log')
            plt.colorbar(im,ax=ax)
            plt.title(r'Mean time to lose dominance')
            plt.show()



            ## Species richness 2D
            # if species are changing, additionally give species richness, not
            # just 1-P(0)
            if key1 == 'nbr_species' or key2 == 'nbr_species':
                species_richness = np.zeros( ( np.shape(getattr(self,key1))[0],
                              np.shape(getattr(self,key2))[0] ) )

                if key1 == 'nbr_species':
                    species_richness = richness * getattr(self,key1)
                else:
                    species_richness = ( richness.T * getattr(self,key1) ).T

                # initialize
                f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
                im = ax.imshow(species_richness, interpolation='none'
                                               , **imshow_kw)

                # labels and ticks
                ax.set_xticks([i for i, cval in enumerate(xrange)
                                    if i % POINTS_BETWEEN_X_TICKS == 0])
                ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval)
                                    for i, xval in enumerate(xrange)
                                    if (i % POINTS_BETWEEN_X_TICKS==0)]
                                    , fontsize=FS)
                ax.set_yticks([i for i, kval in enumerate(yrange)
                                    if i % POINTS_BETWEEN_Y_TICKS == 0])
                ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval)
                                    for i, yval in enumerate(yrange)
                                    if i % POINTS_BETWEEN_Y_TICKS==0]
                                    , fontsize=FS)
                plt.xlabel(key1); plt.ylabel(key2)
                ax.invert_yaxis()
                plt.title(r'species richness')
                #plt.xscale('log'); plt.yscale('log')

                # colorbar
                plt.colorbar(im,ax=ax)
                plt.show()

        return 0

    def mlv_compare_abundance_approx(self):
        """
        Compare different approximations for abundance Distribution

        """
        matlab_files = ['Results_01','Results_05','Results_1']

        # plot a couple different abundance distributions
        for i, comp_o in enumerate([0.1,0.5,0.999]):

            # similar models except for comp_o
            EqnsMLV = self.model(comp_overlap=comp_o)

            # list of approximations to check
            prob_sid, abund_sid = EqnsMLV.abund_sid()
            prob_nava, abund_nava = EqnsMLV.abund_1spec_MSLV(
                                                    dstbn_approx = 'Nava')
            #abund_moranton_const = EqnsMLV.abund_moranton()
            #abund_moranton = EqnsMLV.abund_moranton(const_J=False)
            #abund_kolmog = EqnsMLV.abund_kolmog(const_J=False)
            #prob_jeremy, abund_jeremy = EqnsMLV.abund_1spec_MSLV(
            #                                        dstbn_approx='Jeremy')

            # Nava sent some mathematica files
            dict_matlab = spo.loadmat('nava_results'+os.sep+matlab_files[i])
            abund_nava_m = dict_matlab['QN']
            abund_sid_m = dict_matlab['Q_Ant']

            # plot figure
            fig = plt.figure(); end = 300 # cut somehere
            plt.plot(EqnsMLV.population[:end], abund_sid[:end],
                        label='A.-S. approx.')
            plt.plot(EqnsMLV.population[:end], abund_nava[:end],
                        label='Nava approx.')
            plt.plot(EqnsMLV.population[:end],
                        EqnsMLV.nbr_species*abund_sid_m[:end],
                        label='A.-S. Matlab', linestyle='-.')
            plt.plot(EqnsMLV.population[:end],
                        EqnsMLV.nbr_species*abund_nava_m[:end],
                        label='Nava Matlab', linestyle='-.')
            #plt.plot(EqnsMLV.population,abund_moranton_const,
                      #label='Moranton cst.')
            #plt.plot(EqnsMLV.population,abund_moranton,label='Moranton')
            #plt.plot(EqnsMLV.population,abund_kolmog,label='Kolmog. approx')
            #plt.plot(EqnsMLV.population[:end], abund_jeremy[:end],
                      #label='Jeremy approx.')
            plt.legend(loc='best')
            plt.ylabel("abundance")
            plt.xlabel("population size")
            plt.yscale('log')
            plt.show()

            return 0

def vary_species_count(species=150):
    """
    We want to vary the total number of species at high mu, low rho to see how
    the solution's peak differs from the carrying capacity
    """

    nbr_species = np.arange( 1, species )
    peak_diff = np.zeros( species - 1  )

    for i, nbr_spec in enumerate( nbr_species ):
        params = {'carry_capacity' : 50 , 'immi_rate' : 0.001
                                        , 'comp_overlap' : 0.9
                                        , 'nbr_species' : nbr_spec }
        Model = Model_MultiLVim( **params )
        dstbn, _ = Model.abund_sid()

        offset = 10 # in case peak is zero
        print( i, np.argmax(dstbn[offset:]) )
        peak_diff[i] = params['carry_capacity'] \
                                - np.argmax(dstbn[offset:]) - offset

    fig = plt.figure(); end = 300 # cut somehere
    plt.plot( nbr_species, peak_diff )
    plt.xlabel(r"number of species, $S$")
    plt.ylabel(r"$K$ - peak mean field")
    plt.title(r"$\mu=${}, $\rho=${} ".format( params['immi_rate']
                                                , params['comp_overlap']))
    plt.show()

    return 0



if __name__ == "__main__":

    #vary_species_count(200)

    problematic_params = {'birth_rate' : 20.0, 'death_rate'     : 1.0
                                            , 'immi_rate'       : 0.001
                                            , 'carry_capacity'  : 100
                                            , 'comp_overlap'    : 0.8689
                                            , 'nbr_species'     : 30
                                            }
    model = Model_MultiLVim(**problematic_params)

    distribution = model.abund_jer()
    print((model.nbr_species-1)*model.deterministic_mean())

    fig = plt.figure(); end = int(1.5*model.carry_capacity) # cut somehere
    plt.plot( np.arange(end), distribution[:end])
    plt.xlabel(r"population, $n_i$")
    plt.ylabel(r"probability $P(n_i)$")
    plt.yscale('log')
    #plt.title(r"$\mu=${}, $\rho=${} ".format( params['immi_rate']
    #                                            , params['comp_overlap']))
    plt.show()

    #compare = CompareModels()
    #compare.mlv_compare_abundance_approx()
    #compare.mlv_metric_compare_heatmap("comp_overlap","immi_rate", plot=False, load_npz=False)
