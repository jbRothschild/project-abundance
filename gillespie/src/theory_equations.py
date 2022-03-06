import numpy as np; import csv, os, glob, csv, time, datetime

import matplotlib as mpl; import matplotlib.pyplot as plt
import matplotlib.patches as patches # histgram animation
import matplotlib.path as path # histogram animation
from matplotlib import animation # animation
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import itertools; from itertools import combinations

from scipy.special import gamma, poch, factorial, hyp1f1, comb, beta # certain functions
import scipy, scipy.stats; import scipy.io as spo
from scipy.optimize import fsolve as sp_solver # numerical solver, can change to
                                               # a couple others, i.e. broyden1
#import seaborn as sns
from src.settings import VAR_NAME_DICT

#plt.style.use('custom.mplstyle')

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
        self.nbr_species = nbr_species; self.carry_capacity = carry_capacity;
        self.population = np.arange( int( 1.5 * self.carry_capacity +
                                    ( self.nbr_species - 1 )
                                    * 1.5 * self.carry_capacity
                                    * ( 1. - self.comp_overlap) ) );
        self.dstbn_J = None; self.dstbn_n = None

        if 'conditional' in kwargs:
            self.conditional = kwargs['conditional']

    def __setattr__(self, name, value):
        """
        Updates population size to
        """
        self.__dict__[name] = value;
        if name == "carry_capacity":
            self.population = np.arange( int( 1.5 * self.carry_capacity +
                                        ( self.nbr_species - 1 )
                                        * 1.5 * self.carry_capacity
                                        * ( 1. - self.comp_overlap) ) );

    def deterministic_mean(self, nbr_species_present=0):
        """
        Calculates the mean of the LV equations, for com_overlap between 0
        and 1.
        """
        if nbr_species_present==0:
            nbr_species_present = self.nbr_species
        return self.carry_capacity*( ( 1. + np.sqrt( 1.+ 4.*self.immi_rate*
               ( 1. + self.comp_overlap*( nbr_species_present - 1. ) ) /
               (self.carry_capacity*(self.birth_rate-self.death_rate) ) ) )
               / ( 2.*( 1.+self.comp_overlap*( nbr_species_present-1.) ) ) )

    def dstbn_n_given_J(self, J):
        """
        Marginal Distribution of n given J.

        Return
            Marginal probability distribution P(n|J)
        """
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
        """
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
            """
            Convolutions of distributions
            """
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
                prob_J_unnormalized = np.zeros( len(self.population) );
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

            #print('var',av_n_squared_approx)

            # Probability distribution of J
            prob_J = fcn_prob_J(av_n_squared_approx)

        # mean of J
        mean_tot_pop = np.dot(self.population, prob_J)

        self.dstbn_J = prob_J

        return mean_tot_pop, prob_J

    def abund_moranton(self, const_J = True, dstbn_approx = 'Jeremy'):
        """
        Solving the Moran model for mutation is similar to Moran model for
        immigration. Doing this, we can obtain some

        Input
            const_J (binary)    : Approximating J to be fixed or not
            dstbn_tot (string)  : Which approximation of distribution of J to
                                    use, Jeremy, HL or Nava

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
                    + ( self.birth_rate-self.death_rate)*(
                    (self.nbr_species-1)*mean_n + n )*self.comp_overlap
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

        self.dstbn_n = dstbn_n

        return dstbn_n, abundance

    def abund_jer(self):
        """
        <n_j|n_i> is given by the deterministic lotka voltera equations, however
        n_i is fixed for one species. Solve the equation for other species to
        get:
                n^{+,-} = (K-rho n_i)(1{+,-})
        """
        sqr = np.sqrt( 1. + (4.0*self.immi_rate
                * (1.0 + self.comp_overlap*( self.nbr_species-2 ))
                / ( self.carry_capacity*( self.birth_rate-self.death_rate ) ) ))
        N = self.carry_capacity / self.comp_overlap
        n_j = ( (self.carry_capacity - self.comp_overlap
                    * np.arange( len(self.population) ) ) / ( 2. * ( 1.
                    + self.comp_overlap * ( self.nbr_species - 2. ) ) ) );
        n_j[:int(np.ceil(N))] *= (1. + sqr )
        n_j[int(np.floor(N)):] *= (1. - sqr )
        prob_n_unnormalized = np.zeros( np.shape(self.population) );
        prob_n_unnormalized[0] = 1.0#1E-250;
        for n in np.arange( 1, len(prob_n_unnormalized)):
            previous_n = n - 1
            prob_n_unnormalized[n] = prob_n_unnormalized[previous_n] * (
                ( self.immi_rate + self.birth_rate*previous_n)
                / ( n* (self.death_rate + (1-self.comp_overlap)*n*(
                self.birth_rate-self.death_rate)/self.carry_capacity
                + ( self.birth_rate-self.death_rate)*(
                (self.nbr_species-1)*n_j[n] + n )*self.comp_overlap
                / self.carry_capacity ) ) )

        prob_n = prob_n_unnormalized / ( np.sum( prob_n_unnormalized ) )

        return prob_n


    def abund_sid_J(self):
        """
        Mean field approximation :  Solving Master equation by assuming
                                    <J|n_1>=S <n>
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
                    + ( self.birth_rate-self.death_rate)*(
                    (self.nbr_species)*mean_n )*self.comp_overlap
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

        self.dstbn_n = dstbn_n

        return dstbn_n, abundance

    def abund_kolmog(self, const_J = True, dstbn_approx = 'Jeremy'):
        # TODO REMOVE
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

    def prob_Jgivenn_det(self, n):
        # TODO REMOVE
        """
        Approximate P(J-n|n): P(Jtilde|n)~e^(-abs(dJtilde/dt)*1/mu)
        It's an aweful approximation.

        Input
            n : the species in P(J|n)
        output
            P(J-n|n) : array
        """
        dJtildedt   = np.zeros( np.shape(self.population) );
        dndt        = np.zeros( np.shape(self.population) );
        mean_n = self.deterministic_mean();
        for j, J in enumerate(self.population):

            #here we have use the approximation that sum<n_i^2> ~ Jtilde<n_i>
            dJtildedt[j] = ( self.birth_rate - self.death_rate ) * J * ( 1.0  -
                            ( self.comp_overlap * ( J + n ) + ( 1.0
                            - self.comp_overlap) * mean_n)
                            / self.carry_capacity )\
                            + (self.nbr_species - 1.0) * self.immi_rate

            """
            #here we have use the approximation that sum<n_i^2> ~ Jtilde<n_i>
            dJtildedt[j] = ( self.birth_rate - self.death_rate ) * ( J * ( 1.0
                            - self.comp_overlap * ( J + n ) /
                            self.carry_capacity ) - ( 1.0 - self.comp_overlap)
                            * mean_n**2 * (self.nbr_species - 1.0 )
                            / self.carry_capacity ) \
                            + (self.nbr_species - 1.0) * self.immi_rate
            """

            dndt[j] = ( self.birth_rate - self.death_rate ) * n * ( 1.0
                        - ( n + self.comp_overlap * J ) / self.carry_capacity )\
                        + self.immi_rate

        #prob_J_given_n = np.exp( - np.sqrt( np.square(dJtildedt)
        #                     + np.square(dndt) ) / (self.carry_capacity) )
        prob_J_given_n = 1./np.sqrt(np.square(dJtildedt) + np.square(dndt))

        if np.sum(prob_J_given_n) == 0.0:
            print("Problem, P(J|n) = 0.0 for all J!")
            return prob_J_given_n
        else:
            return prob_J_given_n/np.sum(prob_J_given_n)

    def prob_Jgivenn_rates(self, n):
        # TODO REMOVE
        """
        Approximate P(J-n|n): P(Jtilde|n) ~ 1 / (sum of rates))
        It's an aweful approximation.

        Input
            n : the species in P(J|n)
        output
            P(J-n|n) : array
        """
        ratesJ   = np.zeros( np.shape(self.population) );
        ratesn   = np.zeros( np.shape(self.population) );
        mean_n = self.deterministic_mean();
        for j, J in enumerate(self.population):
            """
            #here we have use the approximation that sum<n_i^2> ~ Jtilde<n_i>
            ratesJ[j] = J * ( ( self.birth_rate + self.death_rate )  +  (
                            self.birth_rate - self.death_rate ) * (
                            ( self.comp_overlap * ( J + n ) + ( 1.0
                            - self.comp_overlap) * mean_n )
                            / self.carry_capacity ) )\
                            + (self.nbr_species - 1.0) * self.immi_rate

            """
            #here we have use the approximation that sum<n_i^2> ~ (S-1)<n_i>^2
            ratesJ[j] = J * ( ( self.birth_rate + self.death_rate )
                            + ( self.birth_rate - self.death_rate ) * (
                            self.comp_overlap * ( J + n ) / self.carry_capacity
                             ) ) + ( 1.0 - self.comp_overlap) * mean_n**2 * (
                             self.nbr_species - 1.0 ) / self.carry_capacity \
                            + (self.nbr_species - 1.0) * self.immi_rate


            ratesn[j] = n * ( ( self.birth_rate + self.death_rate ) +
                        ( self.birth_rate + self.death_rate ) * (
                        n + self.comp_overlap * J ) / self.carry_capacity )\
                        + self.immi_rate

        #prob_J_given_n = np.exp( - np.sqrt( np.square(dJtildedt)
        #                     + np.square(dndt) ) / (self.carry_capacity) )
        prob_J_given_n = 1./( ratesJ + ratesn)

        if np.sum(prob_J_given_n) == 0.0:
            print("Something may be wrong")
            return prob_J_given_n
        else:
            return prob_J_given_n/np.sum(prob_J_given_n)


    def mean_Jtilde_given_n( self, n, dstbn_J, approx):
        # TODO REMOVE
        """
        <J-n|n>, according to different approximations

        Input :

        """

        if approx ==  'prob_Jgiveni_deterministic':
            prob_J_given_n = self.prob_Jgivenn_det(n)
            return np.dot( prob_J_given_n, self.population )

        elif approx == 'prob_Jgiveni_rates':
            prob_J_given_n = self.prob_Jgivenn_rates(n)
            return np.dot( prob_J_given_n, self.population )
        elif approx == 'simulation':
            if n >= np.shape(self.conditional)[0]:
                return 0.0

            else:
                if (np.sum(self.conditional[n][:]) != 1.0 and
                    np.sum(self.conditional[n][:]) != 0):
                    #print('Normalizing issues with P(n_i|n_j) : '
                    #                + str(np.sum(self.conditional[i][:])) )
                    self.conditional[n][:] /= np.sum(self.conditional[n][:])
                av_ni_given_n = np.dot(np.arange(np.shape(self.conditional)[1])
                        , np.array(self.conditional[n][:]) )
                return (self.nbr_species - 1) * av_ni_given_n
        else:
            print("Warning :  We don't have any other approximation for <J|n>")
            raise SystemExit

    def abund_jer_old( self, approx='prob_Jgiveni_deterministic' ):
        """
        Approximation of abundance distribution of stochastic Lotka-Volterra
        with immigration. Instead of exact <J|n> use some approximation

        Input
            approx      : Which approximation to use,
                            'prob_Jgiveni_deterministic', 'simulation'
                            , 'prob_Jgiveni_rates', 'Haeg_Lor'
        Output
            probability : array P(n)
        """

        probability = np.zeros( np.shape(self.population)[0] )
        _, dstbn_J     = self.abund_J(technique='Nava') # Q(J)
        prob_approx = np.zeros( np.shape(self.population)[0] ) # Q(n)
        full_dstbn_n_given_J = np.zeros( (np.shape(self.population)[0]
                                            , np.shape(self.population)[0] ) )


        for J, prob_J in enumerate( dstbn_J ):
            full_dstbn_n_given_J[J,:] = self.dstbn_n_given_J(J) # Q(n|J)
            prob_approx += full_dstbn_n_given_J[J,:] * prob_J # sum Q(n|J)*Q(J)

        prob_approx /= np.sum(prob_approx) # Q(N), should be normalized

        print(np.shape(prob_approx), np.shape(full_dstbn_n_given_J)
                    , np.shape(dstbn_J) )

        probability[0] = 1.0
        # TODO DELETE IF BELOW WORKS
        """
        for i in np.arange(1,len(probability)):
            probability[i] = probability[i-1] * ( ( self.immi_rate
                + self.birth_rate*(i-1) ) / ( i * ( self.death_rate
                + (self.birth_rate - self.death_rate) * ( i +
                 self.mean_Jtilde_given_n(i, probJ, approx) * self.comp_overlap
                )  / self.carry_capacity ) ) )
        """
        for i in np.arange(1,np.size(self.population)):
            # prob_J_given_i array
            prob_J_given_i = full_dstbn_n_given_J[:,i]*dstbn_J/prob_approx[i]
            probability[i] = probability[i-1] * ( ( self.immi_rate
                + self.birth_rate*(i-1) ) / ( i * ( self.death_rate
                + (self.birth_rate - self.death_rate) * ( i * (1.0
                - self.comp_overlap ) + np.dot(self.population
                ,prob_J_given_i) * self.comp_overlap ) / self.carry_capacity )))

        probability = probability / np.sum(probability)

        self.dstbn_n = probability

        return probability


    def abund_1spec_MSLV(self, dstbn_approx = 'Nava'):
        """
        Abundance distribution of stochastic Lotka-Volterra with immigration.
        Exact solution for comp_overlap = 0 or 1. Approximations involving the
        distribution of the total number of individuals for other comp_overlap

        Input
            dstbn_approx : which approximation to use for P(J)
                                'Nava', 'HL', 'Jeremy'
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

        self.dstbn_n = probability

        return probability, abundance

    def mean_time_extinction(self, n_start, n_end, dstbn=[]):
        """
        Using theory of mean first passage times of 1 specie, here is the
        equation for the mean time it takes to go from n_start to n_end
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

    def binomial_diversity_dstbn( self, P0, nbr_species=None):
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

        return scipy.stats.binom.pmf(species, self.nbr_species\
                                            , 1 - P0 )

    def mfpt_a2b(self, a, b):
        """
        From distribution, get the mfpt <T_{b}(a)>, a<b
        """
        mfpt = 0.0
        for i in np.arange(a,b):
            mfpt += ( np.sum(self.dstbn_n[:i+1] ) ) / ( ( self.birth_rate*i
                                + self.immi_rate )*self.dstbn_n[i] )
        return mfpt

    def mfpt_b2a(self, a, b):
        """
        From distribution, get the mfpt <T_{a}(b)>
        """
        mfpt = 0
        for i in np.arange(a,b):
            mfpt += ( np.sum(self.dstbn_n[i+1:]) ) / ( ( self.birth_rate*i
                               + self.immi_rate ) * self.dstbn_n[i] )
        return mfpt

    def mfpt_sub_dom(self):
        """
        Calculate arrays of the mfpt from dominance phase to exclusion phase for
        different deterministic fixed points.
        """
        mfpt_2sub   = np.zeros( self.nbr_species )
        mfpt_2dom   = np.zeros( self.nbr_species )

        for i in np.arange( self.nbr_species ):
            S = i + 1
            mean_n = int(self.deterministic_mean(S))
            mfpt_2dom[i] = self.mfpt_a2b( 0, mean_n )
            mfpt_2sub[i] = self.mfpt_b2a( 0, mean_n )

        return mfpt_2dom, mfpt_2sub

    def mean_richness_mfpt(self):
        """
        From arrays mfpt_2sub ( <T_{0}(n(R))> ) and mfpt_2dom ( <T_{n(R)}(0)> )
        get the richness distribution and average richness from detailed balance
        """

        rich_dstbn  = np.ones( self.nbr_species + 1 )
        richness    = np.arange( self.nbr_species + 1 )

        mfpt_2dom, mfpt_2sub = self.mfpt_sub_dom()

        for i in np.arange( self.nbr_species ):
            R = i + 1
            rich_dstbn[R] = rich_dstbn[R-1] * ( ( self.nbr_species - R + 1 ) *
                            mfpt_2sub[R-1] ) / ( R * mfpt_2dom[R-1] )

        rich_dstbn /= np.sum( rich_dstbn )

        return rich_dstbn, np.dot(rich_dstbn,richness), mfpt_2sub, mfpt_2dom



class CompareModels(object):
    """
    Trying to compare models in certain ways.
    """
    def __init__(self, comp_overlap=np.logspace(-3,0,41),
                 birth_rate=np.logspace(-2,2,41),
                 death_rate=np.logspace(-2,2,41),
                 carry_capacity=(np.logspace(1,3,41)).astype(int),
                 immi_rate=np.logspace(-3,1,41),
                 nbr_species=(np.logspace(0,1,41)).astype(int),
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

    def mlv_mfpt_dom_sub_ratio(self, key1, key2, file='mfptratio.npz'
                                    , plot=False, load_npz=False):
        """
        Compares along 3 metrics: entropy, richness and gini-simpson index

        Input
            key1        : The variable to vary
            key2        : 2nd variable
            files       : Name of file we want to save
            plot        : Whether or not to plot
            load_npz    : Whether or not to load from an npz file

        Output

        """
        filename = THRY_FIG_DIR + os.sep + file

        if load_npz: # load
            if not os.path.exists(filename): # check it exists
                print('No file to load!')
                raise SystemExit
            with np.load(filename) as f:
                mfpt_2sub = f['mfpt_2sub']; mfpt_2dom = f['mfpt_2dom'];
                prob_0 = f['prob_0']; av_rich_mfpt = f['av_rich_mfpt'];
                rich_dstbn_mfpt = f['rich_dstbn_mfpt'];
                xrange = f['xrange']; yrange = f['yrange'];
            print(">> Done loading " + str(filename))

        else:
            xrange = getattr(self,key1); yrange = getattr(self,key2)
            # <T_0(n(S))>
            mfpt_2sub = np.zeros( ( np.shape(getattr(self,key1))[0]
                                    , np.shape(getattr(self,key2))[0]
                                    , self.model.nbr_species ) )
            # <T_{n(S)}(0)>
            mfpt_2dom = np.zeros( ( np.shape(getattr(self,key1))[0]
                                    , np.shape(getattr(self,key2))[0]
                                    , self.model.nbr_species ) )
            # P(0)
            prob_0 = np.zeros( ( np.shape(getattr(self,key1))[0]
                                , np.shape(getattr(self,key2))[0] ) )
            # Use 1/<T> to get rates, use them to get richness distribution
            av_rich_mfpt = np.zeros( ( np.shape(getattr(self,key1))[0]
                                        , np.shape(getattr(self,key2))[0] ) )
            rich_dstbn_mfpt = np.zeros( ( np.shape(getattr(self,key1))[0]
                                        , np.shape(getattr(self,key2))[0]
                                        , self.model.nbr_species + 1 ) )

            # create heatmap array for metrics
            for i, valuei in enumerate(getattr(self,key1)):
                for j, valuej in enumerate(getattr(self,key2)):
                    setattr(self.model,key1,valuei)
                    setattr(self.model,key2,valuej)
                    #t = time.time()
                    probability, _  = self.model.abund_sid()
                    prob_0[i,j] = self.model.dstbn_n[0]
                    #print(time.time() - t)

                    rich_dstbn_mfpt[i,j], av_rich_mfpt[i,j], mfpt_2sub[i,j]\
                        , mfpt_2dom[i,j] = self.model.mean_richness_mfpt()

            metric_dict = {  'mfpt_2dom' : mfpt_2dom, 'mfpt_2sub' : mfpt_2sub
                            , 'prob_0'  : prob_0 , 'av_rich_mfpt' : av_rich_mfpt
                            , 'xrange'  : xrange , 'yrange'  : yrange
                            , 'rich_dstbn_mfpt' : rich_dstbn_mfpt
                            }
            np.savez(filename, **metric_dict)
            print(">> Done calculating MFPTs.")

        if plot:
            # style
            print(" >> Starting plotting...")
            plt.style.use('custom_heatmap.mplstyle')

            ## Mean richness 1-P0 vs from MFPT
            # calculate
            nbr_present  = self.model.nbr_species*( 1.0 - prob_0 )
            lines = np.arange(5,self.model.nbr_species,4)

            # plotting
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            MF = ax.contour( nbr_present.T, lines, linestyles='dotted'
                                , linewidths = 4)#, cmap='YlGnBu' )
            MFPT = ax.contour( av_rich_mfpt.T, lines, linestyles='solid'
                                , linewidths = 2)#, cmap='YlGnBu' )

            # labels and ticks
            POINTS_BETWEEN_X_TICKS = 10; POINTS_BETWEEN_Y_TICKS = 20
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
            plt.xlabel(VAR_NAME_DICT[key1]); plt.ylabel(VAR_NAME_DICT[key2]);
            #ax.invert_yaxis()

            # colorbar + legend
            CB = fig.colorbar(MF, shrink=1.0, extend='both')
            #CB.ax.get_children()[5].set_linewidths(5.0)
            fake_legend = [ Line2D([0],[0], color='k', lw=2, linestyle='dotted'
                            , label=r'$S(1-P(0))$')
                            , Line2D([0],[0], color='k', lw=2, linestyle='solid'
                            , label=r'$\langle R \rangle_{P_S}$')
                            ]
            ax.legend(handles=fake_legend, loc=4);
            plt.title(r'Mean species present'); plt.show()

            # save
            # style
            print(" >> Starting plotting...")
            plt.style.use('custom_heatmap.mplstyle')

            ## richness MFPT, S-R
            # calculate
            nbr_present = self.model.nbr_species*( 1.0 - prob_0 )
            rich_mfpt   = self.model.nbr_species*( 1.0/(mfpt_2dom[:,:,-1]/mfpt_2sub[:,:,-1] + 1 ) )
            lines = np.arange(5,self.model.nbr_species,4)

            # plotting
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            MF = ax.contour( nbr_present.T, lines, linestyles='dotted'
                                , linewidths = 4)#, cmap='YlGnBu' )
            MFPT = ax.contour( rich_mfpt.T, lines, linestyles='solid'
                                , linewidths = 2)#, cmap='YlGnBu' )

            # labels and ticks
            POINTS_BETWEEN_X_TICKS = 10; POINTS_BETWEEN_Y_TICKS = 20
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
            plt.xlabel(VAR_NAME_DICT[key1]); plt.ylabel(VAR_NAME_DICT[key2]);
            #ax.invert_yaxis()

            # colorbar + legend
            CB = fig.colorbar(MF, shrink=1.0, extend='both')
            #CB.ax.get_children()[5].set_linewidths(5.0)
            fake_legend = [ Line2D([0],[0], color='k', lw=2, linestyle='dotted'
                            , label=r'$S(1-P(0))$')
                            , Line2D([0],[0], color='k', lw=2, linestyle='solid'
                            , label=r'$ R_{fpt}$')
                            ]
            ax.legend(handles=fake_legend, loc=4);
            plt.title(r'Mean species present'); plt.show()

            # style
            print(" >> Starting plotting...")
            plt.style.use('custom_heatmap.mplstyle')

            ## richness mu/mean_time, S-R
            # calculate
            nbr_present = self.model.nbr_species*( 1.0 - prob_0 )
            mu_mfpt   = xrange * mfpt_2sub[:,:,-1].T
            lines = np.arange(5,self.model.nbr_species,4)

            # plotting
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            MF = ax.contour( nbr_present.T, lines, linestyles='dotted'
                                , linewidths = 4)#, cmap='YlGnBu' )
            MFPT = ax.contour( mu_mfpt, lines, linestyles='solid'
                                , linewidths = 2)#, cmap='YlGnBu' )

            # labels and ticks
            POINTS_BETWEEN_X_TICKS = 10; POINTS_BETWEEN_Y_TICKS = 20
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
            plt.xlabel(VAR_NAME_DICT[key1]); plt.ylabel(VAR_NAME_DICT[key2]);
            #ax.invert_yaxis()

            # colorbar + legend
            CB = fig.colorbar(MF, shrink=1.0, extend='both')
            #CB.ax.get_children()[5].set_linewidths(5.0)
            fake_legend = [ Line2D([0],[0], color='k', lw=2, linestyle='dotted'
                            , label=r'$S(1-P(0))$')
                            , Line2D([0],[0], color='k', lw=2, linestyle='solid'
                            , label=r'$ \mu  \langle T_0(n^*) \rangle$')
                            ]
            ax.legend(handles=fake_legend, loc=4);
            plt.title(r'Mean species present'); plt.show()

            # style
            print(" >> Starting plotting...")
            plt.style.use('custom_heatmap.mplstyle')

            ## richness mu/mean_time, S-R
            # calculate
            nbr_present = self.model.nbr_species*( 1.0 - prob_0 )
            rich_mu_mfpt = self.model.nbr_species / ( 1
                            + 1/(xrange*mfpt_2sub[:,:,-1].T ))
            lines = np.arange(5,self.model.nbr_species,4)

            # plotting
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            MF = ax.contour( nbr_present.T, lines, linestyles='dotted'
                                , linewidths = 4)#, cmap='YlGnBu' )
            MFPT = ax.contour( rich_mu_mfpt, lines, linestyles='solid'
                                , linewidths = 2)#, cmap='YlGnBu' )

            # labels and ticks
            POINTS_BETWEEN_X_TICKS = 10; POINTS_BETWEEN_Y_TICKS = 20
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
            plt.xlabel(VAR_NAME_DICT[key1]); plt.ylabel(VAR_NAME_DICT[key2]);
            #ax.invert_yaxis()

            # colorbar + legend
            CB = fig.colorbar(MF, shrink=1.0, extend='both')
            #CB.ax.get_children()[5].set_linewidths(5.0)
            fake_legend = [ Line2D([0],[0], color='k', lw=2, linestyle='dotted'
                            , label=r'$S(1-P(0))$')
                            , Line2D([0],[0], color='k', lw=2, linestyle='solid'
                            , label=r'$ R(\mu, \langle T_0(n^*)) $')
                            ]
            ax.legend(handles=fake_legend, loc=4);
            plt.title(r'Mean species present'); plt.show()

        # style
        print(" >> Starting plotting...")
        plt.style.use('custom_heatmap.mplstyle')

        ## richness mu/mean_time, S-R
        # calculate
        nbr_present = self.model.nbr_species*( 1.0 - prob_0 )
        rich_mu_mfpt = self.model.nbr_species / ( 1
                        + 1/(xrange*mfpt_2sub[:,:,-1].T ))
        lines = np.arange(5,self.model.nbr_species,4)

        # plotting
        f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
        MF = ax.contour( nbr_present.T, lines, linestyles='dotted'
                            , linewidths = 4)#, cmap='YlGnBu' )
        MFPT = ax.contour( rich_mu_mfpt, lines, linestyles='solid'
                            , linewidths = 2)#, cmap='YlGnBu' )

        # labels and ticks
        POINTS_BETWEEN_X_TICKS = 10; POINTS_BETWEEN_Y_TICKS = 20
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
        plt.xlabel(VAR_NAME_DICT[key1]); plt.ylabel(VAR_NAME_DICT[key2]);
        #ax.invert_yaxis()

        ax.legend(handles=fake_legend, loc=4);
        plt.title(r'Return time $\langle T_0(0) \rangle$'); plt.show()


        return 0

    def richness_from_mfpt():

        return 0

    def mlv_metric_compare_heatmap(self, key1, key2, file='metric3.npz'
                                        , plot=False, load_npz=False):
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
                JS = f['JS']; xrange = f['xrange']; yrange = f['yrange'];
                approx_dist_sid = f['approx_dist_sid']
                approx_dist_nava = f['approx_dist_nava']
                #mean_time_dominance = f['mean_time_dominance']

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
                                    int(self.model.nbr_species*(1
                                    - probability[0])), 0, probability)
                    det_mean[i,j] = self.model.deterministic_mean()

                    print('>'+str(j))
                print('>>>'+str(i))

            # save
            metric_dict = {'H' : H, 'GS' : GS, 'richness' : richness, 'JS' : JS
                            , 'approx_dist_sid' : approx_dist_sid
                            , 'approx_dist_nava' : approx_dist_nava
                            , 'det_mean' : det_mean
                            , 'mean_time_dominance' : mean_time_dominance
                            ,'xrange' : xrange, 'yrange' : yrange}
            np.savez(filename, **metric_dict)

        if plot:

            # some settings for the heatmaps
            imshow_kw = {'cmap': 'YlGnBu', 'aspect': None
                         #,'vmin': vmin, 'vmax': vmax
                         #,'norm': mpl.colors.LogNorm(vmin,vmax)
                    }
            approx_dist = approx_dist_nava
            # setting of xticks
            POINTS_BETWEEN_X_TICKS = 10
            POINTS_BETWEEN_Y_TICKS = 20
            xrange   = getattr(self,key1)
            yrange   = getattr(self,key2)
            ## Entropy 2D
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(H.T, interpolation='none', **imshow_kw)

            # labels and ticks
            print
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
            plt.title('shannon entropy')
            plt.show()

            ## Gini-Simpson 2D
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(GS.T, interpolation='none', **imshow_kw)

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
            plt.xlabel(VAR_NAME_DICT[key1]); plt.ylabel(VAR_NAME_DICT[key2])
            ax.invert_yaxis()
            plt.title(r'Gini-Simpson index')
            #plt.xscale('log'); plt.yscale('log')

            # colorbar
            plt.colorbar(im,ax=ax)
            plt.show()

            ## Richness 2D
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(richness.T, interpolation='none', **imshow_kw)

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
            plt.title(r'$1-P(0)$')
            plt.show()


            ## Jensen-shannon
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(JS.T, interpolation='none', **imshow_kw)

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
            plt.title(r'$Jensen-Shannon divergence$')
            plt.show()

            ## Mean time extinction
            """ TODO uncomment
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
            """

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
                plt.xlabel(VAR_NAME_DICT[key1]); plt.ylabel(VAR_NAME_DICT[key2])
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
        #print( i, np.argmax(dstbn[offset:]) )
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

def deterministic_mean(param_dict, immi_rate=1.0, comp_overlap=1.0
                                , nbr_species_present=0):
    """
    Calculates the mean of the LV equations, for com_overlap between 0
    and 1.
    """
    if nbr_species_present==0:
        nbr_species_present = param_dict['nbr_species']
    return param_dict['carry_capacity']*( ( 1. + np.sqrt( 1.+ 4.*immi_rate*
           ( 1. + comp_overlap*( nbr_species_present - 1. ) ) /
           (param_dict['carry_capacity']*(param_dict['birth_rate']-param_dict['death_rate']) ) ) )
           / ( 2.*( 1.+comp_overlap*( nbr_species_present-1.) ) ) )

def our_asymp(param_dict, immi_rate=1.0, comp_overlap=1.0, maxn=1000):
    J = param_dict['nbr_species'] * deterministic_mean(param_dict, immi_rate, comp_overlap)
    K = param_dict['carry_capacity']
    prob_n_unnormalized = np.zeros( maxn );
    prob_n_unnormalized[0] = 1.0#1E-250;
    n = np.arange(1, maxn)
    prob_n_unnormalized[1:] = ( param_dict['birth_rate'] / ( param_dict['death_rate'] +
                ( (param_dict['birth_rate'] - param_dict['death_rate'] )
                * J / K ) ) ** n * n ** (immi_rate/param_dict['birth_rate']-1.0) )

    dstbn = prob_n_unnormalized / np.sum(prob_n_unnormalized)
    return np.arange(maxn), dstbn

def our_approx_neutral(param_dict, immi_rate=1.0, comp_overlap=1.0, maxn=1000):
    J = param_dict['nbr_species'] * deterministic_mean(param_dict, immi_rate, comp_overlap)
    prob_n_unnormalized = np.zeros( maxn );
    a = immi_rate/param_dict['birth_rate']
    prob_n_unnormalized[0] = 1.0
    factor = ( param_dict['birth_rate'] )
    K = param_dict['carry_capacity']
    prob_n_unnormalized[1:] *= factor
    for n in np.arange(1, maxn):
        prob_n_unnormalized[n] = prob_n_unnormalized[n-1] * ( a + n - 1. ) / n

    limitJ = np.arange(1, maxn)
    prob_n_unnormalized[1:] /= ( param_dict['death_rate'] +
                        ( (param_dict['birth_rate'] - param_dict['death_rate'])
                        * limitJ / K ) ) ** np.arange(1, maxn)

    dstbn = prob_n_unnormalized / np.sum(prob_n_unnormalized)
    return np.arange(maxn), dstbn

def baxter(param_dict, immi_rate=1.0, maxn=1000):
    S = param_dict['nbr_species']
    J = int( S * deterministic_mean(param_dict, immi_rate, 1.0) )
    a = immi_rate#/param_dict['birth_rate']
    prob_n_unnormalized = np.zeros( maxn );
    prob_n_unnormalized[0] = 1.0
    n = np.arange(1,int(J))
    prob_n_unnormalized = ( n/J ) ** ( a - 1. ) * np.exp(-n/J *( 2 *a*(S-1) - 1.) )
    dstbn = prob_n_unnormalized / np.sum(prob_n_unnormalized)
    return n, dstbn


def alonso(param_dict, immi_rate=1.0, maxn=1000):
    S = param_dict['nbr_species']
    P = 1.0 / S
    J = int( S * deterministic_mean(param_dict, immi_rate, 1.0) )
    m = immi_rate / ( J * P * ( param_dict['birth_rate'] + immi_rate * S +
                param_dict['death_rate'] + ( param_dict['birth_rate']
                - param_dict['birth_rate'] ) / param_dict['carry_capacity'] ) )
    nstar = ( J - m ) / ( 1.0 - m ) - P
    Pstar = m * ( J - 1.0 ) * P / ( 1.0 - m )
    n = np.arange(maxn)
    dstbn = comb(J,n) * beta( n + P, nstar - n ) / beta( Pstar, nstar - J )

    return np.arange(maxn), dstbn


if __name__ == "__main__":
    # multimodal phase
    """
    multimodal_params = {'birth_rate' : 20.0, 'death_rate'     : 1.0
                                            , 'immi_rate'       : 0.001
                                            , 'carry_capacity'  : 100
                                            , 'comp_overlap'    : 0.8689
                                            , 'nbr_species'     : 30
                                            }

    model = Model_MultiLVim(**multimodal_params)
    distribution = model.abund_jer()
    fig = plt.figure(); end = int(1.5*model.carry_capacity) # cut somehere
    plt.plot( np.arange(end), distribution[:end])
    plt.xlabel(r"population, $n_i$")
    plt.ylabel(r"probability $P(n_i)$")
    plt.yscale('log')
    #plt.title(r"$\mu=${}, $\rho=${} ".format( params['immi_rate']
    #                                            , params['comp_overlap']))
    plt.show()
    """

    compare = CompareModels()
    #compare.mlv_mfpt_dom_sub_ratio("immi_rate","comp_overlap", file='mfptratio.npz', plot=True, load_npz=True)
    compare.mlv_metric_compare_heatmap("immi_rate","comp_overlap", file='metric45.npz', plot=True, load_npz=True)
