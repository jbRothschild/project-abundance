import numpy as np; import csv, os, glob, csv, time, datetime

import matplotlib as mpl; import matplotlib.pyplot as plt
import matplotlib.patches as patches # histgram animation
import matplotlib.path as path # histogram animation
from matplotlib import animation # animation

import itertools
from itertools import combinations

from scipy.special import gamma, poch, factorial, hyp1f1 # certain functions
import scipy
import scipy.io as spo
from scipy.optimize import fsolve as sp_solver # numerical solver, can change to
                                               # a couple others, i.e. broyden1

#import seaborn as sns

import theory_equations as te

plt.style.use('custom.mplstyle')

# TODO : all these scripts are a bit of a mess. FIgure out a good modular way
#        to do this all.


FIGURE_DIR = 'figures'

while not os.path.exists( os.getcwd() + os.sep + FIGURE_DIR ):
    os.makedirs(os.getcwd() + os.sep + FIGURE_DIR);

class Model_MultiLVim(object):
    """
    Various solutions from different models of populations with competition in
    the death rate. Will describe where they come from each.
    """
    def __init__(self, comp_overlap=0.5, birth_rate=2.0, death_rate=1.0,
                 carry_capacity=100, immi_rate=0.01, nbr_species=30,
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

    def abund_J(self, technique='Jeremy'):
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
        Solving Master equation by assuming <J|n_1>=(S-1)<n>+n_1
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

    def abund_1spec_MSLV(self, dstbn_approx = 'Nava'):
        """
        Abundance distribution of stochastic Lotka-Volterra with immigration.
        Exact solution for comp_overlap = 0 or 1. Approximations involving the
        distribution of the total number of individuals for other comp_overlap
        """

        if self.comp_overlap == 0.0:
            a = self.immi_rate / self.birth_rate;
            b = self.death_rate * self.carry_capacity / ( 1.0*(self.birth_rate
                - self.death_rate) );
            c = self.birth_rate * self.carry_capacity / ( 1.0*(self.birth_rate
                - self.death_rate) );

            probability = np.zeros( np.shape(self.population) )
            #probability[0] = (1.0/hyp1f1(a,b,c))
            probability[0] = 1.0

            for i in np.arange(1,len(probability)):
                probability[i] = probability[i-1]*(self.birth_rate*(i-1)
                                   + self.immi_rate ) / (i*( self.death_rate +
                                   (self.birth_rate-self.death_rate)*( i )
                                   / self.carry_capacity )  )

            probability = probability/np.sum(probability)

        #elif self.comp_overlap == 1.0:
            # TODO : Type it all out
        #    abundance = self.population

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

class CompareModels(object):
    """
    Various solutions from different models of populations with competition in
    the death rate. Will describe where they come from each.
    """
    def __init__(self, comp_overlap=np.logspace(-2,0,100),
                 birth_rate=np.logspace(-2,2,100),
                 death_rate=np.logspace(-2,2,100),
                 carry_capacity=(np.logspace(1,3,100)).astype(int),
                 immi_rate=np.logspace(-2,2,100),
                 nbr_species=(np.logspace(0,3,100)).astype(int),
                 model=Model_MultiLVim):
        self.comp_overlap = comp_overlap; self.birth_rate = birth_rate;
        self.death_rate = death_rate; self.immi_rate = immi_rate;
        self.carry_capacity = carry_capacity; self.nbr_species = nbr_species;

        self.model = model

    def metric_compare(self, key1, key2=None):
        """
        Input
            key : The variable to vary

        Output
            H : shannon entropy as a function of var (var_array)
            richness : 1 - P(0), probability of being present
            species_richness : average number of species in the system
            GS : Gini-Simpson index
        """
        if key2 == None:
            H = np.zeros( np.shape(getattr(self,key1)) )
            richness = np.zeros( np.shape(getattr(self,key1)) )
            GS = np.zeros( np.shape(getattr(self,key1)) )

            for i, value in enumerate(getattr(self,key1)):
                setattr(self.model,key1,value)
                probability, _ = self.model.abund_sid()
                richness[i] = 1.0 - probability[0]
                # TODO : Should I be excluding probability[0] in shannon entropy and GS?
                H[i] = - np.dot(probability,np.log(probability))
                GS[i] = 1 - np.dot(probability,probability)
                print(i)

            ## Entropy 1D
            fig = plt.figure()
            plt.plot(getattr(self,key1),H)
            plt.ylabel("entropy")
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
            if key1 == 'nbr_species':
                species_richness = richness * getattr(self,key1)
                fig = plt.figure()
                plt.plot(getattr(self,key1),species_richness)
                plt.ylabel(r"$average species richness$")
                plt.xlabel(key)
                #plt.yscale('log')
                plt.xscale('log')
                plt.show()

        ### HEATMAPS
        else:
            H = np.zeros( ( np.shape(getattr(self,key1))[0],
                         np.shape(getattr(self,key2))[0] ) )
            richness = np.zeros( ( np.shape(getattr(self,key1))[0],
                         np.shape(getattr(self,key2))[0] ) )
            GS = np.zeros( ( np.shape(getattr(self,key1))[0],
                         np.shape(getattr(self,key2))[0] ) )

            for i, valuei in enumerate(getattr(self,key1)):
                for j, valuej in enumerate(getattr(self,key2)):
                    setattr(self.model,key1,valuei)
                    setattr(self.model,key2,valuej)
                    probability, _ = self.model.abund_sid()
            # TODO : Should I be excluding probability[0] in shannon entropy and GS?
                    H[i,j] = - np.sum(probability*np.log(probability))
                    GS[i,j] = 1 - np.dot(probability,probability)
                    richness[i,j] = 1.0 - probability[0]

                print(i)


            imshow_kw = {'cmap': 'YlGnBu', 'aspect': None
                         #,'vmin': vmin, 'vmax': vmax
                         #,'norm': mpl.colors.LogNorm(vmin,vmax)
                        }

            xrange = getattr(self,key1); yrange = getattr(self,key2)
            POINTS_BETWEEN_X_TICKS = 20
            POINTS_BETWEEN_Y_TICKS = 20
            FS = 12

            ## Entropy 2D
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(H, interpolation='none', **imshow_kw)

            # labels and ticks
            ax.set_xticks([i for i, cval in enumerate(xrange) if i % POINTS_BETWEEN_X_TICKS == 0])
            ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval) for i, xval in enumerate(xrange) if (i % POINTS_BETWEEN_X_TICKS==0)], fontsize=FS)
            ax.set_yticks([i for i, kval in enumerate(yrange) if i % POINTS_BETWEEN_Y_TICKS == 0])
            ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval) for i, yval in enumerate(yrange) if i % POINTS_BETWEEN_Y_TICKS==0], fontsize=FS)

            plt.xlabel(key1); plt.ylabel(key2)
            ax.invert_yaxis()
            #plt.xscale('log'); plt.yscale('log')
            plt.title('shannon entropy')
            plt.show()

            ## Gini-Simpson 2D
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(GS, interpolation='none', **imshow_kw)

            # labels and ticks
            ax.set_xticks([i for i, cval in enumerate(xrange) if i % POINTS_BETWEEN_X_TICKS == 0])
            ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval) for i, xval in enumerate(xrange) if (i % POINTS_BETWEEN_X_TICKS==0)], fontsize=FS)
            ax.set_yticks([i for i, kval in enumerate(yrange) if i % POINTS_BETWEEN_Y_TICKS == 0])
            ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval) for i, yval in enumerate(yrange) if i % POINTS_BETWEEN_Y_TICKS==0], fontsize=FS)

            plt.xlabel(key1); plt.ylabel(key2)
            ax.invert_yaxis()
            #plt.xscale('log'); plt.yscale('log')
            plt.title(r'Gini-Simpson index')
            plt.show()

            ## Richness 2D
            f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
            im = ax.imshow(richness, interpolation='none', **imshow_kw)

            # labels and ticks
            ax.set_xticks([i for i, cval in enumerate(xrange) if i % POINTS_BETWEEN_X_TICKS == 0])
            ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval) for i, xval in enumerate(xrange) if (i % POINTS_BETWEEN_X_TICKS==0)], fontsize=FS)
            ax.set_yticks([i for i, kval in enumerate(yrange) if i % POINTS_BETWEEN_Y_TICKS == 0])
            ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval) for i, yval in enumerate(yrange) if i % POINTS_BETWEEN_Y_TICKS==0], fontsize=FS)

            plt.xlabel(key1); plt.ylabel(key2)
            ax.invert_yaxis()
            #plt.xscale('log'); plt.yscale('log')
            plt.title(r'$1-P(0)$')
            plt.show()

            ## Species richness 2D
            if key1 == 'nbr_species' or key2 == 'nbr_species':
                species_richness = np.zeros( ( np.shape(getattr(self,key1))[0],
                              np.shape(getattr(self,key2))[0] ) )

                if key1 == 'nbr_species':
                    species_richness = richness * getattr(self,key1)
                else:
                    species_richness = ( richness.T * getattr(self,key1) ).T

                f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
                im = ax.imshow(species_richness, interpolation='none', **imshow_kw)

                # labels and ticks
                POINTS_BETWEEN_X_TICKS = 20
                POINTS_BETWEEN_Y_TICKS = 50
                ax.set_xticks([i for i, cval in enumerate(xrange) if i % POINTS_BETWEEN_X_TICKS == 0])
                ax.set_xticklabels([r'$10^{%d}$' % np.log10(xval) for i, xval in enumerate(xrange) if (i % POINTS_BETWEEN_X_TICKS==0)], fontsize=FS)
                ax.set_yticks([i for i, kval in enumerate(yrange) if i % POINTS_BETWEEN_Y_TICKS == 0])
                ax.set_yticklabels([r'$10^{%d}$' % np.log10(yval) for i, yval in enumerate(yrange) if i % POINTS_BETWEEN_Y_TICKS==0], fontsize=FS)

                plt.xlabel(key1); plt.ylabel(key2)
                ax.invert_yaxis()
                #plt.xscale('log'); plt.yscale('log')
                plt.title(r'species richness')
                plt.show()

            return H, richness

    def compare_abundance_approx_MSLVim(self):
        """
        Compare different approximations for abundance Distribution

        """

        matlab_files = ['Results_01','Results_05','Results_1']

        for i, comp_o in enumerate([0.1,0.5,0.999]):
            EqnsMLV = self.model(comp_overlap=comp_o)
            #abund_moranton_const = EqnsMLV.abund_moranton()
            #abund_moranton = EqnsMLV.abund_moranton(const_J=False)
            prob_sid, abund_sid = EqnsMLV.abund_sid()
            #abund_kolmog = EqnsMLV.abund_kolmog(const_J=False)
            #prob_jeremy, abund_jeremy = EqnsMLV.abund_1spec_MSLV(dstbn_approx='Jeremy')
            tic = time.clock()
            prob_nava, abund_nava = EqnsMLV.abund_1spec_MSLV(dstbn_approx =
                                                             'Nava')
            toc = time.clock()
            print(str(datetime.timedelta(seconds=int(toc-tic))))

            dict_matlab = spo.loadmat('nava_results'+os.sep+matlab_files[i])

            abund_nava_m = dict_matlab['QN']
            abund_sid_m = dict_matlab['Q_Ant']

            #print(np.sum(abund_sid),np.sum(abund_jeremy),np.sum(abund_nava))
            end = 300
            fig = plt.figure()
            #plt.plot(EqnsMLV.population,abund_moranton_const,
                      #label='Moranton cst.')
            #plt.plot(EqnsMLV.population,abund_moranton,label='Moranton')
            plt.plot(EqnsMLV.population[:end], abund_sid[:end],
                     label='A.-S. approx.')
            #plt.plot(EqnsMLV.population,abund_kolmog,label='Kolmog. approx')
            #plt.plot(EqnsMLV.population[:end], abund_jeremy[:end],
                      #label='Jeremy approx.')
            plt.plot(EqnsMLV.population[:end], abund_nava[:end],
                     label='Nava approx.')
            plt.plot(EqnsMLV.population[:end],
                     EqnsMLV.nbr_species*abund_sid_m[:end],
                     label='A.-S. Matlab', linestyle='-.')
            plt.plot(EqnsMLV.population[:end],
                     EqnsMLV.nbr_species*abund_nava_m[:end],
                     label='Nava Matlab', linestyle='-.')
            plt.legend(loc='best')
            plt.ylabel("abundance")
            plt.xlabel("population size")
            plt.yscale('log')
            plt.show()

            return 0

if __name__ == "__main__":

    #dict_matlab = spo.loadmat('nava_results/Results_01')
    #print(dict_matlab['NumberOfSpecies'])

    compare = CompareModels()
    compare.compare_abundance_approx_MSLVim()
    #compare.metric_compare("carry_capacity","immi_rate")
