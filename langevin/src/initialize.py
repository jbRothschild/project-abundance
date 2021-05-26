from scipy.optimize import fsolve as sp_solver
import random
import numpy as np

def mean_field_solution(carryCapacity, birthRate, deathRate, compOverlap
                                    , immiRate, nbrSpecies):
    """
    Mean field approximation :  Solving Master equation by assuming
                                <J|n_1>=(S-1)<n>+n_1
    """
    population_max = int(carryCapacity*2)
    population = np.arange(0,population_max)
    def fcn_prob_n(mean_n):
        prob_n_unnormalized = np.zeros( population_max );
        prob_n_unnormalized[0] = 1.0#1E-250;
        for n in np.arange( 1, len(prob_n_unnormalized)):
            previous_n = n - 1
            prob_n_unnormalized[n] = prob_n_unnormalized[previous_n] * (
                ( immiRate + birthRate*previous_n)
                / ( n* (deathRate + (1.0-compOverlap)*n*(
                birthRate-deathRate)/carryCapacity
                + ( birthRate-deathRate)*(
                (nbrSpecies-1.0)*mean_n + n )*compOverlap
                / carryCapacity ) ) )

        prob_n = prob_n_unnormalized / ( np.sum( prob_n_unnormalized ) )

        return prob_n

    def equations(vars):
        mean_n = vars

        dstbn_n = fcn_prob_n(mean_n); eqn = 1.0

        eqn = np.dot(dstbn_n, population)-mean_n

        return eqn

    # TODO Maybe something better?
    initial_guess = carryCapacity / ( 1.0 +
                    compOverlap*(nbrSpecies-1) )

    # numerically solve for <n>
    mean_n_approx = sp_solver(equations, initial_guess)

    # Probability distribution of n
    dstbn_n = fcn_prob_n(mean_n_approx)

    # Abundance of n
    abundance = dstbn_n * nbrSpecies

    return dstbn_n, abundance

def initialize_pop(param):
    ss_probability, _ = mean_field_solution( **param )
    ss_cum = np.cumsum(ss_probability)

    # initialize species state
    initial_state = np.zeros( (param['nbrSpecies']) )

    # Create an initial steady state abundance
    for i in np.arange(param['nbrSpecies']):
        sample = random.random()
        j = 0
        while sample > ss_cum[j]:
            j += 1
        initial_state[i] = j
    #sys.exit()
    initial_state[np.argmin(initial_state)] = 0

    return initial_state/param['carryCapacity']
