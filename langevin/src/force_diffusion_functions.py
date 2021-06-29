import numpy as np

class AttrDict(dict):
    # Class that makes an obect with callable attributes from the dictionary
    # keys in dict. I find it easier to work with.
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def lotkvol_force( param_dict ):
    params = AttrDict( param_dict )

    def wrapped( population ): # <- factori-ed / created function
        force = np.zeros( len(population) ); J = np.sum(population)
        """
        for i in np.arange( 0, len(population) ):
            force[i] = params.immiRate/params.carryCapacity + ( params.birthRate
                        - params.deathRate ) * population[i] * ( 1.0 -
                        ( ( 1.0 - params.compOverlap ) * population[i] +
                        params.compOverlap * J ) )
        """
        force = params.immiRate/params.carryCapacity + ( params.birthRate
                        - params.deathRate ) * population * ( 1.0 -
                        ( ( 1.0 - params.compOverlap ) * population +
                        params.compOverlap * J ) )
        return force

    return wrapped

def master_diff( param_dict ):
    params = AttrDict( param_dict )

    def wrapped( population ):
        diff = np.zeros( len(population) ); J = np.sum(population)
        """
        for i in np.arange( 0, len(population) ):
            diff[i] = ( params.immiRate/params.carryCapacity +
                        ( params.birthRate + params.deathRate ) * population[i]
                         + ( params.birthRate - params.deathRate )
                         * population[i] * ( ( 1.0 - params.compOverlap )
                         * population[i] + params.compOverlap * J ) )
        """
        diff = ( params.immiRate/params.carryCapacity +
                    ( params.birthRate + params.deathRate ) * population
                     + ( params.birthRate - params.deathRate )
                     * population * ( ( 1.0 - params.compOverlap )
                     * population + params.compOverlap * J ) )
        return np.sqrt( diff / params.carryCapacity )

    return wrapped

def mehta_diff( param_dict ):
    params = AttrDict( param_dict )

    def wrapped( population ):
        # simply sqrt the population size, ignoring all quadratic terms
        return np.sqrt( ( params.birthRate + params.deathRate ) * population
                        / params.carryCapacity )

    return wrapped
