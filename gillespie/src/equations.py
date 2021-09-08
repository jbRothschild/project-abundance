import numpy as np

def trial():
    print("done")
    return 0

def deterministic_mean(nbr_species, mu, rho, rplus, rminus, K):
    # Deterministic mean fixed point Solution to LV equations
    det_mean = K*( ( 1. + np.sqrt( 1.+ 4.*mu*( 1. + rho*( nbr_species - 1. ) ) /
                (K*(rplus-rminus)) ) ) / ( 2.*( 1. + rho*(nbr_species-1.) ) ) )
    return det_mean

def meanJ_sim(dstbn, nbr_species):
    # calculating <J> from a distribution
    return nbr_species * np.dot(dstbn, np.arange( len(dstbn) ) )

def meanJ_est(dstbn, nbr_species):
    # 2D array of <J>
    meanJ = np.zeros((np.shape(dstbn)[0],np.shape(dstbn)[1]))
    for i in np.arange(np.shape(dstbn)[0]):
        for j in np.arange(np.shape(dstbn)[1]):
            dstbnij = dstbn[i,j]
            meanJ[i,j] = meanJ_sim(dstbnij, nbr_species)
    #f = plt.figure(); fig = plt.gcf(); ax = plt.gca() im = ax.imshow(meanJ.T);
    #plt.colorbar(im,ax=ax); ax.invert_yaxis(); plt.show()
    return meanJ

def death_rate( n, dstbn, rplus, rminus, K, rho, S):
    # death rate for n
    J = meanJ_sim(dstbn, S) # could change to just meanJ
    r = rplus - rminus
    return rminus * n + r * n * ( ( 1. - rho ) + rho * J ) / K

def richness_from_rates( dstbn, rplus, rminus, K, rho, mu, S ):
    # Approximation for the richness given the distribution... stupid, because
    # why not simply use P(0)? We changed this I believe
    deathTimesP1 = death_rate(1, dstbn, rplus, rminus, K, rho, S)*dstbn[1]
    return ( mu*S - deathTimesP1 )/( deathTimesP1 - mu )

def mfpt_a2b( dstbn, a, b, mu=1.0, rplus=2.0 ):
    """
    From distribution, get the mfpt <T_{b}(a)>, a<b
    """
    mfpt = 0.0
    for i in np.arange(a,b):
        mfpt += np.divide( np.sum( dstbn[:i+1] )  , ( rplus*i + mu )*dstbn[i] )
    return mfpt

def mfpt_b2a( dstbn, a, b, mu=1.0, rplus=2.0 ):
    """
    From distribution, get the mfpt <T_{a}(b)>
    """
    mfpt = 0
    for i in np.arange(a,b):
        mfpt += np.divide( np.sum(dstbn[i+1:]) , ( rplus*i + mu ) * dstbn[i] )
    return mfpt

def mfpt_020( dstbn, mu=1.0 ):
    """
    From distribution, get the mfpt <T(0\rightarrow 0)>
    """
    return np.divide( 1., ( dstbn[0] * mu ) )

def mfpt_a2a( dstbn, a, mu=1.0, rplus=2.0, rminus=1.0, K=100, rho=1.0, S=30 ):
    """
    From distribution, get the mfpt <T_{a}(a)> for a!=0
    """
    return np.divide( ( 1. + dstbn[a] ) , ( dstbn[a]
            * ( rplus*a + mu + a*( rminus + ( rplus-rminus )
            * ( ( 1. - rho ) * a + rho * meanJ_sim(dstbn, S) ) / K ) ) ) )

def peak_analytical_method1(meanN, rplus, rminus, K, rho, mu, S, pos=1.):
    """
    Find the peak of the distribution if <n_j|n_i> = <n>, (S-1)*<n>
    pos (int) : +1 or -1
    """
    assert (pos==1. or pos==-1.), "pos neither +1 or -1. Not solution to quadratic eq."
    return ( ( K - (S - 1.) * np.multiply( rho, meanN) ) + pos * np.sqrt(( K - (S - 1.)
                * np.multiply( rho, meanN) )**2 + 4 * ( mu - rplus ) * K / (rplus-rminus) + 0j ) ) / 2.0


def peak_analytical_method3(meanN, rplus, rminus, K, rho, mu, S, pos=1.):
    """
    Find the peak of the distribution if <J|n_i> = <J> = S<n>
    """
    assert (pos==1. or pos==-1.), "pos neither +1 or -1. Not solution to quadratic eq."
    return ( ( K - S * np.multiply( rho, meanN) )/2.0 * ( 1.0 /( 1.0 - rho ) ) * ( 1.0
                + pos * np.sqrt(1.0 + 4 * K * ( mu - rplus ) * ( 1.0 - rho ) *
                ( 1.0 / ( K - S * np.multiply( rho, meanN) )**2 ) / (rplus-rminus) + 0j) ) )

def boundaryI(meanN, rplus, rminus, K, rho, mu, S):
    """
    return 2 if in Regime I, 0 if not. Using method 1
    """
    regimeI = ( 1. / mu ) * ( rminus + (rplus-rminus) * ( 1. + ( S - 1.0 )
                                            * np.multiply( rho, meanN) ) / K )
    return regimeI

def realBoundaryIII(meanN, rplus, rminus, K, rho, mu, S):
    """
    return 1 if in Regime III, 0 if not. Using method 1
    """
    boundaryIII = ( rplus - rminus) * ( K - ( S - 1.0) * np.multiply( rho, meanN) )**2 \
                        + 4 * K *( mu - rplus )

    conditionAlt = peak_analytical_method1( meanN, rplus, rminus, K, rho, mu, S)
    array2 = np.zeros( ( np.shape(conditionAlt)[0], np.shape(conditionAlt)[1] ) )
    array2[np.imag(conditionAlt) != 0.0 ] = 1.0

    return boundaryIII, array2

def realBoundaryIII3(meanN, rplus, rminus, K, rho, mu, S):
    """
    return 2 if in Regime III, 0 if not. Using method 3
    """
    boundaryIII =  ( ( (rplus-rminus)*(K- S * np.multiply( rho, meanN) )**2 ) * (1./ (4. * K)) *
                        ( 1.0 / ( rplus - mu ) ) * ( 1.0 /( 1.0 - rho ) ) )
    boundaryIII = (rplus-rminus) * (K - S * np.multiply( rho, meanN) )**2 \
                                + 4 * K * ( mu - rplus ) * ( 1.0 - rho )
    # positive condition
    conditionNeg = peak_analytical_method3( meanN, rplus, rminus, K, rho, mu, S, -1.0)
    conditionPos = peak_analytical_method3( meanN, rplus, rminus, K, rho, mu, S, 1.0)
    array = np.zeros( ( np.shape(conditionPos)[0], np.shape(conditionPos)[1] ) )
    array2 = np.zeros( ( np.shape(conditionNeg)[0], np.shape(conditionNeg)[1] ) )
    array3 = np.zeros( ( np.shape(boundaryIII)[0], np.shape(boundaryIII)[1] ) )
    array[np.real(conditionPos) >= 0.0] = 1.0; array2[np.real(conditionNeg) >= 0.0] = 1.0
    array3[boundaryIII < 0.0] = 1.0
    condition = array + array2 + array3; condition[condition > 0.0] = 1.0
    boundaryIII *=  condition

    return boundaryIII, condition

def positiveBoundaryIII(meanN, rplus, rminus, K, rho, mu, S, pos=1.):
    """
    return 2 if in Regime III, 0 if not. Using method 1
    """
    posSol = peak_analytical_method1( meanN, rplus, rminus, K, rho, mu, S, 1.0)
    negSol = peak_analytical_method1( meanN, rplus, rminus, K, rho, mu, S, -1.0)

    #array = np.zeros( ( np.shape(condition)[0], np.shape(condition)[0] ) )
    #array2 = np.zeros( ( np.shape(conditionAlt)[0], np.shape(conditionAlt)[0] ) )

def mfpt_1species( fNmin, fluxNtilde, S):
    return fluxNtilde * (1.0 / fNmin ) / S

def mfpt_hubbel_regime(fluxNtilde, mu, nbr_spec_pres, S ):
    return ( fluxNtilde * nbr_spec_pres ) * ( 1.0 / ( mu * ( S - nbr_spec_pres) ) )

def mfpt_return_full(fNmin, mu, nbr_spec_pres, S):
    return ( ( S - nbr_spec_pres ) * mu ) * (1.0 / fNmin ) / S**2
