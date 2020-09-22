import numpy as np
from numpy.linalg import inv, eig
from scipy.linalg import expm

class Params(object):

    def __init__(self):
        self.beta= 20.; self.mu = 1.; self.gamma = 10.; self.N = 4.

def create_full_markov( params ):
    """
    Makes the transition matrix which is M[i,j]=M[I*N+S,I*N+S)], wherein
    transition is from state i to j.
    To get S : use modulo of indices i%N
    To get I : use quotient of inddices i//N
    """
    M = np.zeros( ( int(params.N+1)**2,int(params.N+1)**2 ) )

    # bulk
    for S in np.arange(1,int(params.N)):
        for I in np.arange(1,int(params.N)):
            # death of infected
            M[int((I-1)*(params.N+1.)+S),int(I*(params.N+1.)+S)] = params.gamma*I
            # death of susceptible
            M[int(I*(params.N+1.)+S-1),int(I*(params.N+1.)+S)]   = params.mu*S
            # birth of susceptible
            M[int(I*(params.N+1.)+S+1),int(I*(params.N+1.)+S)]   = params.mu*params.N
            # infection
            M[int((I+1)*(params.N+1.)+S-1),int(I*(params.N+1.)+S)] = (params.beta*S*I
                                                                    /params.N)

    ## FILL IN S=0, no death of susceptible or becoming infected
    S = 0
    for I in np.arange(0,int(params.N)+1):
        # birth susceptible
        M[int(I*(params.N+1.)+S+1),int(I*(params.N+1.)+S)] = params.mu*params.N

    for I in np.arange(1,int(params.N)+1):
        # death of infected
        M[int((I-1)*(params.N+1.)+S),int(I*(params.N+1.)+S)] = params.gamma*I


    ## FILL IN S=N, no birth of susceptible
    S = params.N
    for I in np.arange(0,int(params.N)+1):
        # death of susceptible
        M[int(I*(params.N+1.)+S-1),int(I*(params.N+1.)+S)] = params.mu*S

    for I in np.arange(1,int(params.N)+1):
        # death of infected
        M[int((I-1)*(params.N+1.)+S),int(I*(params.N+1.)+S)] = params.gamma*I

    for I in np.arange(0,int(params.N)):
        # infection
        M[int((I+1)*(params.N+1.)+S-1),int(I*(params.N+1.)+S)] = (params.beta*S*I
                                                            /params.N)

    ## FILL IN I=0, No death of infected or infection
    I = 0

    for S in np.arange(0,int(params.N)+1):
        # birth susceptible
        M[int(I*(params.N+1.)+S+1),int(I*(params.N+1.)+S)] = params.mu*params.N

    for S in np.arange(1,int(params.N)+1):
        # death of susceptible
        M[int(I*(params.N+1.)+S-1),int(I*(params.N+1.)+S)] = params.mu*S

    ## FILL IN I=N, No infection
    I = params.N

    for S in np.arange(0, int(params.N)+1):
        # death of infected
        M[int((I-1)*(params.N+1.)+S),int(I*(params.N+1.)+S)] = params.gamma*I

    for S in np.arange(0,int(params.N)):
        # birth susceptible
        M[int(I*(params.N+1.)+S+1),int(I*(params.N+1.)+S)] = params.mu*params.N

    for S in np.arange(1,int(params.N)+1):
        # death of susceptible
        M[int(I*(params.N+1.)+S-1),int(I*(params.N+1.)+S)] = params.mu*S

    for i in np.arange(np.shape(M)[0]):
        M[i,i] = - np.sum(M[i,:])

    return M#np.transpose(M)

if __name__ == "__main__":
    params = Params()
    MATRIX = create_full_markov(params)
    print(MATRIX)
    eigenvals, eigenvecs = eig(MATRIX)
    idx = eigenvals.argsort()[-1]
    eigenval = eigenvals[idx]
    eigenvec = eigenvecs[:,idx]
    print(eigenval,eigenvec)

    eigenvals, eigenvecs = eig(MATRIX.T)
    idx = eigenvals.argsort()[-1]
    eigenval = eigenvals[idx]
    eigenvec = eigenvecs[:,idx]
    print(eigenval,eigenvec)
    #print(MATRIX)
    #print(expm(MATRIX))
    #MINV= inv(MATRIX)
