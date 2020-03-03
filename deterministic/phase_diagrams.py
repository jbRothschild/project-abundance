import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('parameters.mplstyle')

K = 100; alpha = 1.0; r = 10; mu = 0.25; numspecies = 10;
ss = K * ( 1 + np.sqrt( 1 + 4*( alpha*( numspecies-1 ) + 1  )*mu/(K*r) ) ) / ( 2*( alpha*( numspecies-1 ) + 1 ) )
print(ss)

def dndt(n1, n2):
    return r*n1*(1 - (n1 + alpha*( ss*(numspecies-2) + n2) )/K  ) + mu

def dndt_eigenvec_unstable(n,x):
    # n is how much each  most of the species are at, x is how much it differs for the non identical species.
    #unstable eigenvector is (+1,+1,.....,+1), stable (-1, 0, ... , 0, 1, 0, ... ,0)
    dndt_most_species = r*(numspecies-2)*n*( 1 - ( n + alpha*( n*(numspecies-1) ) )/K ) + (numspecies-2)*mu
    dndt_negative_species = r*(n-x)*( 1 - ( (n-x) + alpha*( n*(numspecies-2) + n + x ) )/K ) + mu
    dndt_positive_species = r*(n+x)*( 1 - ( (n+x) + alpha*( n*(numspecies-2) + n - x ) )/K ) + mu

    return dndt_most_species + dndt_negative_species + dndt_positive_species

def dndt_eigenvec_stable(n,x):
    #stable eigenvector is (-1,0,...,0,+1,0,..,0)
    dndt_negative_species = r*(n-x)*( 1 - ( (n-x) + alpha*( n*(numspecies-2) + n + x ) )/K ) + mu
    dndt_positive_species = r*(n+x)*( 1 - ( (n+x) + alpha*( n*(numspecies-2) + n - x ) )/K ) + mu
    return dndt_positive_species - dndt_negative_species

def dndt_eigenvec_unstable2(n, n1):
    # n is how much each species is at, x is how much it differs for the non identical species.
    #unstable eigenvector is (+1,+1,.....,+1) other is simple
    dndt_most_species = r*(numspecies-1)*n*( 1 - ( n + alpha*( n*(numspecies-2) + n1 ) )/K ) + (numspecies-1)*mu
    dndt_single_species = r*n1*( 1 - ( n1 + alpha*( n*(numspecies-1) ) )/K ) + mu

    return dndt_most_species + dndt_single_species

def fcn_simple(N,  t):
    n1, n2 = N
    return [dndt(n1,n2), dndt(n2,n1)]

def fcn_eigen(N, t):
    n1, n2 = N
    return [dndt_eigenvec_unstable(n1,n2), dndt_eigenvec_stable(n1,n2)]

def fcn_unstb_eigen(N, t):
    n1, n2 = N
    return [dndt_eigenvec_unstable2(n1,n2), dndt(n1,n2)]

def phase(n1, n2, dict_dir, xstring, ystring):
    t = 0

    N1, N2 = np.meshgrid(n1, n2)

    u, v = np.zeros(N1.shape), np.zeros(N2.shape)

    NI, NJ = N1.shape

    for i in range(NI):
        for j in range(NJ):
            x = N1[i,j]; y = N2[i,j]
            sol = dict_dir['function']([x,y], t)
            u[i,j] = sol[0]; v[i,j] = sol[1]

    #fix, ax =  plt.figure()
    if  dict_dir['name'] == 'simple':
        Q = plt.quiver(N1, N2, u, v, color='k')

    else:
        Q = plt.quiver(N1*numspecies, 2*N2, u, v, color='k')

    plt.xlabel(xstring); plt.ylabel(ystring); #plt.title(str(numspecies) + r' species, $\alpha = $' + str(alpha))
    plt.plot(dict_dir['steady_state']['x'],dict_dir['steady_state']['y'],'o')
    plt.ylim(bottom = np.min(N2), top = np.max(N2) ); plt.xlim(left = np.min(N1), right = np.max(N1) );

    if dict_dir['name'] == 'eigen' or dict_dir['name'] == 'unstb_eigen':
        plt.xlim( left = int(ss) , right = np.max(N1*numspecies) ); plt.ylim(bottom = np.min(2*N2), top = np.max(2*N2) )

    #plt.xlim([0,int(ss)+10]); plt.ylim([0,int(ss)+10])

    plt.savefig(os.getcwd() + os.sep + dict_dir['name'] + '_phase_diagram_' + str(numspecies) + 'species_alpha' + str(alpha) + '.pdf', transparent=True)
    #plt.savefig(os.getcwd() + os.sep + dict_dir['name'] + '_phase_diagram_' + str(numspecies) + 'species_alpha' + str(alpha) + '.eps')
    plt.close()

if __name__ == '__main__':
    num_points = 21
    dict_dir = {'simple' : {'name' : 'simple', 'function' : fcn_simple , 'steady_state': {'x' : ss, 'y' : ss}},
                'eigen' : {'name' : 'eigen', 'function' : fcn_eigen  , 'steady_state': {'x' : numspecies*ss, 'y' : 0}},
                'unstb_eigen' : {'name' : 'unstb_eigen', 'function' : fcn_unstb_eigen , 'steady_state': {'x' : numspecies*ss, 'y' : ss} }}
    phase(np.linspace(0, int(ss)+10, num_points),  np.linspace(0, int(ss)+10, num_points), dict_dir['simple'], r'$n_i$', r'$n_j$')
    phase(np.linspace(0, int(ss)+10, num_points),  np.linspace(-int(ss), int(ss), num_points), dict_dir['eigen'], r'$\sum n_i$', r'$n_j-n_k$')
    phase(np.linspace(0, int(ss)+10, num_points),   np.linspace(0, int(ss)+10, num_points), dict_dir['unstb_eigen'], r'$\sum n_i$', r'$n_i$')
