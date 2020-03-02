import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('parameters.mplstyle')

K = 100; alpha = 1.0; r = 10; mu = 0.5; numspecies = 2;
ss = K * ( 1 + np.sqrt( 1 + 4*( alpha*( numspecies-1 ) + 1  )*mu/(K*r) ) ) / ( 2*( alpha*( numspecies-1 ) + 1 ) )

def dndt(n1, n2):
    return r*n1*(1 - (n1 + alpha*( ss*(numspecies-2) + n2) )/K  ) + mu

print(dndt(ss,ss))

def function(N,  t):
    n1, n2 = N
    return [dndt(n1,n2), dndt(n2,n1)]

n1 = np.linspace(0, int(ss)+10, int(ss)+11); n2 = np.linspace(0, int(ss)+10, int(ss)+11)

t = 0

N1, N2 = np.meshgrid(n1, n2)

u, v = np.zeros(N1.shape), np.zeros(N2.shape)

NI, NJ = N1.shape

for i in range(NI):
    for j in range(NJ):
        x = N1[i,j]; y = N2[i,j]
        sol = function([x,y], t)
        u[i,j] = sol[0]; v[i,j] = sol[1]

Q = plt.quiver(N1, N2, u, v, color='k')

plt.xlabel(r'$n_1$'); plt.ylabel(r'$n_2$'); plt.title(str(numspecies) + r' species, $\alpha =$' + str(alpha))
plt.plot([ss],[ss],'o')
#plt.xlim([0,int(ss)+10]); plt.ylim([0,int(ss)+10])

plt.savefig(os.getcwd() + os.sep + 'phase_diagram_' + str(numspecies) + 'species' + '.pdf', transparent=True)
plt.savefig(os.getcwd() + os.sep + 'phase_diagram_' + str(numspecies) + 'species' + '.eps')
