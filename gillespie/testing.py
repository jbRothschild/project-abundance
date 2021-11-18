import numpy as np
import matplotlib.pyplot as pl
from scipy.fftpack import fft, ifft, ifftshift

def autocorrelation(x) :
    xp = (x - np.average(x))/np.std(x)
    f = fft(xp)
    p = np.absolute(f)**2
    pi = ifft(p)
    return np.real(pi)[:len(xp)//2]/(len(xp))

def autocorrelation2(x):
    maxdelay = len(x)//5
    N = len(x)
    mean = np.average(x)
    var = np.var(x)
    xp = (x - mean)/np.sqrt(var)
    autocorrelation = np.zeros(maxdelay)
    for r in range(maxdelay):
        for k in range(N-r):
            autocorrelation[r] += xp[k]*xp[k+r]
        autocorrelation[r] /= float(N-r)
    return autocorrelation

def autocorrelation3(x):
    xp = (x - np.mean(x))/np.std(x)
    result = np.correlate(xp, xp, mode='full')
    return result[result.size//2:]//len(xp)

def autocorrelation4(x):
    #xp = x
    xp = ifftshift((x - np.average(x))/np.std(x))
    n, = xp.shape
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = fft(xp)
    p = np.absolute(f)**2
    pi = ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)

def propensities():
    current_state = np.array([50,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    immi_rate = 0.001
    comp_overlap = 1.0
    birth_rate = 2.0
    death_rate = 1.0
    carry_capacity = 100
    quadratic = 0.0
    emmi_rate = 0.0

    prop = np.zeros( len(current_state)*2 )
    prop2 = np.zeros( len(current_state)*2 )

    prop[::2] = ( immi_rate + current_state * ( birth_rate ) )
                    #- ( quadratic *
                    #( birth_rate - death_rate ) *
                    #(1.0 - comp_overlap ) * current_state +
                    #( comp_overlap * np.sum(current_state) ) )
                    #/ carry_capacity ) )
                    # birth + immigration
    prop[1::2] = ( current_state * ( death_rate #+ emmi_rate
                      #+ ( birth_rate - death_rate )*( 1.0
                      #- quadratic )*( (1.0 - comp_overlap ) * current_state
                      + ( birth_rate - death_rate )
                      * ( (1.0 - comp_overlap ) * current_state
                      + comp_overlap * np.sum(current_state) ) / carry_capacity ) )
                      # death + emmigration

    for i in np.arange(0,len(current_state)):
            prop2[i*2] = ( current_state[i] * ( birth_rate
                        - quadratic * ( current_state[i]
                        + comp_overlap*np.sum(
                        np.delete(current_state,i)))/carry_capacity )
                        + immi_rate)
                        # birth + immigration
            prop2[i*2+1] = (current_state[i] * ( death_rate + emmi_rate
                          + ( birth_rate - death_rate )*( 1.0
                          - quadratic )*(current_state[i]
                          + comp_overlap*np.sum(
                          np.delete(current_state,i)))/carry_capacity ) )
                          # death + emmigration

    #print(prop==prop2)
    return 0


def main():
    """
    t = np.linspace(0,20,1024)
    x = np.exp(-t**2)
    pl.plot(t[:200], autocorrelation(x)[:200],label='scipy fft')
    pl.plot(t[:200], autocorrelation2(x)[:200],label='direct autocorrelation')
    pl.plot(t[:200], autocorrelation3(x)[:200],label='numpy correlate')
    pl.plot(t[:200], autocorrelation4(x)[:200],label='stack exchange')
    pl.yscale('log')
    pl.legend()
    pl.show()
    """
    #propensities()
    ss = np.zeros(30)
    st = np.zeros(30)
    a = np.array([0,0,0,0,10,2,3,3,0])
    unique, counts = np.unique(a,return_counts=True)
    ss[unique.astype(int)] += 0.33*counts
    print(ss)

    for i in a:
        st[i] += 0.33
    print(ss)

if __name__=='__main__':
    main()
