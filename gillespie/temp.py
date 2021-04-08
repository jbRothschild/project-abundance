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

def main():
    t = np.linspace(0,20,1024)
    x = np.exp(-t**2)
    pl.plot(t[:200], autocorrelation(x)[:200],label='scipy fft')
    pl.plot(t[:200], autocorrelation2(x)[:200],label='direct autocorrelation')
    pl.plot(t[:200], autocorrelation3(x)[:200],label='numpy correlate')
    pl.plot(t[:200], autocorrelation4(x)[:200],label='stack exchange')
    pl.yscale('log')
    pl.legend()
    pl.show()


if __name__=='__main__':
    main()
