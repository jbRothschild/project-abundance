import os, glob, csv, pickle, copy, time, scipy

import matplotlib as mpl; from matplotlib import colors
import matplotlib.pyplot as plt; from matplotlib.lines import Line2D
import numpy as np; import pandas as pd
import scipy.io as sio;
from scipy.signal import savgol_filter, find_peaks, argrelextrema
from scipy.optimize import curve_fit

from scipy.fftpack import fft, fftfreq, ifft, ifftshift

def autocorrelation_dfft(times, trajs):
    """
    This function never worked... need to download correct packages
    """

    # for now do the first trajectory
    nodes = times; values = trajs[0]

    from pynfft import NFFT, Solver
    M = len(times)
    N = 128
    f     = np.empty(M,     dtype=np.complex128)
    f_hat = np.empty([N,N], dtype=np.complex128)

    this_nfft = NFFT(N=[N,N], M=M)
    this_nfft.x = np.array([[node_i,0.] for node_i in nodes])
    this_nfft.precompute()

    this_nfft.f = f
    ret2=this_nfft.adjoint()

    #print this_nfft.x # nodes in [-0.5, 0.5), float typed

    this_solver = Solver(this_nfft)
    this_solver.y = values          # '''right hand side, samples.'''

    #this_solver.f_hat_iter = f_hat # assign arbitrary init sol guess, default 0

    this_solver.before_loop()       # initialize solver internals

    while not np.all(this_solver.r_iter < 1e-2):
        this_solver.loop_one_step()

    # plotting
    fig=plt.figure(1,(20,5))
    ax =fig.add_subplot(111)

    foo=[ np.abs( this_solver.f_hat_iter[i][0])**2\
                                for i in range(len(this_solver.f_hat_iter) ) ]

    ax.plot( np.abs(np.arange(-N/2,+N/2,1)) )
    plt.show()

    return autocor, spectrum

def exponential_fit_autocorrelation(autocor, times, fracTime=1, plot=False):
    """

    Input
        autocor : array of autocorrelation of 1 species
    """
    def func(x, a, b):
        return a * np.exp(b * x)

    fracAutocor = autocor[:len(autocor)//fracTime]
    fracTimes = times[:len(autocor)//fracTime]

    # Here you give the initial parameters for a,b,c which Python then iterates
    # over to find the best fit
    guess = - np.log(fracAutocor[-1])/fracTimes[-1]
    popt, pcov = curve_fit(func,fracTimes,fracAutocor,p0=(1.0, guess))
    if plot:
        residuals = y - func(fracTimes,popt[0],popt[1])
        fres = sum( (residuals**2)/func(x,popt[0],popt[1]) ) # chi-sqaure of fit

        """ Now if you need to plot, perform the code below """
        curvey = func(fracTimes,popt[0],popt[1]) # This is your y axis fit-line

        plt.plot(fracTimes, curvey, 'red', label='The best-fit line')
        plt.scatter(x, fracTimes, c='b',label='The data points')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return popt[0], popt[1]

def average_timescale_autocorrelation( autocor, times, fracTime=1, plot=False ):
    """
    For many species, get mean and standard deviation of timescale associated
    with the autocorrelation function of each

    """
    timescale = []

    for i in np.arange(0, np.shape(autocor)[0]):
        a, b = exponential_fit_autocorrelation(autocor[i], times, fracTime, plot)
        #a*exp(bx)
        if b != 0.0:
            timescale.append( 1./b )
        else: timescale.append( 10**50 )

    return np.mean(timescale), np.std(timescale)

def sum_2_exponential_fit_autocorrelation(autocor):
    """

    Input
        autocor : array of autocorrelation of 1 species
    """

    return 0

def autocorrelation_spectrum( times, trajs ):
    """
    Takes times, converts to similar time step (using smallest increment).

    Return
        autocor         : (S x T) array time autocrrelation function. Averaged?
        spectrum        : (S x f) array of the frequency, Lorentzian generally
        specAutocor     : array size T time autocorrelation function nbr species
        specSpectrum    : array size T, spectrum for nbr species
    """
    minTimeStep = np.mean( np.diff(times) ) # min takes too long!

    # new means in the new time chosen.
    newTimes, newTrajs, newSpecies = [], [], [];

    currentTime = 0.0;
    for i, timepoint in enumerate(times):
        # set for this time
        abundTime = trajs[i].tolist();
        specTime = len([ x for x in trajs[i] if x > 0]) # nbr species present
        while currentTime <= timepoint:
            newTimes.append(currentTime); newTrajs.append(abundTime)
            newSpecies.append(specTime); currentTime += minTimeStep

    newTimes = np.array(newTimes); newSpecies = np.array(newSpecies);
    newTrajs = np.array(newTrajs)

    # species nbr autocorrelation and spectrum
    specSpectrum, specAutocor = autocorrelation( newSpecies )

    # for each species, abundance autocorrelation
    spectrum, autocor = [], [];
    for i in np.arange(np.shape(newTrajs)[1]):
        tempSpectrum, tempAutocor = autocorrelation( newTrajs[:,i] )
        spectrum.append(tempSpectrum); autocor.append(tempAutocor);

    return np.array(autocor), np.array(spectrum), specAutocor, specSpectrum,\
                newTimes

def autocorrelation(x):
    """
    https://stackoverflow.com/questions/47850760/using-scipy-fft-to-calculate-autocorrelation-of-a-signal-gives-different-answer
    Discrete FT assumes signals to be periodic. So in your fft based code you
    are computing a wrap-around autocorrelation. To avoid that you'd have to do
    some form of 0-padding
    """
    if np.std(x) == 0.0:
        freq = 10**(-13)*np.ones(len(x)//2); freq[1] = 1.0
        return freq, np.ones(len(x)//2)

    xp = ifftshift((x - np.average(x))/np.std(x)) # mean, stationarity
    n, = xp.shape
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]] # zero-padding
    f = fft(xp)
    p = np.absolute(f)**2
    pi = ifft(p)
    return np.real(p)[:n//2], np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)
