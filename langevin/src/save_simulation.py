import numpy as np

def abundance( param, trajectory ):
    trajMax = np.max(trajectory);trajMin=np.min(trajectory)
    nbins = int((trajMax-trajMin)*param['carryCapacity'])
    counts, binEdges = np.histogram(trajectory, bins=nbins)
    #counts, binsEdges, _ = plt.hist(x=trajectory, bins=nbins, alpha=0.7
    #                                            , rwidth=1.0)
    countsNorm = np.array( counts/np.sum(counts) )
    binCentres = np.array( (binEdges[:-1] + binEdges[1:])/2. )
    return binCentres, countsNorm

def save_sim( param, traj ):
    binCentres, countsNorm = abundance( param, traj )

    param['countsNorm'] = countsNorm
    param['binCentres'] = binCentres

    return param
