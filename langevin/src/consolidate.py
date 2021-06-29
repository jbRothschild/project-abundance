import sys, os
import numpy as np

def make_ordered_2Darray( dir, parameter1, parameter2 ):
    # TODO : Some things are hardcoded in here, such as 'countsNorm' and
    # 'binCentres'. Need to find a better way to not hardcode it.

    # check number of files with relative path to directory
    absPath = os.getcwd() + os.sep + dir
    nbrSims  = len( next( os.walk(absPath) )[2] )

    # initialize arrays
    param1 = np.zeros(nbrSims); param2 = np.zeros(nbrSims)
    countsNorm = []; binCentres = []

    # fill arrays with data
    for i in np.arange(1,nbrSims+1):
        filename = 'sim' + str(i) + '_results.npy'
        if not os.path.exists( dir + os.sep + filename ):
            sys.exit("Missing simulation number : " + str(i) )
        else:
            results = np.load(dir + os.sep + filename, allow_pickle=True)[()]
            # fill
            param1[i-1] = results[parameter1]; param2[i-1] = results[parameter2]
            # TODO : Hardcoded not good
            countsNorm.append( results['countsNorm'] )
            binCentres.append( results['binCentres'] )

    # order the arrays now by param1 and param 2
    longestSim = len( max(countsNorm,key=len) )
    arrParam1 = np.unique(param1); arrParam2 = np.unique(param2)

    # 2D arrays with the correct placements for heatmap
    arrCountsNorm = np.zeros( ( len(arrParam1),len(arrParam2),longestSim ) )
    arrBinCentres = np.zeros( ( len(arrParam1),len(arrParam2),longestSim ) )

    # filling 2D arrays
    for sim in np.arange(nbrSims):
        i                   = np.where( arrParam1==param1[sim] )[0][0]
        j                   = np.where( arrParam2==param2[sim] )[0][0]
        arrCountsNorm[i,j,:len(countsNorm[sim])]  = countsNorm[sim]
        arrBinCentres[i,j,:len(binCentres[sim])]  = binCentres[sim]

    results[parameter1]     = arrParam1
    results[parameter2]     = arrParam2
    results['countsNorm']   = arrCountsNorm
    results['binCentres']   = arrBinCentres

    return results

def other_consolidate():

    return 0
