import numpy as np
import matplotlib.pyplot as plt
import argparse, os, pathlib
from scipy.signal import savgol_filter

from default import DATA_FOLDER

def plot_abundance(binCentres, countsNorm):
    print(binCentres)
    plt.hist(binCentres, weights=countsNorm, bins=len(binCentres))
    windowSize = 4*int(len(binCentres)/10)+3;
    polynomial = 5
    smoothedFcn = savgol_filter(countsNorm, windowSize, polynomial)
    plt.plot(binCentres, smoothedFcn, 'r')
    #sns.histplot(trajectory, bins=nbins, stat='probability', kde=True)
    plt.yscale('log'); #plt.xlim((0.0,trajMax))
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        description = "Plotting results from a simulation.")
    parser.add_argument('-d', '--directory', type = str, default = 'default'
                        , nargs = '?', help = "Directory to save sim in.")
    parser.add_argument('--id', help='identification')
    parser.add_argument('-s','--save', dest='save', action='store_true'
                        , default=False, required=False, help = "Use to save.")

    args = parser.parse_args()

    dir = DATA_FOLDER + os.sep + args.directory
    path = pathlib.Path( dir )

    if args.id is not None:
        file = dir + os.sep + 'sim' + str(args.id) + "_results" + '.npy'
        param = np.load(file, allow_pickle=True)[()]
        plot_abundance(param['binCentres'], param['countsNorm'])
