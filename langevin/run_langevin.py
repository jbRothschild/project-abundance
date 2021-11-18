import argparse, pathlib, os
import numpy as np

from src.force_diffusion_functions import lotkvol_force, mehta_diff, master_diff
from src.euler_scheme import NumericalLangevin
from src.initialize import initialize_pop
from src.save_simulation import save_sim
from default import DATA_FOLDER

# TODO Change functions in here to functions that I can select for automatically
# kind of like a function factory (save_sim, mehta_diff)

class StoreDictKeyPair(argparse.Action):
    """
    Allows one to store from command line dictionary keys and values
    using 'KEY=VAL' after the -p key
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest,
              nargs=nargs, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split('=')
            if k == 'sim_dir':
                my_dict[k] = str(v)
            else:
                my_dict[k] = float(v)
        setattr(namespace, self.dest, my_dict)

if __name__ == '__main__':

    np.random.seed( ) # set random seed

    # get commandline arguments
    parser = argparse.ArgumentParser(description = "Simulating Langevin Eqn.")

    param = {'carryCapacity' : 100.0, 'birthRate'   : 2.0
                                    , 'deathRate'   : 1.0
                                    , 'compOverlap' : 0.1
                                    , 'immiRate'    : 0.001
                                    , 'nbrSpecies'  : 30
                                    }

    parser.add_argument('-p', '--params', metavar='KEY=VAL'
                        , default=param
                        , dest='my_dict', nargs='*'
                        , action=StoreDictKeyPair
                        , required=False
                        , help='Additional parameters to be passed on for the \
                        simulation')
    parser.add_argument('-t', '--nbrSteps', type = int, default = 10**(6)
                        , nargs = '?', help = "Number of time steps")
    parser.add_argument('-dt', '--timeStep', type = float, default = 10**(-3)
                        , nargs = '?', help = "Time step")
    parser.add_argument('-d', '--directory', type = str, default = 'default'
                        , nargs = '?', help = "Directory to save sim in.")
    parser.add_argument('-s','--save', dest='save', action='store_true'
                        , default=False, required=False, help = "Use to save.")
    parser.add_argument('-id', '--identification', type = int, default = 1
                        , nargs = '?', help = "Identification of simulation.")
    parser.add_argument('-savet','--saveTrajectory', dest='saveTraj'
                        , action='store_true', default=False
                        , required=False, help = "Use save whole trajetcory.")

    args = parser.parse_args()
    dir = DATA_FOLDER + os.sep + args.directory
    path = pathlib.Path( dir )
    path.mkdir(parents=True, exist_ok=True)

    for key in args.my_dict: # TODO : must be a more pythonic way to do this
        param[key] = args.my_dict[key]

    timeStep = args.timeStep
    nbrSteps = args.nbrSteps

    population = initialize_pop( param )
    fcn_force = lotkvol_force( param ); fcn_diff = master_diff( param )

    simulation = NumericalLangevin(fcn_force=fcn_force, fcn_diff=fcn_diff
                                        , population=population)

    traj = simulation.euler_scheme(nbrSteps, timeStep)

    save_sim( param, traj, args.saveTraj )

    file = dir + os.sep + 'sim' + str(args.identification) + "_results"
    if args.save:
        np.save( file , param, allow_pickle=True)
