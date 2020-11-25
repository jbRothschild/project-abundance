import os, glob, csv, pickle, copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from gillespie_models import RESULTS_DIR, MultiLV, SIR
import theory_equations as theqs
from settings import VAR_NAME_DICT, COLOURS, IMSHOW_KW, NPZ_SHORT_FILE

SIM_FIG_DIR = 'figures' + os.sep + 'simulations'
while not os.path.exists( os.getcwd() + os.sep + SIM_FIG_DIR ):
    os.makedirs(os.getcwd() + os.sep + SIM_FIG_DIR);

plt.style.use('custom.mplstyle')

def mlv_consolidate_sim_results(dir, parameter1='immi_rate'
                                    , parameter2='comp_overlap'):
    """
    Analyze how the results from different simulations differ for varying
    (parameter)

    Input :
        dir       : directory that we're plotting from
        parameter1 : the parameter that changes between different simulations
                    (string)
        parameter2 : second parameter that changes between different simulations
                    (string)
    """
    filename =  dir + os.sep + NPZ_SHORT_FILE

    # count number of subdirectories
    nbr_sims        = len( next( os.walk(dir) )[1] )
    # initialize the
    param1          = np.zeros(nbr_sims); param2            = np.zeros(nbr_sims)
    sim_dist_vary   = []                ; rich_dist_vary    = []
    conv_dist_vary  = []                ; mf_dist_vary      = []

    # TODO change to dictionary
    for i in np.arange(nbr_sims):
        with open(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
                   'results_0.pickle', 'rb') as handle:
            param_dict  = pickle.load(handle)

        model           = MultiLV(**param_dict)
        theory_model    = theqs.Model_MultiLVim(**param_dict)

        # distribution
        start = time.time()
        ss_dist_sim     = model.results['ss_distribution'] \
                                / np.sum(model.results['ss_distribution'])
        ss_dist_conv, _ = theory_model.abund_1spec_MSLV()
        ss_dist_mf, _   = theory_models.abund_sid()
        richness_dist   = model.results['richness']
        rich_dist_vary.append( np.array( richness_dist ) )
        sim_dist_vary.append( np.array( ss_dist_sim ) )
        conv_dist_vary.append( np.array( ss_dist_conv ) )
        mf_dist_vary.append( np.array( ss_dist_mf ) )
        end = time.time()
        hours, rem = divmod( end-start, 3600 )
        minutes, seconds = divmod( rem, 60 )
        print(">>{} Time elapsed : {:0>2}:{:0>2}:{:05.2f}".format(i, int(hours)
                                                    , int(minutes), seconds) )
        # Value of parameters
        param1[i] = param_dict[parameter1]; param2[i] = param_dict[parameter2]

    # making all sims have same distribution length
    len_longest_sim     = len( max(sim_dist_vary,key=len) )
    length_longest_rich = len( max(rich_dist_vary,key=len) )
    sim_dist            = np.zeros( ( nbr_sims,len_longest_sim ) )
    conv_dist           = np.zeros( ( nbr_sims,len_longest_sim ) )
    mf_dist             = np.zeros( ( nbr_sims,len_longest_sim ) )
    rich_dist           = np.zeros( ( nbr_sims,length_longest_rich ) )

    for i in np.arange(nbr_sims):
        conv_idx    = np.min( [ len(conv_dist_vary[i]), len_longest_sim ] )
        mf_idx      = np.min( [ len(mf_dist_vary[i]), len_longest_sim ] )
        sim_dist[i,:len(ss_dist_vary[i])]       = sim_dist_vary[i]
        conv_dist[i,:conv_idx]                  = conv_dist_vary[i,:conv_idx]]
        mf_dist[i,:mf_idx]                      = mf_dist_vary[i,:mf_idx]]
        rich_dist[i,:len(rich_dist_vary[i])]    = rich_dist_vary[i]

    # For heatmap stuff
    param1_2D = np.unique(param1); param2_2D = np.unique(param2)
    dim_1     = len(param1_2D)   ; dim_2      = len(param2_2D)

    # initialize
    mf_dist2D   = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    conv_dist2D = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    sim_dist2D  = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    rich_dist2D = np.zeros( ( dim_1,dim_2,length_longest_rich ) )

    # put into a 2d array all the previous results
    for sim in np.arange(nbr_sims):
        i                   = np.where( param1_2D==param1[sim] )[0][0]
        j                   = np.where( param2_2D==param2[sim] )[0][0]
        sim_dist2D[i,j]     = sim_dist[sim]
        mf_dist2D[i,j]      = mf_dist[sim]
        conv_dist2D[i,j]    = conv_dist[sim]
        rich_dist2D[i,j]    = rich_dist[sim]

    # arrange into a dictionary to save
    dict_arrays = {  parameter1 : param1_2D, parameter2     : param2_2D
                                            , 'sim_dist'    : sim_dist2D
                                            , 'mf_dist'     : mf_dist2D
                                            , 'conv_dist'   : conv_dist2D
                                            , 'rich_dist'   : rich_dist2D
                                            }
    # save results in a npz file
    np.savez(filename, **dict_arrays)

if __name__ == "__main__":
    sim_dir = RESULTS_DIR + os.sep + 'multiLV45'
    mlv_consolidate_sim_results( sim_dir )
