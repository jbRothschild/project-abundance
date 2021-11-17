import os, pickle, time

import numpy as np
import src.equations as eq; import src.theory_equations as theqs
from src.gillespie_models import MultiLV
from src.autocorrelation import average_timescale_autocorrelation, exponential_fit_autocorrelation, autocorrelation_spectrum
from src.settings import VAR_NAME_DICT, COLOURS, IMSHOW_KW, NPZ_SHORT_FILE\
                    , VAR_SYM_DICT
# These are incredibly shitty functions that should be redone completely...

def average_ni_given_nj( conditional ):
    """
    for vector n_j, give the average of another species.
    """
    pop = conditional.shape[0]
    avNi_Nj = np.zeros(pop)
    for i in np.arange(0,pop):
        avNi_Nj[i] = np.dot(np.arange(0,pop),conditional[i,:])

    return avNi_Nj


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
    mf3_dist_vary   = []
    av_ni_given_nj_vary = []
    coeff_ni_nj     = np.zeros(nbr_sims); corr_ni_nj        = np.zeros(nbr_sims)
    coeff_J_n       = np.zeros(nbr_sims); corr_J_n          = np.zeros(nbr_sims)
    coeff_Jminusn_n = np.zeros(nbr_sims); corr_Jminusn_n    = np.zeros(nbr_sims)
    coeff_ni_nj_S     = np.zeros(nbr_sims); corr_ni_nj_S     = np.zeros(nbr_sims)
    coeff_J_n_S       = np.zeros(nbr_sims); corr_J_n_S       = np.zeros(nbr_sims)
    coeff_Jminusn_n_S = np.zeros(nbr_sims); corr_Jminusn_n_S = np.zeros(nbr_sims)
    # TEMP TIMES
    time_autocor_spec = np.zeros( nbr_sims )
    mean_time_autocor_abund = np.zeros( nbr_sims )
    std_time_autocor_abund  = np.zeros( nbr_sims )
    dominance_turnover      = np.zeros( nbr_sims )
    suppress_turnover       = np.zeros( nbr_sims )
    dominance_return        = np.zeros( nbr_sims )
    suppress_return         = np.zeros( nbr_sims )

    # TODO change to dictionary
    for i in np.arange(nbr_sims):
        sim_nbr = i + 1
        if not os.path.exists(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
                   'results_0.pickle'):

            print( "Missing simulation: " + str(sim_nbr) )
            rich_dist_vary.append( np.array( [0] ) )
            sim_dist_vary.append( np.array( [0] ) )
            #conv_dist_vary.append( np.array( ss_dist_conv ) )
            mf_dist_vary.append( np.array( [0] ) )
            mf3_dist_vary.append( np.array( [0] ) )

        else:

            with open(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
                       'results_0.pickle', 'rb') as handle:
                param_dict  = pickle.load(handle)

            model           = MultiLV(**param_dict)
            theory_model    = theqs.Model_MultiLVim(**param_dict)

            # distribution
            start = time.time()
            ss_dist_sim     = model.results['ss_distribution'] \
                                    / np.sum(model.results['ss_distribution'])
            #ss_dist_conv, _ = theory_model.abund_1spec_MSLV()
            ss_dist_mf, _   = theory_model.abund_sid()
            ss_dist_mf3, _  = theory_model.abund_sid_J()
            richness_dist   = model.results['richness']
            rich_dist_vary.append( np.array( richness_dist ) )
            sim_dist_vary.append( np.array( ss_dist_sim ) )
            #conv_dist_vary.append( np.array( ss_dist_conv ) )
            mf_dist_vary.append( np.array( ss_dist_mf ) )
            mf3_dist_vary.append( np.array( ss_dist_mf3 ) )

            # TEMP FIX FOR SOME WRONG COEFFICIENT OF VARIATION
            #correlations_fix(model, dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
            #           'results_0.pickle')

            # TEMP FIX FOR SOMETHING WRONG CONDITIONAL MLV71
            conditional = model.results['conditional']
            #for j in np.arange(0,conditional.shape[0]):
            #    conditional[j,j] *= 2
            #    if np.sum(conditional[j][:]) != 0.0:
            #        conditional[j,:] /= np.sum(conditional[j,:])

            av_ni_given_nj = average_ni_given_nj( conditional )
            av_ni_given_nj_vary.append( av_ni_given_nj )

            corr_ni_nj[sim_nbr-1] = model.results['corr_ni_nj']
            coeff_ni_nj[sim_nbr-1] = model.results['coeff_ni_nj']
            corr_J_n[sim_nbr-1] = model.results['corr_J_n']
            coeff_J_n[sim_nbr-1] = model.results['coeff_J_n']
            corr_Jminusn_n[sim_nbr-1] = model.results['corr_Jminusn_n']
            coeff_Jminusn_n[sim_nbr-1] = model.results['coeff_Jminusn_n']
            corr_ni_nj_S[sim_nbr-1] = model.results['corr_ni_nj_S']
            coeff_ni_nj_S[sim_nbr-1] = model.results['coeff_ni_nj_S']
            corr_J_n_S[sim_nbr-1] = model.results['corr_J_n_S']
            coeff_J_n_S[sim_nbr-1] = model.results['coeff_J_n_S']
            corr_Jminusn_n_S[sim_nbr-1] = model.results['corr_Jminusn_n_S']
            coeff_Jminusn_n_S[sim_nbr-1] = model.results['coeff_Jminusn_n_S']

            ############################################################################################################################
            # TEMP AUTOCORRELATION TIME SAVING. This is an aweful way of doing it. Fix it.
            ############################################################################################################################
            n=2; fracTime=100
            """
            autocor, _, specAutocor, _, newTimes =\
                     autocorrelation_spectrum(model.results['times'][n:],\
                    model.results['trajectory'][n:])

            _, time_autocor_spec[sim_nbr-1] = exponential_fit_autocorrelation(specAutocor, newTimes, fracTime)
            mean_time_autocor_abund[sim_nbr-1], std_time_autocor_abund[sim_nbr-1] =\
                    average_timescale_autocorrelation( autocor, newTimes, fracTime)
            """

            S = model.nbr_species; K = model.carry_capacity
            mu = model.immi_rate; rho = model.comp_overlap
            rplus = model.birth_rate; rminus = model.death_rate
            nbr_species = int( S*(1.0-ss_dist_sim[0]) )
            nbr = int( eq.deterministic_mean(nbr_species, mu, rho, rplus, rminus, K) )

            dominance_turnover[sim_nbr-1] = eq.mfpt_a2a(ss_dist_sim, nbr, mu, rplus, rminus, K, rho, S)
            suppress_turnover[sim_nbr-1]  = eq.mfpt_020(ss_dist_sim, mu)
            dominance_return[sim_nbr-1]   = eq.mfpt_a2b(ss_dist_sim, 0, nbr, mu, rplus)
            suppress_return[sim_nbr-1]    = eq.mfpt_b2a(ss_dist_sim, 0, nbr, mu, rplus)

            ############################################################################################################################
            # TEMP AUTOCORRELATION TIME SAVING. This is an aweful way of doing it. Fix it.
            ############################################################################################################################

            end = time.time()
            hours, rem = divmod( end-start, 3600 )
            minutes, seconds = divmod( rem, 60 )
            print(">>{}: Time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(i,int(hours)
                                                        , int(minutes),seconds))
            # Value of parameters
            param1[i] = param_dict[parameter1]
            param2[i] = param_dict[parameter2]

    # making all sims have same distribution length
    len_longest_sim     = len( max(sim_dist_vary,key=len) )
    length_longest_rich = len( max(rich_dist_vary,key=len) )
    sim_dist            = np.zeros( ( nbr_sims,len_longest_sim ) )
    conv_dist           = np.zeros( ( nbr_sims,len_longest_sim ) )
    mf_dist             = np.zeros( ( nbr_sims,len_longest_sim ) )
    mf3_dist            = np.zeros( ( nbr_sims,len_longest_sim ) )
    rich_dist           = np.zeros( ( nbr_sims,length_longest_rich ) )
    av_ni_given_nj      = np.zeros( ( nbr_sims,len_longest_sim ) )

    for i in np.arange(nbr_sims):
        #conv_idx    = np.min( [ len(conv_dist_vary[i]), len_longest_sim ] )
        mf_idx      = np.min( [ len(mf_dist_vary[i]), len_longest_sim ] )
        mf3_idx     = np.min( [ len(mf3_dist_vary[i]), len_longest_sim ] )
        sim_dist[i,:len(sim_dist_vary[i])]      = sim_dist_vary[i]
        #conv_dist[i,:conv_idx]                  = conv_dist_vary[i][conv_idx]
        mf_dist[i,:mf_idx]                      = mf_dist_vary[i][:mf_idx]
        mf3_dist[i,:mf_idx]                     = mf3_dist_vary[i][:mf_idx]
        rich_dist[i,:len(rich_dist_vary[i])]    = rich_dist_vary[i]
        av_ni_given_nj[i,:len(av_ni_given_nj_vary[i])] = av_ni_given_nj_vary[i]


    # For heatmap stuff
    param1_2D = np.unique(param1); param2_2D = np.unique(param2)
    dim_1     = len(param1_2D)   ; dim_2      = len(param2_2D)

    # initialize
    mf_dist2D           = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    mf3_dist2D          = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    conv_dist2D         = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    sim_dist2D          = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    av_ni_given_nj2D    = np.zeros( ( dim_1,dim_2,len_longest_sim ) )
    rich_dist2D         = np.zeros( ( dim_1,dim_2,length_longest_rich ) )
    corr_ni_nj2D        = np.zeros( ( dim_1,dim_2 ) )
    coeff_ni_nj2D       = np.zeros( ( dim_1,dim_2 ) )
    corr_J_n2D          = np.zeros( ( dim_1,dim_2 ) )
    coeff_J_n2D         = np.zeros( ( dim_1,dim_2 ) )
    corr_Jminusn_n2D    = np.zeros( ( dim_1,dim_2 ) )
    coeff_Jminusn_n2D   = np.zeros( ( dim_1,dim_2 ) )

    time_autocor_spec2D       = np.zeros( ( dim_1,dim_2 ) )
    mean_time_autocor_abund2D = np.zeros( ( dim_1,dim_2 ) )
    std_time_autocor_abund2D  = np.zeros( ( dim_1,dim_2 ) )
    dominance_turnover2D      = np.zeros( ( dim_1,dim_2 ) )
    suppress_turnover2D       = np.zeros( ( dim_1,dim_2 ) )
    dominance_return2D        = np.zeros( ( dim_1,dim_2 ) )
    suppress_return2D         = np.zeros( ( dim_1,dim_2 ) )

    # put into a 2d array all the previous results
    for sim in np.arange(nbr_sims):
        i                       = np.where( param1_2D==param1[sim] )[0][0]
        j                       = np.where( param2_2D==param2[sim] )[0][0]
        sim_dist2D[i,j]         = sim_dist[sim]
        mf_dist2D[i,j]          = mf_dist[sim]
        mf3_dist2D[i,j]         = mf3_dist[sim]
        av_ni_given_nj2D[i,j]   = av_ni_given_nj[sim]
        #conv_dist2D[i,j]       = conv_dist[sim]
        rich_dist2D[i,j]        = rich_dist[sim]
        corr_ni_nj2D[i,j]       = corr_ni_nj[sim]
        coeff_ni_nj2D[i,j]      = coeff_ni_nj[sim]
        corr_J_n2D[i,j]         = corr_J_n[sim]
        coeff_J_n2D[i,j]        = coeff_J_n[sim]
        corr_Jminusn_n2D[i,j]   = corr_Jminusn_n[sim]
        coeff_Jminusn_n2D[i,j]  = coeff_Jminusn_n[sim]

        #----------------------------------------------------------------------------------------------------------------------------
        # TEMP AUTOCORRELATION TIME SAVING. This is an aweful way of doing it. Fix it.
        #----------------------------------------------------------------------------------------------------------------------------

        time_autocor_spec2D[i,j]        = time_autocor_spec[sim]
        mean_time_autocor_abund2D[i,j]  = mean_time_autocor_abund[sim]
        std_time_autocor_abund2D[i,j]   = std_time_autocor_abund[sim]
        dominance_turnover2D[i,j]       = dominance_turnover[sim]
        suppress_turnover2D[i,j]        = suppress_turnover[sim]
        dominance_return2D[i,j]         = dominance_return[sim]
        suppress_return2D[i,j]          = suppress_return[sim]

        #----------------------------------------------------------------------------------------------------------------------------
        # TEMP AUTOCORRELATION TIME SAVING. This is an aweful way of doing it. Fix it.
        #----------------------------------------------------------------------------------------------------------------------------

    # arrange into a dictionary to save
    dict_arrays = { 'sim_dist'  : sim_dist2D, 'mf_dist'     : mf_dist2D
                                        , 'mf3_dist'        : mf3_dist2D
                                        , 'conv_dist'       : conv_dist2D
                                        , 'rich_dist'       : rich_dist2D
                                        , 'rich_dist2D'     : rich_dist
                                        , 'corr_ni_nj2D'    : corr_ni_nj2D
                                        , 'coeff_ni_nj2D'   : coeff_ni_nj2D
                                        , 'corr_J_n2D'      : corr_J_n2D
                                        , 'coeff_J_n2D'     : coeff_J_n2D
                                        , 'corr_Jminusn_n2D' : corr_Jminusn_n2D
                                        , 'coeff_Jminusn_n2D': coeff_Jminusn_n2D
                                        , 'carry_capacity': model.carry_capacity
                                        , 'birth_rate'      : model.birth_rate
                                        , 'death_rate'      : model.death_rate
                                        , 'nbr_species'     : model.nbr_species
                                        , 'immi_rate'       : model.immi_rate
                                        , 'comp_overlap'    : model.comp_overlap
                                        , 'time_autocor_spec2D'       : time_autocor_spec2D
                                        , 'mean_time_autocor_abund2D' : mean_time_autocor_abund2D
                                        , 'std_time_autocor_abund2D'  : std_time_autocor_abund2D
                                        , 'dominance_turnover2D'      : dominance_turnover2D
                                        , 'suppress_turnover2D'       : suppress_turnover2D
                                        , 'dominance_return2D'        : dominance_return2D
                                        , 'suppress_return2D'         : suppress_return2D
                                        , 'av_ni_given_nj2D'          : av_ni_given_nj2D
                                        }
    dict_arrays[parameter1] = param1_2D
    dict_arrays[parameter2] = param2_2D
    # save results in a npz file
    np.savez(filename, **dict_arrays)

    return filename, dict_arrays

def mlv_consolidate_sim_results_testing(dir, parameter1='immi_rate'
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

    with open(dir + os.sep + 'sim1' + os.sep
                    + 'results_0.pickle', 'rb') as handle:
        param_dict  = pickle.load(handle)

    model = MultiLV(**param_dict)
    dict_array_flat = {}
    dict_array_2D = {}

    entries_to_remove = {'time_btwn_ext'}

    for k in entries_to_remove:
        (model.results).pop(k, None)

    list_results_keys = (model.results).keys()

    for key in list_results_keys:
        dict_array_flat[key] = []

    # Additional computations
    dict_array_flat['sim_dist'] = []; dict_array_flat['rich_dist'] = []
    dict_array_flat['mf3_dist'] = []; dict_array_flat['mf_dist'] = []
    dict_array_flat['conv_dist'] = []; dict_array_flat['av_ni_given_nj'] = []

    dict_array_flat['time_autocor_spec'] = []
    dict_array_flat['mean_time_autocor_abund'] = []
    dict_array_flat['std_time_autocor_abund'] = []
    dict_array_flat['dominance_turnover'] = []
    dict_array_flat['suppress_turnover'] = []
    dict_array_flat['dominance_return'] = []
    dict_array_flat['suppress_return'] = []

    # count number of subdirectories
    nbr_sims = len( next( os.walk(dir) )[1] )
    param1 = np.zeros(nbr_sims)
    param2 = np.zeros(nbr_sims)

    #nbr_sims=100
    # TODO change to dictionary
    for i in np.arange(nbr_sims):
        sim_nbr = i + 1
        with open(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
                   'results_0.pickle', 'rb') as handle:
            param_dict  = pickle.load(handle)

        model           = MultiLV(**param_dict)
        theory_model    = theqs.Model_MultiLVim(**param_dict)
        param1[i] = param_dict[parameter1]
        param2[i] = param_dict[parameter2]

        for key in list_results_keys:
            dict_array_flat[key].append(model.results[key])

        # distribution
        start = time.time()
        #print(model.results['ss_distribution'])
        ss_dist_sim     = model.results['ss_distribution'] \
                                / np.sum(model.results['ss_distribution'])
        #ss_dist_conv, _ = theory_model.abund_1spec_MSLV()
        ss_dist_mf, _   = theory_model.abund_sid()
        ss_dist_mf3, _  = theory_model.abund_sid_J()
        dict_array_flat['rich_dist'].append(model.results['richness'])
        dict_array_flat['sim_dist'].append( np.array( ss_dist_sim ) )
        #conv_dist_vadict_array_flat['rich_dist'] = []ry.append( np.array( ss_dist_conv ) )
        dict_array_flat['mf_dist'].append( np.array( ss_dist_mf ) )
        dict_array_flat['mf3_dist'].append( np.array( ss_dist_mf3 ) )

        conditional = model.results['conditional']
        av_ni_given_nj = average_ni_given_nj( conditional )
        dict_array_flat['av_ni_given_nj'].append( av_ni_given_nj )

        S = model.nbr_species; K = model.carry_capacity
        mu = model.immi_rate; rho = model.comp_overlap
        rplus = model.birth_rate; rminus = model.death_rate
        #print(ss_dist_sim)
        nbr_species = int( S*(1.0-ss_dist_sim[0]) )

        nbr = int( eq.deterministic_mean(nbr_species, mu, rho, rplus, rminus, K) )
        dict_array_flat['dominance_turnover'].append( eq.mfpt_a2a(ss_dist_sim, nbr, mu, rplus, rminus, K, rho, S))
        dict_array_flat['suppress_turnover'].append( eq.mfpt_020(ss_dist_sim, mu) )
        dict_array_flat['dominance_return'].append( eq.mfpt_a2b(ss_dist_sim, 0, nbr, mu, rplus) )
        dict_array_flat['suppress_return'].append( eq.mfpt_b2a(ss_dist_sim, 0, nbr, mu, rplus) )

        """ This previous code needs to be fixed.
        # TEMP FIX FOR SOME WRONG COEFFICIENT OF VARIATION
        correlations_fix(model, dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
                  'results_0.pickle')

        # TEMP FIX FOR SOMETHING WRONG CONDITIONAL MLV71
        conditional = model.results['conditional']
        for j in np.arange(0,conditional.shape[0]):
            conditional[j,j] *= 2
            if np.sum(conditional[j][:]) != 0.0:
                conditional[j,:] /= np.sum(conditional[j,:])

        # TEMP AUTOCORRELATION TIME SAVING. This is an aweful way of doing it. Fix it.

        n=2; fracTime=100

        autocor, _, specAutocor, _, newTimes =\
                 autocorrelation_spectrum(model.results['times'][n:],\
                model.results['trajectory'][n:])

        _, time_autocor_spec[sim_nbr-1] = exponential_fit_autocorrelation(specAutocor, newTimes, fracTime)
        mean_time_autocor_abund[sim_nbr-1], std_time_autocor_abund[sim_nbr-1] =\
                average_timescale_autocorrelation( autocor, newTimes, fracTime)
                """
        end = time.time()
        hours, rem = divmod( end-start, 3600 )
        minutes, seconds = divmod( rem, 60 )
        print(">>{}: Time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(i,int(hours)
                                                    , int(minutes),seconds))

    # For heatmap stuff
    param1_2D = np.unique(param1); param2_2D = np.unique(param2)
    dim1     = len(param1_2D)   ; dim2      = len(param2_2D)

    for keys in dict_array_flat.keys():
        if dict_array_flat[keys] == []:
            pass
        elif not hasattr(dict_array_flat[keys][0],'__len__'):
            dict_array_2D[keys] = np.zeros( (dim1,dim2) )
            print(keys,'here')
        elif len(np.shape(dict_array_flat[keys][0])) < 2:
            # 2 dimensional properties (joint dist) ignored for now
            max_len = len( max(dict_array_flat[keys], key=len) )
            dict_array_2D[keys] = np.zeros( (dim1,dim2,max_len) )
            print(keys,'done')
        else:
            pass

    # put into a 2d array all the previous results
    for key in (dict_array_2D).keys():
        if not hasattr(dict_array_flat[key][0],'__len__'):
            for sim in np.arange(nbr_sims):
                i = np.where( param1_2D==param1[sim] )[0][0]
                j = np.where( param2_2D==param2[sim] )[0][0]
                dict_array_2D[key][i,j] = dict_array_flat[key][sim]
        else:
            for sim in np.arange(nbr_sims):
                i = np.where( param1_2D==param1[sim] )[0][0]
                j = np.where( param2_2D==param2[sim] )[0][0]
                dict_array_2D[key][i,j,:len(dict_array_flat[key][sim])] \
                                                = dict_array_flat[key][sim]

    dict_array_2D['carry_capacity'] = model.carry_capacity
    dict_array_2D['birth_rate'] = model.birth_rate
    dict_array_2D['death_rate'] = model.death_rate
    dict_array_2D['nbr_species'] = model.nbr_species
    dict_array_2D['immi_rate'] = model.immi_rate
    dict_array_2D['comp_overlap'] = model.comp_overlap
    dict_array_2D[parameter1] = param1_2D
    dict_array_2D[parameter2] = param2_2D
    # save results in a npz file
    np.savez(filename, **dict_array_2D)

    return filename, dict_array_2D

def mlv_multiple_folder_consolidate(list_dir, consol_name_dir, parameter1=None
                                            , parameter2=None, consol = False):
    """
    Takes simulations from multiple folders and combines them by averaging their
    results all together. The only reason I do this is to average the
    distributions... no need to average other quantities for now

    Input
        list_dir        : list of directories to average
        consol_name_dir : What to name the directory of averaged folders
        parameter1      : 1st paramater to vary (generally )
        parameter2      : 2nd parameter to vary
        consol          : Whether we want to use previously calculated short.npz
    """
    dict_arr = []
    for dir in list_dir:
        # Get dictionary of results from each of these sets of simualtions
        if not consol:
            _, dict_temp = mlv_consolidate_sim_results(dir, parameter1
                                                        , parameter2)
        else:
            dict_temp = dict( np.load(dir + os.sep + NPZ_SHORT_FILE) )
            dict_temp['carry_capacity'] = 100.0; dict_temp['birth_rate'] = 2.0;
            dict_temp['death_rate'] = 1.0; dict_temp['nbr_species'] = 30
        dict_arr.append(dict_temp)
        del dict_temp

    df = pd.DataFrame(dict_arr); mean_dict = {}
    for column in df:
        # average of each of these simulations. Might be a problem if they're
        # different length arrays (n may vary greatly)!
        mean_dict[column] = df[column].mean()

    while not os.path.exists( consol_name_dir ):
        # make a new directory that is the combination of these different files
        os.makedirs( consol_name_dir );

    filename = consol_name_dir + os.sep + NPZ_SHORT_FILE

    # save results in a npz file
    np.savez(filename, **mean_dict)

    return 0
