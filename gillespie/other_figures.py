import os, glob, csv, pickle, copy, time, scipy

import numpy as np; import pandas as pd; import scipy.io as sio;
import matplotlib as mpl; import matplotlib.pyplot as plt;
from matplotlib import colors, ticker; from matplotlib.lines import Line2D

import src.equations as eq
import src.consolidate as cdate
import src.plotting_fcns as pltfcn
import src.analysis as nlss
import src.theory_equations as theqs

from src.gillespie_models import RESULTS_DIR, MultiLV
from src.manual_revision import DICT_REVISION, correlations_fix
from src.settings import VAR_NAME_DICT, COLOURS, IMSHOW_KW, NPZ_SHORT_FILE\
                    , VAR_SYM_DICT

np.seterr(divide='ignore', invalid='ignore')

OTHER_FIG_DIR = 'figures' + os.sep + 'other' # FIX TO NOT BE GLOBAL

#POINTS_BETWEEN_X_TICKS = 20; POINTS_BETWEEN_Y_TICKS = 20

def mlv_plot_single_sim_results(dir, sim_nbr = 1):
    """
    Plot information collected in a single results_(sim_nbr).pickle

    Input :
        dir     : directory that we're plotting from
        sim_nbr : simulation number (subdir sim%i %(sim_nbr))

    Output :
        Plots of a single simulation
    """
    # TODO : replace with a dict
    param_dict, ss_dist_sim, richness_sim, time_present_sim, mean_pop_sim\
                  , mean_rich_sim, mean_time_present_sim, _, _, _, _, _ , _, _\
                  , conditional, av_J\
                  = nlss.mlv_extract_results_sim(dir, sim_nbr=sim_nbr)

    # theory equations
    #theory_models   = theqs.Model_MultiLVim(**param_dict)
    #conv_dist, _ = theory_models.abund_1spec_MSLV()
    #mf_dist, mf_abund = theory_models.abund_sid()
    title = r'$\rho=$' + str(param_dict['comp_overlap']) + r', $\mu=$' \
            + str(param_dict['immi_rate']) + r', $S=$' + str(param_dict['nbr_species'])

    fig = plt.figure()
    plt.scatter(np.arange(len(richness_sim)), richness_sim, color='b')
    plt.ylabel(r"probability of richness")
    plt.xlabel(r'richness')
    plt.axvline( mean_rich_sim, color='k' , linestyle='dashed', linewidth=1)
    plt.title(title)
    fname = 'richness' + 'sim' + str(sim_nbr)
    plt.savefig(dir + os.sep + f'sim{sim_nbr}' + os.sep + fname + '.pdf');
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.show()
    plt.close()

    ## dstbn present
    if time_present_sim != []:
        dirTime = os.getcwd() + os.sep + RESULTS_DIR + os.sep + dir + os.sep + 'time'
        while not os.path.exists( dirTime ):
            os.makedirs(dirTime);

        nbins   = 100
        logbins = np.logspace(np.log10(np.min(time_present_sim))
                              , np.log10(np.max(time_present_sim)), nbins)
        counts, bin_edges = np.histogram(time_present_sim, density=True
        #                                , bins=logbins)
                                         , bins = nbins)
        fig  = plt.figure()
        axes = plt.gca()
        plt.scatter((bin_edges[1:]+bin_edges[:-1])/2,
                 counts, color='g')
        plt.axvline( mean_time_present_sim, color='k', linestyle='dashed'
                    , linewidth=1 ) # mean
        plt.ylabel(r"probability of time present")
        plt.xlabel(r'time present between extinction')
        plt.yscale('log')
        axes.set_ylim([np.min(counts[counts!=0.0]),2*np.max(counts)])
        plt.title(title)
        fname = 'time_present_' + 'sim' + str(sim_nbr)
        plt.savefig(dirTime + os.sep + fname + '.pdf');
        #plt.xscale('log')
        #plt.show()
        plt.close()

    ## ss_dstbn (compare with deterministic mean)
    """
    fig  = plt.figure()
    axes = plt.gca()

    plt.scatter(np.arange(len(ss_dist_sim)), ss_dist_sim, label='simulation')
    plt.plot(np.arange(len(conv_dist)),conv_dist,label='convolution approx.')
    plt.plot(np.arange(len(mf_dist)),mf_dist,label='mean field approx.')
    plt.ylabel(r"probability distribution function")
    plt.xlabel(r'n')
    plt.axvline( mean_pop_sim, color='r' , linestyle='dashed'
                , linewidth=1 ) #mean
    plt.axvline( theory_models.deterministic_mean(), color='k' ,
                linestyle='dashdot', linewidth=1 ) #mean
    setattr(theory_models,'nbr_species',int(mean_rich_sim))
    plt.axvline( theory_models.deterministic_mean(), color='b' ,
                linestyle='-', linewidth=1 ) #mean
    plt.yscale('log')
    axes.set_ylim([np.min(ss_dist_sim[ss_dist_sim!=0.0]),2*np.max(ss_dist_sim)])
    title = r'$\rho=$' + str(param_dict['comp_overlap']) + r', $\mu=$' \
            + str(param_dict['immi_rate']) + r', $S=$' + str(param_dict['nbr_species'])
    plt.title(title)
    axes.set_xlim([0.0,np.max(np.nonzero(ss_dist_sim))])
    plt.legend(loc='best')
    fname = 'distribution' + 'sim' + str(sim_nbr)
    plt.savefig(dir + os.sep + fname + '.pdf');
    #plt.xscale('log')
    #plt.show()
    """

    # P(i|j), understanding conditional probabilities
    dirCond = os.getcwd() + os.sep + dir + os.sep + 'conditional'
    while not os.path.exists( dirCond ):
        os.makedirs(dirCond);

    fig  = plt.figure()
    axes = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('PuBu'))
    my_cmap.set_bad((0,0,0))
    for i in np.arange(0,conditional.shape[0]):
        conditional[i,i] *= 2
        if np.sum(conditional[i,:]) != 0.0:
            conditional[i,:] /= np.sum(conditional[i,:])

    print(conditional[100][99],conditional[99][100])

    print(np.sum(conditional,axis=1))
    #print(np.sum(conditional,axis=0))
    #print(np.sum(conditional,axis=0),np.sum(conditional,axis=1))
    plt.imshow( conditional[:2*param_dict['carry_capacity']
                ,:2*param_dict['carry_capacity']].T
                , norm=mpl.colors.LogNorm(), cmap=my_cmap
                , interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.ylabel(r"i")
    plt.xlabel(r'j')
    title = r'$\rho=$' + str(param_dict['comp_overlap']) + r', $\mu=$' \
            + str(param_dict['immi_rate']) + r', $S=$' + str(param_dict['nbr_species'])
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(r'$P(i|j)$')
    fname = 'P_i_j_' + 'sim' + str(sim_nbr)
    plt.savefig(dirCond + os.sep + fname + '.pdf');
    #plt.xscale('log')
    #plt.show()
    plt.close()

    return 0


def pearson_correlation_plots(filename, xlabel='immi_rate', ylabel='comp_overlap', start=0):

    data = np.load(filename); #plt.style.use('src/custom_heatmap.mplstyle')
    plt.style.use('src/custom.mplstyle')
    rangex = data[xlabel][start:]; rangey = data[ylabel][:]

    pltfcn.heatmap(rangex, rangey, data['corr_ni_nj2D'].T, xlabel, ylabel, r'$cov(n_i,n_j)/\sigma_{n_i}\sigma_{n_j}$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    pltfcn.heatmap(rangex, rangey, data['corr_J_n2D'].T, xlabel, ylabel, r'$cov(J,n)/\sigma_{J}\sigma_{n}$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    pltfcn.heatmap(rangex, rangey, data['corr_Jminusn_n2D'].T, xlabel, ylabel, r'$cov(J-n,n)/\sigma_{J-n}\sigma_{n}$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    return 0

def average_conditional(filename, xlabel='immi_rate', ylabel='comp_overlap', start=0):

    data = np.load(filename); #plt.style.use('src/custom_heatmap.mplstyle')
    S = data['nbr_species']

    rangex = data[xlabel][start:]; rangey = data[ylabel][:]

    sim_dist = data['sim_dist'][start:,:,:]
    av_ni_given_nj = data['av_ni_given_nj2D'][start:,:,:]

    av_n = np.sum(sim_dist * np.arange(0,sim_dist.shape[2])[np.newaxis,np.newaxis,:], axis=2)

    av_av_ni_given_nj = np.sum(sim_dist * av_ni_given_nj, axis=2)
    var_av_ni_given_nj = np.sum( sim_dist * (av_ni_given_nj**2), axis=2 )\
                                    - ( av_av_ni_given_nj )**2
    ratio_var_av = var_av_ni_given_nj / ( av_av_ni_given_nj **2 )


    i = 0; j=0;
    datapoints = av_ni_given_nj[i,j]
    print(datapoints[200])
    horizontal = [np.argmax(datapoints)+10,np.argmax(datapoints)+10]
    locHorizontal = [av_av_ni_given_nj[i,j]-np.sqrt(var_av_ni_given_nj[i,j])/2, av_av_ni_given_nj[i,j]+np.sqrt(var_av_ni_given_nj[i,j])/2]
    plt.figure()
    plt.scatter(np.arange(0, sim_dist.shape[2]), datapoints)
    plt.plot(horizontal, locHorizontal, color='b')
    plt.axhline(y=av_av_ni_given_nj[i,j], color='g')
    plt.show()
    plt.close()

    pltfcn.heatmap(rangex, rangey, av_av_ni_given_nj.T, xlabel, ylabel, r'E$_{n_j}[\langle n_i|n_j \rangle]$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    pltfcn.heatmap(rangex, rangey, var_av_ni_given_nj.T, xlabel, ylabel, r'Var$_{n_j}(\langle n_i|n_j \rangle)$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    pltfcn.heatmap(rangex, rangey, ratio_var_av.T, xlabel, ylabel, r'Var$_{n_j}(\langle n_i|n_j \rangle)/$E$_{n_j}[\langle n_i|n_j \rangle]^2$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    # Same for <J-n_j|n_j> = (S-1)<n_i|n_j>
    pltfcn.heatmap(rangex, rangey, (S-1)*av_av_ni_given_nj.T, xlabel, ylabel, r'E$_{n_j}[\langle J-n_j|n_j \rangle]$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    pltfcn.heatmap(rangex, rangey, ((S-1)**2 * var_av_ni_given_nj).T, xlabel, ylabel, r'Var$_{n_j}(\langle J-n_j|n_j \rangle)$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    pltfcn.heatmap(rangex, rangey, ratio_var_av.T, xlabel, ylabel, r'Var$_{n_j}(\langle J-n_j|n_j \rangle)/$E$_{n_j}[\langle J-n_j|n_j \rangle]^2$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    # <J|n_j> = n_j + (S-1)<n_i|n_j>
    vector_n = np.arange(0, av_ni_given_nj.shape[-1])
    n_j = np.ones( av_ni_given_nj.shape ) * vector_n[np.newaxis,np.newaxis,:]
    av_J_given_nj = (S-1) * av_ni_given_nj + n_j


    av_av_J_given_nj = np.sum(sim_dist * av_J_given_nj, axis=2)
    var_av_J_given_nj = np.sum( sim_dist * (av_J_given_nj**2), axis=2 )\
                                    -  ( av_av_J_given_nj )**2
    ratio_var_av_J = var_av_J_given_nj / ( av_av_J_given_nj **2 )

    pltfcn.heatmap(rangex, rangey, av_av_J_given_nj.T, xlabel, ylabel, r'E$_{n_j}[\langle J|n_j \rangle]$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    pltfcn.heatmap(rangex, rangey, var_av_J_given_nj.T, xlabel, ylabel, r'Var$_{n_j}(\langle J|n_j \rangle)$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)

    pltfcn.heatmap(rangex, rangey, ratio_var_av_J.T, xlabel, ylabel, r'Var$_{n_j}(\langle J|n_j \rangle)/$E$_{n_j}[\langle J|n_j \rangle]^2$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)


    pltfcn.heatmap(rangex, rangey, (ratio_var_av_J/ratio_var_av).T, xlabel, ylabel, r'$J-n/J$'
                    , OTHER_FIG_DIR, pbtx=20, pbty=20, save=True, xlog=True
                    , ylog=True, xdatalim=None, ydatalim=None)


    return 0

if __name__ == "__main__":

    while not os.path.exists( os.getcwd() + os.sep + OTHER_FIG_DIR ):
        os.makedirs(os.getcwd() + os.sep + OTHER_FIG_DIR);

    save = True
    plt.style.use('src/custom.mplstyle')

    sim_immi        = RESULTS_DIR + os.sep + 'multiLV71'
    sim_immi_inset  = RESULTS_DIR + os.sep + 'multiLV79'
    sim_spec        = RESULTS_DIR + os.sep + 'multiLV77'
    sim_corr        = RESULTS_DIR + os.sep + 'multiLV80'
    sim_time        = RESULTS_DIR + os.sep + 'multiLV6'
    sim_avJ         = RESULTS_DIR + os.sep + 'multiLVNavaJ'

    #mlv_plot_single_sim_results(sim_immi, sim_nbr = 1)
    # Create appropriate npz file for the sim_dir
    #cdate.mlv_consolidate_sim_results( sim_spec, 'nbr_species', 'comp_overlap')
    #cdate.mlv_consolidate_sim_results( sim_immi, 'immi_rate', 'comp_overlap')
    #cdate.mlv_consolidate_sim_results( sim_avJ, 'immi_rate', 'comp_overlap')
    #cdate.mlv_consolidate_sim_results( sim_corr, 'immi_rate', 'comp_overlap')
    #cdate.mlv_consolidate_sim_results(sim_time, parameter1='immi_rate', parameter2='comp_overlap')

    # plots many SAD distributions, different colours for different
    #many_parameters_dist(sim_immi+os.sep+NPZ_SHORT_FILE,save=True, fixed=4, start=20)
    #many_parameters_dist(sim_immi+os.sep+NPZ_SHORT_FILE,save=True, fixed=20, start=20)
    #many_parameters_dist(sim_immi+os.sep+NPZ_SHORT_FILE, range='immi_rate', save=save, start=20, fixed=30)

    # plots of maximums as a function of either immi_rate or comp_overlap
    #maximum_plot(sim_immi+os.sep+NPZ_SHORT_FILE, range='immi_rate', save=save, start=20)
    #maximum_plot(sim_immi+os.sep+NPZ_SHORT_FILE, range='comp_overlap', save=save, start=0)

    # Phase diagram
    #figure_richness_phases(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40))
    #figure_modality_phases(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #figure_regimes(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #figure_regimes(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species',xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)

    # Richness between experiments and models
    #compare_richness(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')

    # correlation business
    #fig_corr(sim_corr+os.sep+NPZ_SHORT_FILE, save, revision='80')
    #fig_timecorr(sim_time + os.sep + "sim1" + os.sep + "results_0.pickle")
    #fig3A(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #pearson_correlation_plots(sim_corr + os.sep + NPZ_SHORT_FILE)
    average_conditional(sim_corr + os.sep + NPZ_SHORT_FILE)


    # plots trajectories, however it would appear that MultiLV6 (traj.zip) has been deleted
    #plot_trajectory(sim_time, 19, 100000, colour='mediumturquoise', save=save)
    #plot_trajectory(sim_time, 34, 100000, colour='khaki', save=save)


    # OLD STUFF
    #figure_richness_phases(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40))
    #figure_modality_phases(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #figure_richness_phases(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species', xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)
    #figure_modality_phases(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species', distplots=False, xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)
    #figure_regimes(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species',xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)
    #figure_modality_phases(sim_immi_inset+os.sep+NPZ_SHORT_FILE, save, ylog=False, xlog=False, distplots=False, pby=15)
    #figure_modality_phases(sim_spec+os.sep+NPZ_SHORT_FILE, save)
    #fig3A(npz_file, save)
    #fig3B(npz_file, save)
    #fig3CDE(npz_file, save)
