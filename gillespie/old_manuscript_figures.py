import os, glob, csv, pickle, copy, time, scipy

import numpy as np; import pandas as pd; import scipy.io as sio;
import matplotlib as mpl; import matplotlib.pyplot as plt;
from matplotlib import colors, ticker; from matplotlib.lines import Line2D

from src.autocorrelation import autocorrelation_spectrum as autospec
from src.autocorrelation import average_timescale_autocorrelation, exponential_fit_autocorrelation
import src.equations as eq
import src.consolidate as cdate
import src.plotting_fcns as pltfcn

from src.gillespie_models import RESULTS_DIR, MultiLV
from src.manual_revision import DICT_REVISION, correlations_fix
from src.settings import VAR_NAME_DICT, COLOURS, IMSHOW_KW, NPZ_SHORT_FILE\
                    , VAR_SYM_DICT

np.seterr(divide='ignore', invalid='ignore')

MANU_FIG_DIR = 'figures' + os.sep + 'manuscript' # FIX TO NOT BE GLOBAL

#POINTS_BETWEEN_X_TICKS = 20; POINTS_BETWEEN_Y_TICKS = 20

def plot_prob(probability, i, j, colour):
    DIST_DIR = MANU_FIG_DIR + os.sep + 'distributions'
    os.makedirs(DIST_DIR, exist_ok=True)
    f = plt.figure(figsize=(1.75,1.25)); fig = plt.gcf(); ax = plt.gca()
    plt.plot(probability[:125], color=colour, linewidth=2, zorder=10)
    plt.xticks([0,50,100])
    size = 7; ax.axes.labelsize = size
    plt.xlabel(r'n',fontsize=size, labelpad=2);
    plt.ylabel(r'P(n)', fontsize=size, labelpad=2);
    plt.xlim([0,125]); plt.yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=size)
    #ax.tick_params(axis='both', which='minor', labelsize=6)

    if save:
        plt.savefig(DIST_DIR + os.sep + "dstbn_" + str(i) + '_' + str(j)+'.pdf')
    else:
        plt.show()
    plt.close()

    return 0

def figure_richness_phases(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    """
    Heatmap of richness, with
    """
    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')

    # species simulations didn't all work, erase the 0s
    start = 0
    if xlabel=='nbr_species':
        start = 1

    # transform
    rangex = data[xlabel][start:]; rangey = data[ylabel][:]

    # simulation/mf results
    sim_rich = 1 - data['sim_dist'][start:,:,0]
    mf_rich  = 1 - data['mf_dist'][start:,:,0]

    # if nbr_species varies in simulation
    arr = np.ones( ( np.shape(sim_rich)[0], np.shape(sim_rich)[1] ) )
    if xlabel=='nbr_species':
        sim_nbr_spec = data['nbr_species'][start:,None]*sim_rich
        S =  data['nbr_species'][start:,None]*arr
        sim_frac_spec_minus1_2 = (data['nbr_species'][start:,None]-0.5)/(data['nbr_species'][start:,None])*arr
        mf_nbr_spec = data['nbr_species'][start:,None]*mf_rich
    elif ylabel=='nbr_species':
        sim_nbr_spec = data['nbr_species'][start:,None]*sim_rich
        S =  data['nbr_species'][start:,None]*arr
        sim_frac_spec_minus1_2 = (data['nbr_species']-0.5)/(data['nbr_species'])[start:,None]*arr
        mf_nbr_spec = S*mf_rich
    else:
        sim_nbr_spec = data['nbr_species']*sim_rich
        mf_nbr_spec = data['nbr_species']*mf_rich
        sim_frac_spec_minus1_2 = (data['nbr_species']-0.5)/data['nbr_species']*arr

    # _cat signifies categories (one of the richnesses)
    sim_rich_cat = np.zeros( ( np.shape(sim_rich)[0], np.shape(sim_rich)[1] ) )
    mf_rich_cat = np.zeros( ( np.shape(mf_rich)[0], np.shape(mf_rich)[1] ) )

    # richness determination
    for i in np.arange(np.shape(sim_rich_cat)[0]):
        for j in np.arange(np.shape(sim_rich_cat)[1]):
            if sim_rich[i,j] >= sim_frac_spec_minus1_2[i,j]:
                sim_rich_cat[i,j] = 1.0
            elif sim_nbr_spec[i,j] < 1.5:
                sim_rich_cat[i,j] = -1.0
            else: pass

            if mf_rich[i,j] >= sim_frac_spec_minus1_2[i,j]:
                mf_rich_cat[i,j] = 1.0
            elif mf_nbr_spec[i,j] < 2:
                mf_rich_cat[i,j] = -1.0
            else: pass

    # plots
    f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()

    # boundaries and colours
    lines = [-1,0,1]; bounds = [-1.5, -0.5, 0.5, 1.5];
    line_colours = ['royalblue','cornflowerblue','lightsteelblue']
    cmap = colors.ListedColormap( line_colours )
    norm = colors.BoundaryNorm( bounds, cmap.N )

    im = plt.imshow(sim_rich_cat.T, cmap=cmap, norm=norm, aspect='auto')

    # contours
    mf_rich_cat = scipy.ndimage.filters.gaussian_filter(mf_rich_cat,1.0) #smooth
    if start == 0:
        MF = ax.contour( mf_rich_cat.T, [-0.5], linestyles='solid', colors = 'k'
                                    , linewidths = 2)
    MF2 = ax.contour( mf_rich_cat.T, [0.5], linestyles='solid', colors = 'k'
                                , linewidths = 2)

    pltfcn.set_axis(ax, plt, POINTS_BETWEEN_X_TICKS, POINTS_BETWEEN_Y_TICKS, rangex
                    , rangey, xlog, ylog, xdatalim, ydatalim, xlabel, ylabel)

    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "richness_"+ xlabel + '.pdf');
        #plt.savefig(MANU_FIG_DIR + os.sep + "richness" + '.png');
    plt.close()

    # 2D heatmap of species present, show a couple example richness plots
    if xlabel != 'nbr_species':
        f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()
        R=np.shape(data['rich_dist'])[2]-1
        plt.plot(data['rich_dist'][0,R], color=line_colours[0]
                    ,marker='s', markeredgecolor='None', linewidth=2, zorder=10)
        plt.plot(data['rich_dist'][int(5*(R)/10), int(8*(R)/10)]
                    , color=line_colours[1], marker='o', markeredgecolor='None'
                    ,linewidth=2, zorder=10)
        plt.plot(data['rich_dist'][R,0], color=line_colours[2]
                    , marker='D', markeredgecolor='None',linewidth=2, zorder=10)
        plt.ylim([0,1.0])
        plt.xlim([0,np.shape(data['rich_dist'])[2]-1])
        plt.xlabel(r'species present, $\langle S^* \rangle$')
        plt.ylabel(r'P($S^*$)')

        if save:
            plt.savefig(MANU_FIG_DIR + os.sep + "richness_dist_"+xlabel+'.pdf');
            #plt.savefig(MANU_FIG_DIR + os.sep + "richness" + '.png');
        plt.close()

    return sim_rich_cat, mf_rich_cat, lines, mf_rich, sim_rich

def compare_richness(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    """
    compare sim richness with the richness from the rates found in Appendix I
    """
    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')

    # transform
    rangex = data[xlabel][:]; rangey = data[ylabel][:]

    # simulation/mf results
    K = data['carry_capacity']; rminus = data['death_rate'];
    rplus = data['birth_rate'];
    mu = data['immi_rate']; S = data['nbr_species']
    sim_rich = 1 - data['sim_dist'][:,:,0]
    rates_rich = np.zeros( ( np.shape(sim_rich)[0], np.shape(sim_rich)[1] ) )
    for i in np.arange( np.shape(sim_rich)[0] ):
        for j in np.arange(  np.shape(sim_rich)[1] ):
            dstbn = data['sim_dist'][i,j]
            rates_rich[i,j] = eq.richness_from_rates( dstbn, rplus, rminus, K
                                                    , rangey[i], rangex[j], S )

    # Fig SImualtion richness
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(( 0,0,0 ))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None}
    # heatmap
    im = plt.imshow( sim_rich.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle S^* \rangle$ simulation')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "richness_simulation" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "richness_simulation" + '.png');
    else:
        plt.show()
    plt.close()

    # Fig rates richness
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(( 0,0,0 ))
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None}
    # heatmap
    im = plt.imshow( sim_rich.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle S^* \rangle$ simulation')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "eq.richness_from_rates(" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "eq.richness_from_rates(" + '.png');
    else:
        plt.show()
    plt.close()

    return 0


def figure_modality_phases(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    """
    Heatmap of modality
    Need to smooth out simulation results. How to compare?
    """
    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')

    # exclude first row in nbr_species because certain of those sims failed
    start = 0
    if xlabel=='nbr_species':
        start = 1

    # setting simulation parameters
    rangex = data[xlabel][start:]; rangey = data[ylabel][:]
    K = data['carry_capacity']; rminus = data['death_rate'];
    rplus = data['birth_rate']; mu = data['immi_rate']; S = data['nbr_species']

    # simulation results
    sim_dist = data['sim_dist'][start:,:,:]
    mf_dist = data['mf_dist'][start:,:,:]
    rich_dist = data['rich_dist'][start:,:,:]

    # calculating mean J
    #mf_meanJ = eq.meanJ_est(mf_dist, (np.shape(rich_dist)[2]-1))
    mf_rich = np.shape(mf_dist)[2] * ( 1 - mf_dist[:,:,0] )
    mf_meanJ = eq.meanJ_est(sim_dist, (np.shape(rich_dist)[2]-1))

    # 2D rho-mu arrays
    rho = (rangey*np.ones( (np.shape(sim_dist)[0], np.shape(sim_dist)[1])))
    if xlabel == 'immi_rate':
        mu  = (rangex*np.ones( (np.shape(sim_dist)[0]
                                , np.shape(sim_dist)[1])).T).T

    # calculating modality
    modality_sim, line_names, line_colours = pltfcn.determine_modality(
            sim_dist, distplots, revision, False )
    lines = [float(i) for i in list( range( 0, len(line_names) ) ) ]
    bounds = [ i - 0.5 for i in lines + [lines[-1] + 1.0]  ]
    lines_center = [ a + b for a, b in zip( lines, [0]*len(line_names) ) ]

    # boundaries of modality, just from distribution
    modality_mf, _, _  = pltfcn.determine_modality( mf_dist, False, sampleP0 = False )

    meanN = mf_meanJ / S
    # boundaries of modality equation
    boundaryReIII, boundaryReIIIn = eq.realBoundaryIII3(meanN, rplus, rminus, K, rho, mu, S)
    boundaryI = eq.boundaryI(meanN, rplus, rminus, K, rho, mu, S)

    # plots
    f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()

    cmap = colors.ListedColormap( line_colours )
    norm = colors.BoundaryNorm( bounds, cmap.N )

    im = plt.imshow(modality_sim.T, cmap=cmap, norm=norm, aspect='auto')
    boundaryI = scipy.ndimage.filters.gaussian_filter(boundaryI, 1.0)
    boundaryReIII = scipy.ndimage.filters.gaussian_filter(boundaryReIII, 0.1)
    boundaryReIIIn = scipy.ndimage.filters.gaussian_filter(boundaryReIIIn, 1.0)

    BI = ax.contour( boundaryI.T, [1.0], linestyles='solid'
                        , colors = 'k', linewidths = 2, label=r'Eq. 8')
    BIII = ax.contour( boundaryReIII.T, [0.0], linestyles='solid'
                        , colors = 'b', linewidths = 2, label=r'Eq. 9')
    #BIIIn = ax.contour( boundaryReIIIn.T, [0.5], linestyles='solid'
    #                    , colors = 'r', linewidths = 2)
    #MF2 = ax.contour( mf_unimodal.T, [1.], linestyles='solid'
    #                    , colors = 'k', linewidths = 2)

    pltfcn.set_axis(ax, plt, POINTS_BETWEEN_X_TICKS, POINTS_BETWEEN_Y_TICKS, rangex
                    , rangey, xlog, ylog, xdatalim, ydatalim, xlabel, ylabel)

    h1, _ = BI.legend_elements(); h2, _ = BIII.legend_elements()
    h = [h1[0],h2[0]]
    labels = [r'Eq. 8',r'Eq. 9']
    plt.legend(h, labels, loc='best', facecolor='white', framealpha=0.85)
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "modality_" + xlabel + '.pdf');
        #plt.savefig(MANU_FIG_DIR + os.sep + "modality" + '.png');
    plt.close()

    return modality_sim, boundaryReIII, boundaryI, lines, line_colours

def figure_regimes(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    """
    Heatmap of modality
    Need to smooth out simulation results. How to compare?
    """
    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    sim_rich_cat, mf_rich_cat, lines_rich, mf_rich, sim_rich =\
        figure_richness_phases(filename, False, xlabel, ylabel, xlog, ylog, ydatalim, xdatalim
                            , None, distplots, pbx, pby)
    modality_sim, hubbelRegime, mf_unimodal, lines_mod, colours_mod =\
    figure_modality_phases(filename, False, xlabel, ylabel, xlog, ylog, ydatalim, xdatalim
                        , revision, distplots, pbx, pby)

    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')
    K = data['carry_capacity']; rminus = data['death_rate']; rplus = data['birth_rate'];
    mu = data['immi_rate']; S = data['nbr_species']

    #certain species sims did not end correctly. Set them to 0, need to ignore
    start = 0
    if xlabel=='nbr_species':
        start = 1

    rangex = data[xlabel][start:]; rangey = data[ylabel][:]

    sep = 10
    combined_sim = modality_sim + sep * sim_rich_cat; #defining all categories

    lines = []
    for j in lines_rich:
        lines.extend([ float(i)+sep*j for i in list( range( 0, len(lines_mod) ) ) ])

    bounds = [ i - 0.5 for i in lines + [lines[-1] + 1.0]  ]
    lines_center = [ a + b for a, b in zip( lines, [0]*len(lines) ) ]

    # plots
    f = plt.figure(figsize=(4,3)); fig = plt.gcf(); ax = plt.gca()
    #f = plt.figure(figsize=(2,2)); fig = plt.gcf(); ax = plt.gca()
    line_colours = ['darkgoldenrod', 'darkslateblue', 'blue' , 'indianred', 'slategrey']
    line_colours.extend(colours_mod)
    line_colours.extend(['lemonchiffon', 'mediumpurple', 'paleturquoise', 'mistyrose', 'whitesmoke'])

    # plots
    cmap = colors.ListedColormap( line_colours )
    norm = colors.BoundaryNorm( bounds, cmap.N )

    im = plt.imshow(combined_sim.T, cmap=cmap, norm=norm, aspect='auto')

    # Mean-field contours in black
    #modality_mf = scipy.ndimage.filters.gaussian_filter(modality_mf, 1.0)
    mf_unimodal = scipy.ndimage.filters.gaussian_filter(mf_unimodal, 1.0)
    hubbelRegime = scipy.ndimage.filters.gaussian_filter(hubbelRegime, 1.0)
    #MF = ax.contour( modality_mf.T, [0.5], linestyles='solid'
    #                    , colors = 'k', linewidths = 2)
    MF = ax.contour( hubbelRegime.T, [0.5], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    MF2 = ax.contour( mf_unimodal.T, [1.], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    if start == 0:
        mf_rich_cat = scipy.ndimage.filters.gaussian_filter(mf_rich_cat,0.7) #smooth
        MF3 = ax.contour( mf_rich_cat.T, [-0.5], linestyles='solid', colors = 'k'
                                    , linewidths = 2)
        MF4 = ax.contour( mf_rich_cat.T, [0.5], linestyles='solid', colors = 'k'
                                , linewidths = 2)

    # Equation contours
    testingEqns = False
    if testingEqns:
        # 2D mu-rho
        mu  = (rangex*np.ones( (np.shape(data['sim_dist'])[0]
                                    , np.shape(data['sim_dist'])[1])).T).T

        rho = (rangey*np.ones( (np.shape(data['sim_dist'])[0]
                                    , np.shape(data['sim_dist'])[1])))

        lotkaVolteraSol = eq.deterministic_mean(S, mu, rho, rplus, rminus, K)
        lotkaVolteraAlt = eq.deterministic_mean(S*sim_rich, mu, rho, rplus, rminus, K)
        J = lotkaVolteraSol * S
        Jalt = lotkaVolteraAlt * S * sim_rich

        Eq5 = (1./mu)*(rminus + (rplus-rminus)*( 1. + rho*( J - 1. ) )/K )
        Eq5alt = (1./mu)*(rminus + (rplus-rminus)*( 1. + rho*( Jalt - 1. ) )/K )
        #print(J, lotkaVolteraSol
        Eq6 = ( (rplus-rminus)*(K-rho*J)**2 ) * (1./ (4. * K)) * (1.0 / ( rplus - mu ) ) * ( 1.0 /( 1.0 - rho ) )
        Eq6alt = ( (rplus-rminus)*(K-rho*Jalt)**2 ) * (1./ (4. * K)) * (1.0 / ( rplus - mu ) ) * ( 1.0 /( 1.0 - rho ) )

        #im = plt.imshow(Eq6alt.T,  norm=norm, aspect='auto')
        eq5 = plt.contour( Eq5.T, [1.], linestyles='solid', colors = 'r'
                            , linewidths = 1)
        eq5alt = plt.contour( Eq5alt.T, [1.], linestyles='dashed', colors = 'g'
            , linewidths = 1)
        eq6 = plt.contour( np.real(Eq6.T), [1.], linestyles='solid', colors = 'b'
                            , linewidths = 1)
        eq6alt = plt.contour( np.real(Eq6alt.T), [1.], linestyles='solid', colors = 'orange'
            , linewidths = 1)
        contours = [eq5,eq5alt,eq6,eq6alt]
        h1, _ = eq5.legend_elements(); h2, _ = eq5alt.legend_elements()
        h3, _ = eq6.legend_elements(); h4, _ = eq6alt.legend_elements()
        h = [h1[0], h2[0], h3[0], h4[0]]
        labels = [r'Eq.5; $J=n_{LV}S$',r'Eq.5; $J=n_{LV}^*\langle S^* \rangle$'\
            , r'Eq.6; $J=n_{LV}S$', r'Eq.6; $J=n_{LV}^*\langle S^* \rangle$']
        plt.legend(h, labels, loc='lower left', facecolor='white', framealpha=0.7)

    pltfcn.set_axis(ax, plt, POINTS_BETWEEN_X_TICKS, POINTS_BETWEEN_Y_TICKS, rangex
                    , rangey, xlog, ylog, xdatalim, ydatalim, xlabel, ylabel)

    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "combined_" + xlabel + '.pdf');
        #plt.savefig(MANU_FIG_DIR + os.sep + "modality" + '.png');
    else:
        plt.show()

    return 0

def fig3A(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    """
    Turnover rate
    """
    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')

    # parameter range
    rangex = data[xlabel]; rangey = data[ylabel]
    K = data['carry_capacity']; rminus = data['death_rate'];
    rplus = data['birth_rate']; S = data['nbr_species']

    # 2D mu-rho
    mu  = (rangex*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])).T).T

    rho = (rangey*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])))

    # calculating turnover rate
    nbr = 0
    dist = 'sim_dist' # sim or mf
    meanJ = eq.meanJ_est(data[dist], (np.shape(data['rich_dist'])[2]-1))
    turnover_nava = 1. / (data[dist][:,:,nbr]*mu)
    turnover_calc = ( 1. + data[dist][:,:,nbr] ) / ( data[dist][:,:,nbr]
                        * ( rplus*nbr + mu + nbr*( rminus + ( rplus-rminus )
                        * ( ( 1. - rho ) * nbr + rho * meanJ )/K  ) ) )
    turnover = turnover_nava

    # plots
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('cividis_r')) # copy the default cmap
    my_cmap.set_bad( color_bad )
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    im = plt.imshow( turnover.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)

    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])

    plt.title(r'$\langle T(0\rightarrow 0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "turnover" + '.png');
    else:
        plt.show()

    return 0

def figTaylor(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'):
    """
    Taylors Law
    """
    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')
    rangex = data[xlabel]; rangey = data[ylabel]
    K = data['carry_capacity']; rminus = data['death_rate']; rplus = data['birth_rate'];
    S = data['nbr_species']

    dist = 'mf_dist'
    start = 1
    prob = data[dist][:,:,start:]\
            /np.sum(data[dist][:,:,start:],axis=2)[:,:,np.newaxis]

    square      = np.tensordot( prob,np.arange(start,np.shape(data[dist])[2])**2
                                , axes=(2,0) )
    square_flat = square.flatten()
    mean        = np.tensordot( prob, np.arange(start,np.shape(data[dist])[2])
                                    , axes=(2,0) )
    mean_flat   = mean.flatten()
    variance    = square - mean**2
    variance_flat = square_flat - mean_flat**2

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    """
    plt.scatter(mean,variance,linewidths=0.01)
    plt.xscale('log');plt.yscale('log')
    plt.xlabel(r'$\langle n \rangle$'); plt.ylabel(r'$var(n)$')
    plt.title(r"Taylor's Law")
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "Taylorlaw" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "Taylorlaw" + '.png');
    else:
        plt.show()
    """

    # plot TL scaled
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    cmap = mpl.cm.get_cmap("plasma")
    for i in np.arange(np.shape(mean)[0]):
        for j in np.arange(np.shape(mean)[1]):
            plt.scatter(mean[i,j],variance[i,j], s=(float(j)/float(np.shape(mean)[1]))
            , marker='+', edgecolors=None, c=cmap(i/np.shape(mean)[0]) )
            #plt.scatter(mean[i,j],variance[i,j], s=(float(j)/float(np.shape(mean)[1]))
            #, marker='o', edgecolors=None, c=cmap(i/np.shape(mean)[0]) )
    plt.xscale('log');plt.yscale('log')
    plt.xlabel(r'$\langle n \rangle$'); plt.ylabel(r'$var(n)$')
    plt.title(r"Taylor's Law")
    plt.legend([r"colour : $rho$",r"size : $mu$"])
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "Taylorlaw_SC" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "Taylorlaw_SC" + '.png');
    else:

        plt.show()

def mfpt(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):
    """
    Time to extinction, time to dominance
    """
    # loading
    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')
    K = data['carry_capacity']; rminus = data['death_rate']; rplus = data['birth_rate'];
    S = data['nbr_species']

    start = 0
    if xlabel=='nbr_species':
        start = 1

    # setting simulation parameters
    rangex = data[xlabel][start:]; rangey = data[ylabel][:]
    K = data['carry_capacity']; rminus = data['death_rate'];
    rplus = data['birth_rate'];
    mu = data['immi_rate']; S = data['nbr_species']

    # simulation results
    sim_dist = data['sim_dist'][start:,:,:]
    mf_dist = data['mf_dist'][start:,:,:]
    rich_dist = data['rich_dist'][start:,:,:]

    # 2D rho-mu
    mu  = (rangex*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])).T).T

    rho = (rangey*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])))

    arr             = data['sim_dist']
    meanJ           = eq.meanJ_est(arr, (np.shape(data['rich_dist'])[2]-1))
    max_arr         = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    dom_turnover    = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    sub_turnover    = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    fpt_dominance   = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    fpt_submission  = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    nbr_species_arr = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    min_arr         = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    prob_min_arr    = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    ntilde          = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )
    prob_ntilde_arr = np.zeros( ( np.shape(arr)[0], np.shape(arr)[1] ) )

    for i in np.arange(np.shape(arr)[0]):
        for j in np.arange(np.shape(arr)[1]):
            nbr_species_arr[i,j] = S*(1.0-arr[i,j,0])#np.dot(data['rich_dist'][i,j]
                            #    , np.arange(len(data['rich_dist'][i,j])) )
            nbr = int(eq.deterministic_mean(nbr_species_arr[i,j], mu[i,j],rho[i,j], rplus
                                                , rminus, K))
            ntilde[i,j] = nbr
            #max_arr[i,j] = nbr
            max_arr[i,j] = 1 + np.argmax( arr[i,j,1:] )
            min_arr[i,j] = np.argmin( arr[i,j,:nbr] )
            prob_min_arr[i,j] = sim_dist[i, j, int(min_arr[i,j])]
            prob_ntilde_arr[i,j] = sim_dist[i,j,nbr]

            sub_turnover[i,j]   = eq.mfpt_020( arr[i,j], mu[i,j] )
            dom_turnover[i,j]   = eq.mfpt_a2a( arr[i,j], nbr, mu[i,j], rplus, rminus
                                        , K, rho[i,j], S )
            fpt_submission[i,j] = eq.mfpt_b2a( arr[i,j], 0, nbr, mu[i,j], rplus)
            fpt_dominance[i,j]  = eq.mfpt_a2b( arr[i,j], 0, nbr, mu[i,j], rplus)

    fNmin = ( rplus * min_arr + mu ) * prob_min_arr
    fluxNtilde = ( mu + ntilde * ( ( rplus + rminus ) + ( rplus - rminus )
                    * ( ( 1.0 - rho ) * ntilde + rho * meanJ ) / K )
                    ) * prob_ntilde_arr

    mfpt_approx_excluded = eq.mfpt_1species( fNmin, fluxNtilde, S)
    mfpt_approx_hubbel   = eq.mfpt_hubbel_regime(fluxNtilde, mu, nbr_species_arr, S )

    fpt_cycling    = fpt_dominance + fpt_submission
    ratio_turnover = sub_turnover / dom_turnover
    ratio_dominance_loss = fpt_submission / dom_turnover
    ratio_suppression_loss = fpt_dominance / sub_turnover
    ratio_switch   = fpt_dominance / fpt_submission
    weighted_timescale = ( ( S - nbr_species_arr) * sub_turnover
                                + nbr_species_arr * dom_turnover  ) / S

    color_bad = (211/256,211/256,211/256)
    """
    # Fig3C
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( dom_turnover.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_dom_turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_dom_turnover" + '.png');
    else:
        plt.show()
    plt.close()

    # FIG
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( sub_turnover.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(0\rightarrow 0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_sub_turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_sub_turnover" + '.png');
    else:
        plt.show()
    plt.close()

    ## FIGURE 3D
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( fpt_dominance.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    # title
    plt.title(r'$\langle T(0\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_dominance" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_dominance" + '.png');
    else:
        plt.show()
    plt.close()

    ## FIGURE 3E
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( fpt_submission.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(\tilde{n}\rightarrow 0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_supression" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_supression" + '.png');
    else:
        plt.show()
    plt.close()

    ## FIGURE 3F
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( fpt_cycling.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(\tilde{n}\rightarrow 0) \rangle+\langle T(0\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_cycling" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_cycling" + '.png');
    else:
        plt.show()
    plt.close()

    ## RATIO turnover
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('plasma_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( ratio_turnover.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(0\rightarrow 0) \rangle / \langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_turnover" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_turnover" + '.png');
    else:
        plt.show()
    plt.close()

    ## Ratio go to
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('plasma_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( ratio_switch.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\langle T(\tilde{n}\rightarrow 0) \rangle / \langle T(0\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_switch" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_switch" + '.png');
    else:
        plt.show()
    plt.close()
    """

    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    sim_rich_cat, mf_rich_cat, lines_rich, _, _ =\
        figure_richness_phases(filename, False, xlabel, ylabel, xlog, ylog, ydatalim, xdatalim
                            , None, distplots, pbx, pby)
    modality_sim, modality_mf, mf_unimodal, lines_mod, colours_mod =\
    figure_modality_phases(filename, False, xlabel, ylabel, xlog, ylog, ydatalim, xdatalim
                        , revision, distplots, pbx, pby)

    ## Ratio dominance loss
    f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('plasma_r')) # copy the default cmap

    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    font_label_contour = 8

    im = plt.imshow( ratio_dominance_loss.T, **imshow_kw)
    #im = plt.imshow( (mfpt_approx_excluded/ratio_dominance_loss).T, **imshow_kw)
    #ax1 = ax.contour( ratio_dominance_loss.T, [10.0, 100.0, 10**5], linestyles=['dotted','dashed','solid']
    boundary = S/2; boundary_colour = 'k'


    #ax1 = ax.contour( (ratio_dominance_loss*(1.0-mf_dist[:,:,0])/mf_dist[:,:,0]).T, [boundary]
    ax1 = ax.contour( (ratio_dominance_loss).T, [boundary]
            , linestyles=['dashed'], colors = boundary_colour, linewidths = 1)
    ax2 = ax.contour( (ratio_dominance_loss).T, [K]
            , linestyles=[(0, (3, 1, 1, 1))], colors = boundary_colour, linewidths = 1)
    #ax3 = ax.contour( (ratio_dominance_loss/(S*(1.0-mf_dist[:,:,0]))).T, [1.0]
    #        , linestyles=['dotted'], colors = boundary_colour, linewidths = 1)

    h1, _ = ax1.legend_elements(); h2, _ = ax2.legend_elements()
    #h3, _ = ax3.legend_elements()
    h = [h1[0], h2[0]]#, h3[0]]
    labels = [r'$S/2$', r'$K$']#,r'$\langle S^* \rangle$']
    plt.legend(h, labels, loc='lower left', facecolor='white', framealpha=0.85)
    """

    ax1 = ax.contour( (ratio_dominance_loss/mfpt_approx_excluded).T, [1.0]
            , linestyles=['dotted'], colors = boundary_colour, linewidths = 1)
    ax2 = ax.contour( (ratio_dominance_loss/mfpt_approx_hubbel).T, [1.0]
            , linestyles=['dashed'], colors = boundary_colour, linewidths = 1)

    h1, _ = ax1.legend_elements(); h2, _ = ax2.legend_elements()
    h = [h1[0], h2[0]]
    labels = [r'$Chou$',r'$Hubbel$']
    plt.legend(h, labels, loc='lower left', facecolor='white', framealpha=0.85)
    """
    #ax.clabel(ax1, inline=True, fmt=ticker.LogFormatterMathtext(), fontsize=font_label_contour)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    #plt.title(r'$\langle T(\tilde{n}\rightarrow 0) \rangle / \langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_dominance_loss" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_dominance_loss" + '.png');
    else:
        plt.show()
    plt.close()

    colours_line = ['r','g','b']; plots = [40,20,0];
    markerline = ['o-','x-','D-']
    f = plt.figure(figsize=(3.25,2.5))
    fig = plt.gcf(); ax = plt.gca()
    # lineplot
    font_label_contour = 8
    for i, val in enumerate(plots):
        array = np.array( ratio_dominance_loss.T[:,val] )
        array[array == np.inf] = np.nan
        plt.plot(rangey[ydatalim[0]:ydatalim[1]+1], array[ydatalim[0]:ydatalim[1]+1]
                    , markerline[i], c=colours_line[i]
                    , label=VAR_SYM_DICT[xlabel] + ": " + r'$10^{%d}$' %
                    np.log10(round(rangex[val], 13) ) )

    #plt.text(0.03, S+30, 'niche-like', fontsize=10, va='center', ha='center')#, backgroundcolor='w')
    plt.axhline(y=boundary, color=boundary_colour, linestyle='--')
    plt.axhline(y=K, color=boundary_colour, linestyle='dotted')
    #plt.text(0.03, S-18, 'rare-biosphere', fontsize=10, va='center', ha='center')#, backgroundcolor='w')
    plt.xlim((rangey[ydatalim[0]],rangey[ydatalim[1]]))

    # legend
    plt.legend(loc='best')
    plt.xscale('log'); plt.yscale('log')
    plt.ylabel(r'$ R(\tilde{n} \rightarrow \tilde{n}) / R(\tilde{n} \rightarrow 0)$')
    plt.xlabel(VAR_NAME_DICT[ylabel])
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_dominance_loss_lineplot" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_dominance_loss_lineplot" + '.png');
    else:
        plt.show()
    plt.close()

    ## Ratio supression loss
    f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('plasma_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    boundary = 1./S
    boundary_colour = 'gray'

    mfpt_approx_full = eq.mfpt_return_full(fNmin, mu, nbr_species_arr, S )

    im = plt.imshow( ratio_suppression_loss.T, **imshow_kw)
    #im = plt.imshow( mfpt_approx_full.T, **imshow_kw)
    """
    MF = ax.contour( modality_mf.T, [0.5], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    MF2 = ax.contour( mf_unimodal.T, [1.], linestyles='solid'
                        , colors = 'k', linewidths = 2)
    if start == 0:
        MF3 = ax.contour( mf_rich_cat.T, [-0.5], linestyles='solid', colors = 'k'
                                    , linewidths = 2)
    MF4 = ax.contour( mf_rich_cat.T, [0.5], linestyles='solid', colors = 'k'
                                , linewidths = 2)
    """

    ax1 = ax.contour( ratio_suppression_loss.T, [boundary], linestyles=['dashed']
                        , colors = boundary_colour, linewidths = 1)
    ax2 = ax.contour( (1.0 - mf_dist[:,:,0]).T, [(S-1/2)/S], linestyles=['solid']
                        , colors = boundary_colour, linewidths = 1)
    #ax3 = ax.contour( ratio_suppression_loss.T, [1.], linestyles=['dotted']
    #                    , colors = boundary_colour, linewidths = 1)
    h1, _ = ax1.legend_elements(); h2, _ = ax2.legend_elements()
    #h3, _ = ax3.legend_elements()
    h = [h1[0],h2[0]]#,h3[0]]
    labels = [r'$1/S$',r'$Eq. 10; \langle S^* \rangle = S-1/2$']#,'mean-field']
    plt.legend(h, labels, loc='upper left', facecolor='white', framealpha=0.85)
    #ax.clabel(ax1, inline=True, fmt=ticker.LogFormatterMathtext(), fontsize=font_label_contour)
    """

    ax1 = ax.contour( (ratio_suppression_loss/mfpt_approx_full).T, [1.0], linestyles=['dashed']
                        , colors = boundary_colour, linewidths = 1)

    h1, _ = ax1.legend_elements();
    h = [h1[0]]
    labels = [r'$Full$']
    plt.legend(h, labels, loc='upper left', facecolor='white', framealpha=0.85)
    """
    #ax.clabel(ax1, inline=True, fmt=ticker.LogFormatterMathtext(), fontsize=font_label_contour)

    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    #plt.title(r'$\langle T(0\rightarrow \tilde{n}) \rangle / \langle T(0\rightarrow 0) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_suppression_loss" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_suppression_loss" + '.png');
    else:
        plt.show()
    plt.close()

    f = plt.figure(figsize=(3.25,2.5))
    fig = plt.gcf(); ax = plt.gca()
    # lineplot
    font_label_contour = 8
    for i, val in enumerate(plots):
        array = np.array( ratio_suppression_loss.T[:,val] )
        array[array == np.inf] = np.nan
        plt.plot(rangey[ydatalim[0]:ydatalim[1]+1], array[ydatalim[0]:ydatalim[1]+1]
                    , markerline[i], c=colours_line[i]
                    , label=VAR_SYM_DICT[xlabel] + ": " + r'$10^{%d}$' %
                    np.log10(round(rangex[val], 13) ) )
    #plt.text(0.17, boundary+0.05, 'full coexistence', fontsize=10, va='center', ha='center')#, backgroundcolor='w')
    plt.axhline(y=boundary, color=boundary_colour, linestyle='--')
    plt.axhline(y=(S-1/2)/S, color=boundary_colour, linestyle='-')
    #plt.text(0.08, boundary-0.02, 'partial', fontsize=10, va='center', ha='center')#, backgroundcolor='w')
    plt.xlim((rangey[ydatalim[0]],rangey[ydatalim[1]]))
    # legendimag
    plt.legend(loc='best')
    plt.xscale('log'); plt.yscale('log')
    plt.ylabel(r'$R(0 \rightarrow 0) / R(0 \rightarrow \tilde{n})$ ')
    plt.xlabel(VAR_NAME_DICT[ylabel])
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_suppression_loss_lineplot" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_suppression_loss_lineplot" + '.png');
    else:
        plt.show()
    plt.close()

    ## Weighted timescales
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    # plots
    my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
    my_cmap.set_bad(color_bad)
    imshow_kw = { 'cmap' : my_cmap, 'aspect' : None, 'interpolation' : None
                    , 'norm' : mpl.colors.LogNorm()}
    # heatmap
    im = plt.imshow( weighted_timescale.T, **imshow_kw)
    pltfcn.set_axis(ax, plt, pbx, pby, rangex, rangey, xlog, ylog, xdatalim, ydatalim
                , xlabel, ylabel)
    # colorbar
    cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
    plt.title(r'$\frac{S-\langle S^* \rangle }{S}\langle T(0\rightarrow 0) \rangle + \frac{\langle S^* \rangle }{S}\langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_weighted_timescale" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_weighted_timescale" + '.png');
    else:
        plt.show()
    plt.close()

    return 0

def fig3F(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'):
    """
    Something to do with the deterministic mean?
    """

    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')
    rangex = data[xlabel]; rangey = data[ylabel]
    K = data['carry_capacity']; rminus = data['death_rate']; rplus = data['birth_rate'];
    S = data['nbr_species']
    mu  = (rangex*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])).T).T

    rho = (rangey*np.ones( (np.shape(data['sim_dist'])[0]
                                , np.shape(data['sim_dist'])[1])))

    nbr_species = S # Or something different?
    det_mean = K*( ( 1. + np.sqrt( 1.+ 4.*mu*( 1. + rho*( nbr_species - 1. ) ) /
                (K*(rplus-rminus)) ) ) / ( 2.*( 1. + rho*(nbr_species-1.) ) ) )

def many_parameters_dist(filename, save=False, range='comp_overlap', fixed = 2
                                , start=0, xlabel='immi_rate'
                                , ylabel='comp_overlap'):

    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')
    rangex = data[xlabel]; rangey = data[ylabel][start:]
    if range == xlabel: other = rangey; otherLabel = ylabel; plotRange = rangex; nbr=fixed
    else: other = rangex; otherLabel = xlabel; plotRange = rangex; nbr=start+fixed

    plotRange = data[range]
    # plots
    f = plt.figure(figsize=(3.5,2.5)); fig = plt.gcf(); ax = plt.gca()
    cmap = mpl.cm.get_cmap("viridis")
    divide = 2; plotRange = plotRange[::divide] #divide reduces number points

    for j, element in enumerate(plotRange):
        color = cmap(j/len(plotRange ) )
        if range == xlabel:
            plt.plot(np.arange(0,np.shape(data['sim_dist'])[2])
                    , data['sim_dist'][divide*j][start+fixed], lw=2, c=color )
        else:
            plt.plot(np.arange(0,np.shape(data['sim_dist'])[2])
                    , data['sim_dist'][fixed][start+divide*j], lw=2, c=color )

    plt.yscale('log')
    plt.xlim(0,int(1.5*data['carry_capacity']))
    plt.ylim(0.00001,.1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(
                vmin=np.min(plotRange), vmax=np.max(plotRange)))
    clb = plt.colorbar(sm)
    clb.set_label(VAR_SYM_DICT[range], labelpad=-30, y=1.1, rotation=0)
    plt.xlabel(r'$n$'); plt.ylabel(r'$P(n)$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + str(other[nbr]) +'.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + str(other[nbr]) +'.png');
    else:
        plt.show()

    return 0

def maximum_plot(filename, save=False, range='comp_overlap', start=0
                                , xlabel='immi_rate', ylabel='comp_overlap'):

    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')
    rangex = data[xlabel]; rangey = data[ylabel][start:]
    if range == xlabel: other = rangey; otherLabel = ylabel; plotRange = rangex
    else: other = rangex; otherLabel = xlabel; plotRange = rangey

    # plots
    f = plt.figure(figsize=(3.5,2.5)); fig = plt.gcf(); ax = plt.gca()
    cmap = mpl.cm.get_cmap("viridis")
    divide = 4; plotRange = plotRange[::divide] #divide reduces number points

    for j, element in enumerate(plotRange):
        color = cmap(j/len(plotRange ) )
        max0 = []; loc_max0 = [];
        maxN = []; loc_maxN = []
        for i, otherVal in enumerate(other):
            if range == xlabel:
                dstbn =  data['sim_dist'][divide*j][start+i]
            else:
                dstbn =  data['sim_dist'][i][start+divide*j]
            # Max at 0
            if dstbn[0] > dstbn[1]:
                max0.append(0); loc_max0.append(otherVal)
            max_excl0 = np.argmax(dstbn[1:])
            # Max at N
            if max_excl0 != 0:
                maxN.append(max_excl0+1); loc_maxN.append(otherVal)

        plt.scatter(loc_max0, max0, s=10/np.sqrt(j+1), lw=2, c=color
                            , edgecolor='none' )
        plt.scatter(loc_maxN, maxN, s=10/np.sqrt(j+1), lw=2, c=color
                            , edgecolor='None' )

    plt.xscale('log'); plt.xlim(np.min(other), np.max(other))
    plt.ylim(0.,data['carry_capacity'])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(
                vmin=np.min(plotRange), vmax=np.max(plotRange)))
    clb = plt.colorbar(sm)
    clb.set_label(VAR_SYM_DICT[range], labelpad=-30, y=1.1, rotation=0)
    plt.xlabel(VAR_NAME_DICT[otherLabel]); plt.ylabel(r'$\tilde{n}$')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + 'maximums' +'.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + 'maximums' +'.png');
    else:
        plt.show()

    return 0

def convert_npz_mat(filename):
    data = np.load(filename)
    data_dict = {}
    for key in list(data.keys()):
        data_dict[key] = data[key]

    sio.savemat(filename[:-3] + 'mat', mdict=data_dict)
    print('done', filename[:-3] + 'mat')

    return 0

def fig_corr(filename, save=False, xlabel='immi_rate', ylabel='comp_overlap'
                    , xlog=True, ylog=True, ydatalim=None, xdatalim=None
                    , revision=None, distplots=False, pbx=20, pby=20):

    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')

    # transform
    start = 0
    if xlabel=='nbr_species': start = 1

    rangex = data[xlabel][start:]; rangey = data[ylabel][:]

    K = data['carry_capacity']; rminus = data['death_rate']
    rplus = data['birth_rate']; S = data['nbr_species']

    correff = [ 'corr_ni_nj2D', 'corr_J_n2D', 'corr_Jminusn_n2D'\
                , 'coeff_ni_nj2D', 'coeff_J_n2D', 'coeff_Jminusn_n2D'
                    ]
    corref_name = [r'cov($n_i,n_j$)/$\sigma_{n_i}\sigma_{n_j}$'\
                    , r'cov($J,n$)/$\sigma_{J}\sigma_{n}$'\
                    , r'cov($J-n,n$)/$\sigma_{J-n}\sigma_{n}$'\
                    , r'cov($n_i,n_j$)/$\langle n_i \rangle \langle n_j\rangle$'\
                    , r'cov($J,n$)/$\langle J \rangle \langle n \rangle$'\
                    , r'cov($J-n,n$)/$\langle (J-n) \rangle \langle n \rangle$'\
                    ]

    # plots
    #divnorm = colors.DivergingNorm(vmin=-0.1, vcenter=0.0, vmax=.0)
    divnorm = None
    imshow_kw = { 'aspect' : None, 'interpolation' : None, 'norm' : divnorm }
    imshow_kw['cmap'] = copy.copy(mpl.cm.get_cmap('RdBu'))

    for i, score in enumerate( correff ):
        max = np.max([-np.min(data[score]),np.max(data[score])])
        min = np.min([-np.max(data[score]),np.min(data[score])])
        divnorm = colors.DivergingNorm(vmin=min, vcenter=0.0, vmax=max)
        imshow_kw['norm'] = divnorm
        """
        if score[:3] == 'cor':
            my_cmap = copy.copy(mpl.cm.get_cmap('magma_r')) # copy the default cmap
            my_cmap.set_bad(( 0,0,0 ))
        else:
            my_cmap = copy.copy(mpl.cm.get_cmap('viridis_r')) # copy the default cmap
            my_cmap.set_bad(( 0,0,0 ))
        imshow_kw['cmap'] = my_cmap
        """

        f = plt.figure(figsize=(3.25,2.5)); fig = plt.gcf(); ax = plt.gca()
        im = plt.imshow( (data[score]).T, **imshow_kw)
        pltfcn.set_axis(ax, plt, pbx, pby,rangex, rangey, xlog, ylog, xdatalim
                        , ydatalim, xlabel, ylabel)

        cb = plt.colorbar(ax=ax, cmap=imshow_kw['cmap'])
        #tick_locator = mpl.ticker.MaxNLocator(nbins=6)
        #cb.locator = tick_locator
        #cb.update_ticks()
        plt.title(corref_name[i])
        if save:
            plt.savefig(MANU_FIG_DIR + os.sep + score + '.pdf');
            #plt.savefig(MANU_FIG_DIR + os.sep + score + '.png');
        else:
            plt.show()
    return 0

def fig_timecorr(sim, sim_nbr, save=False, xlabel='immi_rate'
                    , ylabel='comp_overlap', start=0, xlog=True, ylog=True
                    , ydatalim=None, xdatalim=None, revision=None, pbx=20
                    , pby=20 ):

    filename = sim + os.sep + 'sim' + str(sim_nbr) + os.sep + 'results_0.pickle'
    with open(filename, 'rb') as handle: data = pickle.load(handle)

    plt.style.use('src/custom_heatmap.mplstyle')

    n = 2 # TODO : Need the [2:] for now, not sure why...
    autocor, spectrum, specAutocor, specSpectrum, newTimes =\
            autospec(data['results']['times'][n:],\
            data['results']['trajectory'][n:])

    conditions = VAR_SYM_DICT[xlabel] + ': ' + str(data[xlabel])[:6] + ' and '\
                        + VAR_SYM_DICT[ylabel] + ': ' + str(data[ylabel])[:5]
    fracTime = 10

    #
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    plt.plot(newTimes[:len(specAutocor)//fracTime], specAutocor[:len(specAutocor)//fracTime], 'k')
    #plt.yscale('log');
    plt.xlim(left=0.0)
    plt.title('time correlation richness ' + conditions )
    figname = 'rich_corr.pdf'
    if save:
        plt.savefig(sim + os.sep + 'sim' + str(sim_nbr) + os.sep + figname);
        plt.close()
    else: plt.show()

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    for i in np.random.randint(low=0, high=len(autocor), size=(6,)):
        plt.plot(newTimes[:len(autocor[i])//fracTime], autocor[i][:len(autocor[i])//fracTime])
    plt.xlim(left=0.0); plt.yscale('log');
    plt.title('time correlation abundance ' + conditions)
    figname = 'abund_corr.pdf'
    if save:
        plt.savefig(sim + os.sep + 'sim' + str(sim_nbr) + os.sep + figname);
        plt.close()
    else: plt.show()

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    plt.plot(specSpectrum, 'k');
    plt.xscale('log'); plt.yscale('log')
    plt.title('spectrum richness')
    figname = 'rich_spectrum.pdf'
    if save:
        plt.savefig(sim + os.sep + 'sim' + str(sim_nbr) + os.sep + figname);
        plt.close()
    else: plt.show()

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    for i in np.random.randint(low=0, high=len(spectrum), size=(6,)):
        plt.plot( spectrum[i] )
    plt.yscale('log'); plt.xscale('log')
    plt.title('spectrum abundance')
    figname = 'abund_spectrum.pdf'
    if save:
        plt.savefig(sim + os.sep + 'sim' + str(sim_nbr) + os.sep + figname);
        plt.close()
    else: plt.show()

    return 0

def fig_timescales_autocor(sim, save=False, range='comp_overlap', start=0
                                , xlabel='immi_rate', ylabel='comp_overlap'):

    filename = sim + os.sep + NPZ_SHORT_FILE
    with open(filename, 'rb') as handle: data = pickle.load(handle)

    conditions = VAR_SYM_DICT[xlabel] + ': ' + str(data[xlabel])[:7] + ' and '\
                        + VAR_SYM_DICT[ylabel] + ': ' + str(data[ylabel])[:6]

    data = np.load(filename); plt.style.use('src/custom_heatmap.mplstyle')
    rangex = data[xlabel]; rangey = data[ylabel][start:]
    if range == xlabel: other = rangey; otherLabel = ylabel; plotRange = rangex
    else: other = rangex; otherLabel = xlabel; plotRange = rangey

    ######################### plot Species exponential #########################
    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    cmap = mpl.cm.get_cmap("viridis")

    for j, element in enumerate(plotRange):
        color = cmap(j/len(plotRange ) )
        ax.plot(other, data['time_autocor_spec2D'][i,:], lw=2, fmt='-o'
                    , c=color)
        plt.scatter(other, data['suppress_return2D'][i,:]+['dominance_turnover2D'][i,:]
                    , lw=2, c=color, edgecolor='none' )
    ax.plot(np.NaN, np.NaN,yerr=np.NaN, fmt='-o', color='silver'
                    , label=r'$T_{exp}$ : $e^{-t/T_{exp}}$')
    plt.scatter(np.NaN, np.Nan, lw=2, c='silver', edgecolor='none'
            , label=r'$\langle T_0(\tilde{n}) \rangle + \langle T(0\rightarrow \tilde{n}) \rangle$')

    ax.set_title(r'Time scale species correlation $e^{-t/T_{exp}}$');
    plt.legend(); plt.xscale('log'); plt.yscale('log')
    plt.xlim(np.min(other), np.max(other)) #plt.ylim( np.min(,),np.max() )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(
                vmin=np.min(plotRange), vmax=np.max(plotRange)))
    clb = plt.colorbar(sm)
    clb.set_label(VAR_SYM_DICT[range], labelpad=-30, y=1.1, rotation=0)
    plt.xlabel(VAR_NAME_DICT[otherLabel]); plt.ylabel(r'times')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + range
                                    + '_' + 'species_timescale_cor' +'.pdf');
        #plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + 'maximums' +'.png');
        plt.close()
    else: plt.show()

    time_autocor_spec2D[i,:]        = time_autocor_spec[sim]
    mean_time_autocor_abund2D[i,j]  = mean_time_autocor_abund[sim]
    std_time_autocor_abund2D[i,j]   = std_time_autocor_abund[sim]
    dominance_turnover2D[i,j]       = dominance_turnover[sim]
    suppress_turnover2D[i,j]        = suppress_turnover[sim]
    dominance_return2D[i,j]         = dominance_return[sim]
    suppress_return2D[i,j]          = suppress_return[sim]

    ######################### plot abundance exponential #######################

    f = plt.figure(); fig = plt.gcf(); ax = plt.gca()
    cmap = mpl.cm.get_cmap("viridis")

    for j, element in enumerate(plotRange):
        color = cmap(j/len(plotRange ) )
        ax.errorbar(other, data['mean_time_autocor_abund2D'][i,:]
                        , yerr=data['std_time_autocor_abund2D'][i,:], fmt='-o'
                        , c=color)
        plt.scatter(other, data['suppress_turnover2D'][i,:], lw=2, c=color
                        , edgecolor='none', marker='o' )
        plt.scatter(other, data['dominance_turnover2D'][i,:], lw=2, c=color
                        , edgecolor='none', marker='x' )
    ax.errorbar(np.NaN, np.NaN,yerr=np.NaN, fmt='-o', color='silver'
                        , label=r'$T_{exp}$ : $e^{-t/T_{exp}}$')
    plt.scatter(np.NaN, np.Nan, lw=2, c='silver', edgecolor='none', marker='o'
                        , label=r'$\langle T(0\rightarrow 0) \rangle$')
    plt.scatter(np.NaN, np.Nan, lw=2, c='silver', edgecolor='none', marker='x'
                        , label=r'$\langle T(\tilde{n}\rightarrow \tilde{n}) \rangle$')

    ax.set_title(r'Time scale abundance correlation $e^{-t/T_{exp}}$')
    plt.legend(); plt.xscale('log'); plt.yscale('log')
    plt.xlim(np.min(other), np.max(other)) #plt.ylim( np.min(,),np.max() )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.LogNorm(
                vmin=np.min(plotRange), vmax=np.max(plotRange)))
    clb = plt.colorbar(sm)
    clb.set_label(VAR_SYM_DICT[range], labelpad=-30, y=1.1, rotation=0)
    plt.xlabel(VAR_NAME_DICT[otherLabel]); plt.ylabel(r'times')
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + range
                                    + '_' + 'abundance_timescale_cor'+'.pdf');
        #plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + 'maximums' +'.png');
        plt.close()
    else: plt.show()

    return 0

def plot_trajectory(dir, sim_nbr, nbr_steps, plot_dstbn=True, colour='red'
                        , save=False):
    """
    Plot species trajectories for the t=time last reactions. May also print the
    distribution on the side if needed.
    """
    with open(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
               'results_0.pickle', 'rb') as handle:
        data  = pickle.load(handle)

    # distribution
    ss_dist_sim     = data['results']['ss_distribution'] \
                            / np.sum(data['results']['ss_distribution'])

    # TODO : NEW BETTER WAY OF DOING IT
    trajectories = data['results']['trajectory'][-nbr_steps:]
    times        = data['results']['times'][-nbr_steps:]

    # DUMB OLD WAY
    #trajectories = np.loadtxt(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
    #                                'trajectory_0.txt')[-nbr_steps:]
    #times        = np.loadtxt(dir + os.sep + 'sim' + str(sim_nbr) + os.sep +
    #                                'trajectory_0_time.txt')[-nbr_steps:]


    if plot_dstbn: fig, (ax1, ax2) = plt.subplots(1, 2
                                        , gridspec_kw={'width_ratios': [3, 1]}
                                        , figsize=(4,2.5))
    else: fig, ax1 = plt.figure(figsize=(3.25,2.5))

    focus_species = 15
    for i in np.arange( np.shape( trajectories )[1] ):
        ax1.plot(times, trajectories[:,i], alpha=0.1, color='gray')
    ax1.plot(times, trajectories[:,focus_species], color='k')

    ax1.set_xlabel('time'); ax1.set_ylabel(r'abundance, $n$')
    ax1.set_xlim( left=min(times), right=max(times) )

    if plot_dstbn:
        ax2.plot(ss_dist_sim, np.arange(0,len( ss_dist_sim )), color=colour
                            , linewidth=3)
        ax2.set_xscale('log')
        ax2.set_xlabel(r'$P(n)$')

    # set 0 abundance as limit
    ax1.set_ylim(bottom=0, top=int(1.5*data['carry_capacity']) )
    ax2.set_ylim(bottom=0, top=int(1.5*data['carry_capacity']) )

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    ax2.label_outer()

    if save:
        plt.savefig(dir + os.sep + 'traj_' + str(sim_nbr) + '.pdf')
        #plt.savefig(MANU_FIG_DIR + os.sep + range + '_' + 'maximums' +'.png');
        plt.close()
    else: plt.show()

    return 0

def plot_average_n(filename):
    return


if __name__ == "__main__":

    while not os.path.exists( os.getcwd() + os.sep + MANU_FIG_DIR ):
        os.makedirs(os.getcwd() + os.sep + MANU_FIG_DIR);

    save = True
    plt.style.use('src/custom.mplstyle')

    #large_immi      = RESULTS_DIR + os.sep + 'multiLV22'
    sim_immi        = RESULTS_DIR + os.sep + 'multiLV71'
    sim_immi_inset  = RESULTS_DIR + os.sep + 'multiLV79'
    sim_spec        = RESULTS_DIR + os.sep + 'multiLV77'
    sim_corr        = RESULTS_DIR + os.sep + 'multiLV80'
    sim_time        = RESULTS_DIR + os.sep + 'multiLV6'
    sim_avJ         = RESULTS_DIR + os.sep + 'multiLVNavaJ'
    sim_K50         = RESULTS_DIR + os.sep + 'multiLV5'
    sim_K100        = RESULTS_DIR + os.sep + 'multiLV10'
    sim_K200        = RESULTS_DIR + os.sep + 'multiLV20'

    # Create appropriate npz file for the sim_dir
    #cdate.mlv_consolidate_sim_results( large_immi, 'nbr_species', 'comp_overlap')
    #cdate.mlv_consolidate_sim_results( sim_spec, 'nbr_species', 'comp_overlap')
    cdate.mlv_consolidate_sim_results( sim_immi, 'immi_rate', 'comp_overlap')
    #cdate.mlv_consolidate_sim_results( sim_avJ, 'immi_rate', 'comp_overlap')
    #cdate.mlv_consolidate_sim_results( sim_corr, 'immi_rate', 'comp_overlap')
    #cdate.mlv_consolidate_sim_results(sim_time, parameter1='immi_rate', parameter2='comp_overlap')
    #cdate.mlv_consolidate_sim_results( sim_K50, 'immi_rate', 'comp_overlap')
    #cdate.mlv_consolidate_sim_results( sim_K100, 'immi_rate', 'comp_overlap')
    #cdate.mlv_consolidate_sim_results( sim_K200, 'immi_rate', 'comp_overlap')

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
    #figure_modality_phases(sim_immi+os.sep+NPZ_SHORT_FILE, save)
    #figure_regimes(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')
    #figure_regimes(sim_corr+os.sep+NPZ_SHORT_FILE, save)
    #figure_regimes(sim_spec+os.sep+NPZ_SHORT_FILE, xlabel='nbr_species',xlog=False, xdatalim=(0,32), ydatalim=(0,40), save=save, pbx=16)
    figure_regimes(sim_immi+os.sep+NPZ_SHORT_FILE, save)
    #figure_richness_phases(sim_immi+os.sep+NPZ_SHORT_FILE, save)
    #figure_richness_phases(sim_K200+os.sep+NPZ_SHORT_FILE, save)
    #figure_modality_phases(sim_K50+os.sep+NPZ_SHORT_FILE, save)
    #figure_modality_phases(sim_immi+os.sep+NPZ_SHORT_FILE, save)
    #figure_modality_phases(sim_K200+os.sep+NPZ_SHORT_FILE, save)

    # Richness between experiments and models
    #compare_richness(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')

    # correlation business
    #fig_corr(sim_corr+os.sep+NPZ_SHORT_FILE, save, revision='80')
    #fig_timecorr(sim_time + os.sep + "sim1" + os.sep + "results_0.pickle")
    #fig3A(sim_immi+os.sep+NPZ_SHORT_FILE, save, ydatalim=(20,60), xdatalim=(0,40), revision='71')

    # MFPT
    #mfpt(sim_immi+os.sep+NPZ_SHORT_FILE, save=True, ydatalim=(20,60), xdatalim=(0,40), revision='71')

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
