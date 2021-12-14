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

def propensities():
    current_state = np.array([50,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    immi_rate = 0.001
    comp_overlap = 1.0
    birth_rate = 2.0
    death_rate = 1.0
    carry_capacity = 100
    quadratic = 0.0
    emmi_rate = 0.0

    prop = np.zeros( len(current_state)*2 )
    prop2 = np.zeros( len(current_state)*2 )

    prop[::2] = ( immi_rate + current_state * ( birth_rate ) )
                    #- ( quadratic *
                    #( birth_rate - death_rate ) *
                    #(1.0 - comp_overlap ) * current_state +
                    #( comp_overlap * np.sum(current_state) ) )
                    #/ carry_capacity ) )
                    # birth + immigration
    prop[1::2] = ( current_state * ( death_rate #+ emmi_rate
                      #+ ( birth_rate - death_rate )*( 1.0
                      #- quadratic )*( (1.0 - comp_overlap ) * current_state
                      + ( birth_rate - death_rate )
                      * ( (1.0 - comp_overlap ) * current_state
                      + comp_overlap * np.sum(current_state) ) / carry_capacity ) )
                      # death + emmigration

    for i in np.arange(0,len(current_state)):
            prop2[i*2] = ( current_state[i] * ( birth_rate
                        - quadratic * ( current_state[i]
                        + comp_overlap*np.sum(
                        np.delete(current_state,i)))/carry_capacity )
                        + immi_rate)
                        # birth + immigration
            prop2[i*2+1] = (current_state[i] * ( death_rate + emmi_rate
                          + ( birth_rate - death_rate )*( 1.0
                          - quadratic )*(current_state[i]
                          + comp_overlap*np.sum(
                          np.delete(current_state,i)))/carry_capacity ) )
                          # death + emmigration

    #print(prop==prop2)
    return 0


def main():
    """
    t = np.linspace(0,20,1024)
    x = np.exp(-t**2)
    pl.plot(t[:200], autocorrelation(x)[:200],label='scipy fft')
    pl.plot(t[:200], autocorrelation2(x)[:200],label='direct autocorrelation')
    pl.plot(t[:200], autocorrelation3(x)[:200],label='numpy correlate')
    pl.plot(t[:200], autocorrelation4(x)[:200],label='stack exchange')
    pl.yscale('log')
    pl.legend()
    pl.show()
    """
    #propensities()
    ss = np.zeros(30)
    st = np.zeros(30)
    a = np.array([0,0,0,0,10,2,3,3,0])
    unique, counts = np.unique(a,return_counts=True)
    ss[unique.astype(int)] += 0.33*counts
    print(ss)

    for i in a:
        st[i] += 0.33
    print(ss)

if __name__=='__main__':
    main()


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

    if xlabel == 'nbr_species':
        start = 1
    else: start = 0

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

    POINTS_BETWEEN_X_TICKS = pbx; POINTS_BETWEEN_Y_TICKS = pby
    sim_rich_cat, mf_rich_cat, lines_rich, _, _ =\
        figure_richness_phases(filename, False, xlabel, ylabel, xlog, ylog, ydatalim, xdatalim
                            , pbx, pby)
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
    plt.ylabel(r'$ T(\tilde{x} \rightarrow 0) / T(\tilde{x} \rightarrow \tilde{x})$')
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
    plt.ylabel(r'$T(0 \rightarrow \tilde{x}) / T(0 \rightarrow 0)$ ')
    plt.xlabel(VAR_NAME_DICT[ylabel])
    if save:
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_suppression_loss_lineplot" + '.pdf');
        plt.savefig(MANU_FIG_DIR + os.sep + "fpt_ratio_suppression_loss_lineplot" + '.png');
    else:
        plt.show()
    plt.close()

    return 0
