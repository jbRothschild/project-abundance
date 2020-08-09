from matplotlib import colors

VAR_NAME_DICT = { 'birth_rate'      : r'$r^{+}$'
                , 'death_rate'      : r'$r^{-}$'
                , 'immi_rate'       : r'$\mu$'
                , 'emmi_rate'       : r'$\delta$'
                , 'carry_capacity'  : r'$K$'
                , 'comp_overlap'    : r'$\rho$'
                , 'nbr_species'     : r'$S$'
}

COLOURS = {   'time'            : 'g'
            , 'population'      : 'r'
            , 'metric'          : 'b'
            , 'time_h'          : 'viridis'
            , 'metric_h'        : 'cividis'
            , 'population'      : 'magma'
            , 'compare_zero_h'  : 'RdBu'
            , 'compare_h'       : 'YlGnBu'

}

IMSHOW_KW = { 'entropy'                 : {'cmap': 'cividis', 'aspect': None }
                , 'Gini-Simpson index'  : {'cmap': 'cividis', 'aspect': None }
                ,  r'$S(1-P(0))$'       : {'cmap': 'magma', 'aspect': None }
                ,  r'Method 2 richness' : {'cmap': 'magma', 'aspect': None }
                , r'$\langle S \rangle$': {'cmap': 'magma', 'aspect': None }
                , r'$S(1-P(0))/\langle S \rangle$'  : {'cmap': 'RdBu'
                                                ,'aspect': None , 'vmin' : 0, 'vmax': 2}
                , 'Jensen-Shannon Divergence'       : {'cmap': 'coolwarm', 'aspect': None
                                                , 'vmin': 0.0 , 'vmax' : 1.0}
                , r'LV mean / $\langle n \rangle_{sim}$' : {'cmap': 'RdBu'
                                                ,'aspect': None , 'vmin' : 0, 'vmax': 2}
                , r'LV mean $S(1-P(0))$ / $\langle n \rangle_{sim}$' : {'cmap': 'RdBu'
                                                ,'aspect': None , 'vmin' : 0, 'vmax': 2}
                , r'LV mean $(S(1-P(0)))$ / LV mean $S$' : {'cmap': 'YlGnBu'
                                                , 'aspect': None , 'vmin' : 0, 'vmax': 30}
                , r'Local maxima'               : { 'cmap': colors.ListedColormap(['green','blue','yellow'])
                                                , 'aspect': None }

}

imshow_kw = {'cmap': 'YlGnBu', 'aspect': None }

# TODO : save txt file with what parameters vary (is this possible?)

# TODO : save entropy and stuff in theory equations
