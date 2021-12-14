DICT_REVISION = {'71' :
                        { 'multimodal' : [(0,57),(0,58,),(1,57),(1,58),(2,57),(2,58),(3,58),(4,58),(5,58),(6,58),(7,58)]
                        },
                '78' :
                        {'bimodal' : [(0,0)]
                        ,'multimodal' : [0,0]
                        },
                '87' :
                        { 'multimodal' : [(0,37),(0,38,),(1,37),(1,38),(2,37),(2,38),(3,38),(4,38),(5,38),(6,38),(7,38)]
                        },

                }

import pickle
import numpy as np

def correlations_fix(model, filesave):

    print("Warning, previously MLV6 calculated coefficient of variation wrong")

    model.results['corr_ni_nj'] = 0.0
    model.results['coeff_ni_nj'] = 0.0
    model.results['corr_J_n']  = 0.0
    model.results['coeff_J_n']  = 0.0
    model.results['corr_Jminusn_n']  = 0.0
    model.results['coeff_Jminusn_n']  = 0.0
    for i in np.arange(0, model.nbr_species):
            var_J_n = ( ( np.sqrt( model.results['av_ni_sq_temp'][i]
                        - model.results['av_ni_temp'][i]**2 ) ) * (
                        np.sqrt( model.results['av_J_sq']
                        - model.results['av_J']**2 ) ) )
            var_Jminusn_n = ( ( np.sqrt( model.results['av_ni_sq_temp'][i]
                            - model.results['av_ni_temp'][i]**2 ) ) * (
                            np.sqrt( model.results['av_Jminusn_sq'][i]
                            - model.results['av_Jminusn'][i]**2 ) ) )
            # cov(J,n)
            cov_J_n = (model.results['av_J_n'][i] - model.results['av_ni_temp'][i]
                        * model.results['av_J'] )
            # cov(J-n,n)
            cov_Jminusn_n = (model.results['av_Jminusn_n'][i]
                            - model.results['av_ni_temp'][i] *
                            model.results['av_Jminusn'][i] )
            # coefficients of variation
            if model.results['av_J'] != 0.0 and model.results['av_ni_temp'][i] != 0.0:
                model.results['coeff_J_n']+= ( cov_J_n /( model.results['av_J']
                                            * model.results['av_ni_temp'][i] ) )
            if model.results['av_Jminusn'][i] != 0.0 and model.results['av_ni_temp'][i] != 0.0:
                model.results['coeff_Jminusn_n'] += ( cov_Jminusn_n
                                            / ( model.results['av_Jminusn'][i]
                                            * model.results['av_ni_temp'][i] ) )
            # Pearson correlation
            if var_J_n != 0.0:
                model.results['corr_J_n'] += cov_J_n / var_J_n
            if var_Jminusn_n != 0.0:
                model.results['corr_Jminusn_n'] += cov_Jminusn_n / var_Jminusn_n

            for j in np.arange(i+1, model.nbr_species):
                var_nm = ( ( np.sqrt( model.results['av_ni_sq_temp'][i]
                - model.results['av_ni_temp'][i]**2 ) ) * (
                np.sqrt( model.results['av_ni_sq_temp'][j]
                - model.results['av_ni_temp'][j]**2 ) ) )
                # cov(n_i,n_j)
                cov_ni_nj = ( model.results['av_ni_nj_temp'][i][j]
                            - model.results['av_ni_temp'][i] *
                            model.results['av_ni_temp'][j] )
                # coefficients of variation
                if model.results['av_ni_nj_temp'][i][j] != 0.0:
                    model.results['coeff_ni_nj'] += ( cov_ni_nj
                                    / ( model.results['av_ni_temp'][i]
                                    * model.results['av_ni_temp'][j] ) )
                # Pearson correlation
                if var_nm != 0.0:
                    model.results['corr_ni_nj'] += cov_ni_nj / var_nm

    # Taking the average over all species
    model.results['corr_ni_nj'] /= ( model.nbr_species*(model.nbr_species-1)/2)
    model.results['coeff_ni_nj'] /= ( model.nbr_species*(model.nbr_species-1)/2)
    model.results['corr_J_n'] /= ( model.nbr_species)
    model.results['coeff_J_n'] /= ( model.nbr_species)
    model.results['corr_Jminusn_n'] /= ( model.nbr_species)
    model.results['coeff_Jminusn_n'] /= ( model.nbr_species)

    with open(filesave, 'wb') as handle:
        pickle.dump(model.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0
