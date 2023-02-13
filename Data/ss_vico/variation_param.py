import numpy as np
import variation_def as v
from pyDOE import lhs

def latin_hypercube_sample(n, dim, bounds):
    """
    Generate a Latin Hypercube sample of size n in dim dimensions, with bounds on each dimension.
    :param n: Number of samples to generate.
    :param dim: Number of dimensions.
    :param bounds: List of tuples specifying the lower and upper bounds for each dimension.
    """

    # Generate the Latin Hypercube sample
    samples = lhs(dim, samples=n, criterion='center')
    # Scale and translate the samples to the correct bounds
    for i in range(dim):
        samples[:, i] = samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    return samples

def param_variation(N_sample, dico_table, dico_corde, dico_exc) :
    """
    Ce code permet de générer N_sample de paramètres de corde et de table avec une variation autour des paramètres de référence des tables données en argument.

    ## Inputs
    - `N_sample` : nombre de paramètres de corde et de table à générer
    - `dico_table` : dictionnaire contenant les paramètres de la table (voir table.py)
    - `dico_corde` : dictionnaire contenant les paramètres de la corde (voir corde.py)
    - `dico_exc` : dictionnaire contenant les paramètres de l'excitation (voir exc_ref.py)

    ## Outputs
    - `N_sample` : nombre de paramètres de corde et de table générés
    - `table_dicos` : arrayLike, size N_sample, contenant les paramètres de la table sous forme de dictionnaire.
    - `corde_dicos` : arrayLike, size N_sample, contenant les paramètres de la corde sous forme de dictionnaire.
    """

    #Chargement des paramètres de la table
    rho_T = dico_table['rho_T']
    L_x, L_y, h = dico_table['L_x'], dico_table['L_y'], dico_table['h']
    E_nu = dico_table['E_nu']
    xinB = dico_table['xinB']

    #Chargement des paramètres de la corde
    T, rho_l, Lc, B = dico_corde['T'], dico_corde['rho_l'], dico_corde['Lc'], dico_corde['B']

    #Chargement des paramètres de l'excitation
    xexc_ratio = dico_exc['xexc_ratio']
    temps_desc = dico_exc['temps_desc']

    #Variations autour des paramètres de référence de la corde
    T_min = T - T * v.T_delta
    T_max = T + T * v.T_delta

    rho_l_min = rho_l - rho_l * v.rho_delta
    rho_l_max = rho_l + rho_l * v.rho_delta

    Lc_min = Lc - Lc * v.Lc_delta
    Lc_max = Lc + Lc * v.Lc_delta

    B_min =  B - B * v.B_delta
    B_max =  B + B * v.B_delta

    #Variations autour des paramètres de la table
    rhoT_min = rho_T - rho_T * v.rhoT_delta
    rhoT_max = rho_T + rho_T * v.rhoT_delta

    L_xmin = L_x - L_x * v.L_xdelta
    L_xmax = L_x + L_x * v.L_xdelta

    L_ymin = L_y - L_y * v.L_ydelta
    L_ymax = L_y + L_y * v.L_ydelta

    h_min = h - h * v.h_delta
    h_max = h + h * v.h_delta

    E_nu_min = E_nu - E_nu * v.E_nu_delta
    E_nu_max = E_nu - E_nu * v.E_nu_delta

    #Variations de l'excitation
    xexc_ratio_min = xexc_ratio - xexc_ratio * v.xexc_ratio_delta
    xexc_ratio_max = xexc_ratio + xexc_ratio * v.xexc_ratio_delta
    # temps_desc_min = temps_desc - temps_desc * v.temps_desc_delta
    # temps_desc_max = temps_desc + temps_desc * v.temps_desc_delta
    temps_desc_min = 1/2*1e-3
    temps_desc_max = 2*1e-3

    #Proportion de bruit
    noise_prop_max = 0.1
    noise_prop_min = 0.01

    #### Création des paramètres dataset

    bounds = [(T_min,T_max),(rho_l_min,rho_l_max),(Lc_min,Lc_max),(B_min,B_max),(h_min,h_max),(E_nu_min,E_nu_max),
            (rhoT_min,rhoT_max),(L_xmin,L_xmax),(L_ymin,L_ymax), (xexc_ratio_min,xexc_ratio_max), (temps_desc_min,temps_desc_max), (noise_prop_min,noise_prop_max)]

    param_dataset = latin_hypercube_sample(N_sample,len(bounds),bounds)

    #Passage sous forme de dictionnaire
    table_dicos = np.zeros(N_sample, dtype=object)
    corde_dicos = np.zeros(N_sample, dtype=object)
    exc_dicos = np.zeros(N_sample, dtype=object)
    noise_prop = np.zeros(N_sample)
    for i in range(N_sample):
        table_dicos[i] = {
            'rho_T' : param_dataset[i,6],
            'L_x' : param_dataset[i,7],
            'L_y' : param_dataset[i,8],
            'h' : param_dataset[i,4],
            'E_nu' : param_dataset[i,5],
            'xinB' : xinB,
        }
        corde_dicos[i] = {
            'T' : param_dataset[i,0],
            'rho_l' : param_dataset[i,1],
            'Lc' : param_dataset[i,2],
            'B' : param_dataset[i,3],
        }
        exc_dicos[i] = {
            'xexc_ratio' : param_dataset[i,9],
            'temps_desc' : param_dataset[i,10],
        }
        noise_prop[i] = param_dataset[i,11]

    # param_table_dico = {
    #     'rho_T' : param_dataset[:,6],
    #     'L_x' : param_dataset[:,7],
    #     'L_y' : param_dataset[:,8],
    #     'h' : param_dataset[:,4],
    #     'E_nu' : param_dataset[:,5],
    #     'xinB' : xinB,
    # }

    # param_corde_dico = {
    #     'T' : param_dataset[:,0],
    #     'rho_l' : param_dataset[:,1],
    #     'Lc' : param_dataset[:,2],
    #     'B' : param_dataset[:,3],
    # }

    return(N_sample,table_dicos, corde_dicos, exc_dicos, noise_prop)
