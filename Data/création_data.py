
from corde import parametre_corde
from table_corrigé_2 import table
# from table import table
import Variation as v
from pyDOE import lhs
import numpy as np

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

def param_Dataset(N_sample = 1000,article_C= False,acier_1C = False, acier_2C = False, medium_1T = False, medium_2T= False, metal_T = False,plexi_T = False):

    ######################corde : 
    T, rho_l, Lc, B = parametre_corde(article = article_C, acier_1 = acier_1C, acier_2 = acier_2C)

    T_min = T - T * v.T_delta
    T_max = T + T * v.T_delta

    rho_l_min = rho_l - rho_l * v.rho_delta
    rho_l_max = rho_l + rho_l * v.rho_delta

    Lc_min = Lc - Lc * v.Lc_delta
    Lc_max = Lc + Lc * v.Lc_delta

    B_min =  B - B * v.B_delta
    B_max =  B + B * v.B_delta

    ########## table : 
    rhoT, L_x ,L_y, h, E_nu, xinB = table(medium_1 = medium_1T ,medium_2 = medium_2T, metal = metal_T, plexi = plexi_T  )
    
    rhoT_min = rhoT - rhoT * v.rhoT_delta
    rhoT_max = rhoT + rhoT * v.rhoT_delta

    L_xmin = L_x - L_x * v.L_xdelta
    L_xmax = L_x + L_x * v.L_xdelta

    L_ymin = L_y - L_y * v.L_ydelta
    L_ymax = L_y + L_y * v.L_ydelta

    h_min = h - h * v.h_delta
    h_max = h + h * v.h_delta

    E_nu_min = E_nu - E_nu * v.E_nu_delta
    E_nu_max = E_nu - E_nu * v.E_nu_delta

    #### Création des paramètres dataset

    bounds = [(T_min,T_max),(rho_l_min,rho_l_max),(Lc_min,Lc_max),(B_min,B_max),(h_min,h_max),(E_nu_min,E_nu_max),
            (rhoT_min,rhoT_max),(L_xmin,L_xmax),(L_ymin,L_ymax)]

    param_dataset = latin_hypercube_sample(N_sample,9,bounds)
    return(N_sample,param_dataset,xinB)
