
from corde import parametre_corde
from table import table
import Variation as v
from pyDOE import lhs
#import numpy as np

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
    T, rho_l, Lc, r, B_E = parametre_corde(article = article_C, acier_1 = acier_1C, acier_2 = acier_2C)

    T_min = T - T * v.T_delta
    T_max = T + T * v.T_delta

    rho_l_min = rho_l - rho_l * v.rho_delta
    rho_l_max = rho_l + rho_l * v.rho_delta

    Lc_min = Lc - Lc * v.Lc_delta
    Lc_max = Lc + Lc * v.Lc_delta

    r_min = r - r * v.r_delta
    r_max = r + r * v.r_delta

    #calculé
    masseC_min = rho_l_min * Lc_min
    masseC_max = rho_l_max * Lc_max

    I_min =  masseC_min * r_min ** 2 / 2
    I_max = masseC_max * r_max **2 / 2

    if article_C : ##if article B_E est le coef d'inharmonicité
        B_E_min =  B_E - B_E * v.B_delta
        B_E_max =  B_E + B_E * v.B_delta

        E_corde_min = B_E_min / I_min
        E_corde_max = B_E_max / I_max

    if acier_1C or acier_2C : ## if acier, B_E est le modul de Young direct
        E_corde_min = B_E - B_E * v.E_delta
        E_corde_max = B_E + B_E * v.E_delta

    ########## table : 
    masseT, L_x ,L_y, h, E_nu, xinB = table(medium_1 = medium_1T ,medium_2 = medium_2T, metal = metal_T, plexi = plexi_T  )
    
    masseT_min = masseT - masseT * v.masseT_delta
    masseT_max = masseT + masseT * v.masseT_delta

    L_xmin = L_x - L_x * v.L_xdelta
    L_xmax = L_x + L_x * v.L_xdelta

    L_ymin = L_y - L_y * v.L_ydelta
    L_ymax = L_y + L_y * v.L_ydelta

    h_min = h - h * v.h_delta
    h_max = h + h * v.h_delta

    E_nu_min = E_nu - E_nu * v.E_nu_delta
    E_nu_max = E_nu - E_nu * v.E_nu_delta

    #calculé : 
    rhoT_min = masseT_min / (L_xmax * L_ymax * h_max)  #masse volumique composite
    rhoT_max = masseT_max / (L_ymin * h_min * L_xmin)  #masse volumique composite


    #### Création des paramètres dataset

    bounds = [(T_min,T_max),(rho_l_min,rho_l_max),(Lc_min,Lc_max), ( E_corde_min,E_corde_max),(I_min,I_max),(h_min,h_max),(E_nu_min,E_nu_max),
            (rhoT_min,rhoT_max),(L_xmin,L_xmax),(L_ymin,L_ymax)]

    param_dataset = latin_hypercube_sample(N_sample,10,bounds)
    return(N_sample,param_dataset,xinB)
