
from corde import parametre_corde
from table import table
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
    # Normalize the bounds
    normalized_bounds = [(b[0] - (b[1] - b[0]) / 2, (b[1] - b[0]) / 2) for b in bounds]

    # Generate the Latin Hypercube sample
    samples = lhs(dim, samples=n, criterion='centermaximin')

    # Scale and translate the samples to the correct bounds
    for i in range(dim):
        samples[:, i] = samples[:, i] * normalized_bounds[i][1] + normalized_bounds[i][0]

    return samples

def param_Dataset(N_sample = 1000,article= False,acier_C=False,composite_T = False, acier_T = False):

    ######################corde : 
    T,rho_l,Lc,r,B = parametre_corde(article=article, acier=acier_C)

    T_min = T - v.T_delta
    T_max = T + v.T_delta

    rho_l_min = rho_l - v.rho_delta
    rho_l_max = rho_l + v.rho_delta

    Lc_min = Lc - v.Lc_delta
    Lc_max = Lc + v.Lc_delta

    r_min = r - v.r_delta
    r_max = r + v.r_delta

    B_min =  B - v.B_delta
    B_max =  B + v.B_delta

    #calculé
    masseC_min = rho_l_min * Lc_min
    masseC_max = rho_l_max * Lc_max

    I_min =  masseC_min * r_min ** 2 / 2
    I_max = masseC_max * r_max **2 / 2

    E_corde_min = B_min / I_min
    E_corde_max = B_max / I_max

    ########## table : 
    masseT,L_x,L_y,h,nu,ET = table(composite = composite_T, acier = acier_T)
    
    masseT_min = masseT - v.masseT_delta
    masseT_max = masseT + v.masseT_delta

    L_xmin = L_x - v.L_xdelta
    L_xmax = L_x + v.L_xdelta

    L_ymin = L_y - v.L_ydelta
    L_ymax = L_y + v.L_ydelta

    h_min = h - v.h_delta
    h_max = h + v.h_delta

    nu_min = nu - v.nu_delta
    nu_max = nu - v.nu_delta

    ET_min = ET - v.ET_delta
    ET_max = ET + v.ET_delta

    #calculé : 
    rhoT_min = masseT_min / (L_xmax * L_ymax * h_max)  #masse volumique composite
    rhoT_max = masseT_max / (L_ymin * h_min * L_xmin)  #masse volumique composite


    #### Création des paramètres dataset

    bounds = [(T_min,T_max),(rho_l_min,rho_l_max),(Lc_min,Lc_max), ( E_corde_min,E_corde_max),(I_min,I_max),(h_min,h_max),(nu_min,nu_max),
           (ET_min,ET_max),(rhoT_min,rhoT_max),(L_xmin,L_xmax),(L_ymin,L_ymax)]

    param_dataset = latin_hypercube_sample(N_sample,11,bounds)
    return(N_sample,param_dataset)
