import numpy as np

def table(medium_1=False,medium_2=False,metal=False,plexi = False):
    if medium_1 : 
        rho_T = 677.9661016949153
        L_x = 0.3793939393939394
        L_y = 0.30048101265822785 #largeur de la table (m)
        h = 2.95e-3 
        E_nu = 3206458380.6503973
        xinB = np.array([0.088747,0.053920,0.112417,0.052442,0.053975,0.013130,0.013130,0.061390,0.037500])

    if medium_2 :
        rho_T = 711.3592995846896
        L_x = 0.40 
        L_y = 0.259  #largeur de la table (m)
        h = 6.04e-3 
        E_nu = 3214097891.0586696
        xinB = np.array([0.11253203, 0.22842412, 0.06633512, 0.05462377, 0.14303315,
        0.05724961, 0.05446982, 0.04318039, 0.05932118])

    if metal : 
        rho_T = 2623.244373467073
        L_x = 39.5e-2 #longueur de la table (m)
        L_y = 25.5e-2 #largeur de la table (m)
        h = 6.01e-3 #épaisseur de la table (m)       
        E_nu = 83238129955.9411
        xinB = np.array([0.06343621, 0.06040264, 0.05246916, 0.04143557, 0.04148383,
        0.02160338, 0.04902268, 0.04280855, 0.04109246])

    if plexi :
        rho_T =  814.4294729660583 
        L_x = 40e-2 #longueur de la table (m)
        L_y = 25.9e-2 #largeur de la table (m)
        h = 4.51e-3 #épaisseur de la table (m)
        E_nu = 1307742955.0439968  
        xinB = np.array([0.07150503, 0.03790774, 0.07357441, 0.03845619, 0.04045318,
        0.01324686, 0.03842954, 0.02031206, 0.02085583])

    return(rho_T,L_x,L_y,h,E_nu,xinB)