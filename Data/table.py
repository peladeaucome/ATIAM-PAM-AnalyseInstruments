import numpy as np

def table(medium_1=False,medium_2=False,metal=False,plexi = False):
    if medium_1 : 
        masseT = 236e-3
        L_x = 0.40 
        L_y = 0.259  #largeur de la table (m)
        h = 2.95e-3 
        E_nu = 2762549286.9772286
        xinB = np.array([0.08874667, 0.03074233, 0.05392   , 0.11241739, 0.05244179,
        0.05397483, 0.06034859, 0.06737623, 0.01312974])

    if medium_2 :
        masseT = 507e-3 #masse
        L_x = 0.40 
        L_y = 0.259  #largeur de la table (m)
        h = 6.04e-3 
        E_nu = 2787725397.361163
        xinB = np.array([0.11253203, 0.22842412, 0.06633512, 0.05462377, 0.14303315,
        0.05724961, 0.05446982, 0.04318039, 0.05932118])

    if metal : 
        masseT = 1588e-3 #masse
        L_x = 39.5e-2 #longueur de la table (m)
        L_y = 25.5e-2 #largeur de la table (m)
        h = 6.01e-3 #épaisseur de la table (m)       
        E_nu = 11696632410.872404
        xinB = np.array([0.06343621, 0.06040264, 0.05246916, 0.04143557, 0.04148383,
        0.02160338, 0.04902268, 0.04280855, 0.04109246])

    if plexi :
        masseT = 382e-3 #masse 
        L_x = 40e-2 #longueur de la table (m)
        L_y = 25.9e-2 #largeur de la table (m)
        h = 4.51e-3 #épaisseur de la table (m)
        E_nu = 2285967714.1443543 
        xinB = np.array([0.07150503, 0.03790774, 0.07357441, 0.03845619, 0.04045318,
        0.01324686, 0.03842954, 0.02031206, 0.02085583])

    return(masseT,L_x,L_y,h,E_nu,xinB)