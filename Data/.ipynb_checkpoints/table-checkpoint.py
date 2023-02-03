def table(composite=False,acier=False,plexi = False):
    if composite : 
        masseT = 236e-3
        L_x = 0.40 
        L_y = 0.259  #largeur de la table (m)
        h = 2.8e-3 
        nu = 0.2 ##mis au hasard
        ET = 7e9  ##mis au hasard
        
    if acier : 
        masseT = 1588e-3 
        L_x = 40e-2, #longueur de la table (m)
        L_y = 25.9e-2, #largeur de la table (m)
        h = 6e-3, #épaisseur de la table (m)       
        nu = 0.33, #coefficient de Poisson (Pa)
        ET = 210e9, #Module d'Young (Pa)

    if plexi :
        masseT = 362e-3 #masse volumique (kg/m3)
        L_x = 40e-2, #longueur de la table (m)
        L_y = 25.9e-2, #largeur de la table (m)
        h = 3.5e-3, #épaisseur de la table (m)
        nu  = 0.2 #AU HASARD #coefficient de Poisson (Pa)
        ET  = 7.9 #AU HASARD #Module d'Young (Pa)

    return(masseT,L_x,L_y,h,nu,ET)