def table(composite=False,acier=False):
    if composite : 
        masseT = 236e-3
        L_x = 0.40 
        L_y = 0.259  #largeur de la table (m)
        h = 2.8e-3 
        nu = 0.2 ##mis au hasard
        ET = 7e9  ##mis au hasard
    if acier : 
        masseT = 236e-3
        L_x = 0.40 
        L_y = 0.259  #largeur de la table (m)
        h = 2.8e-3 
        nu = 0.2 ##mis au hasard
        ET = 7e9  ##mis au hasard  
    return(masseT,L_x,L_y,h,nu,ET)