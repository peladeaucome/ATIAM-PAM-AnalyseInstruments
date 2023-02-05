def table(medium_1=False,medium_2=False,metal=False,plexi = False):
    if medium_1 : 
       # masseT = 236e-3
        rho_T = 677.9661016949153
        L_x = 0.40204081632653066
        L_y = 0.2766326530612245  #largeur de la table (m)
        h = 0.00395
        E_nu = 3206458380.6503973
        xinB = [0.088747, 0.030742, 0.112417, 0.052442, 0.053975, 0.01313 ,0.042184, 0.06139 , 0.0375]

    if medium_2 :
        #masseT = 507e-3 #masse
        rho_T = 711.3592995846896
        L_x = 0.49592
        L_y = 0.23582  #largeur de la table (m)
        h = 0.00504 
        E_nu = 3214097891.0586696
        xinB = [0.112532, 0.228424, 0.054624, 0.143033, 0.05725 , 0.04318 , 0.066675, 0.065191, 0.03673]

    if metal : 
        #masseT = 1588e-3 #masse
        rho_T = 2623.244373467073
        L_x = 0.4786734693877551 #longueur de la table (m)
        L_y = 0.3305102040816326 #largeur de la table (m)
        h = 0.00501 #épaisseur de la table (m)       
        E_nu = 83238129955.9411 
        xinB = [0.063436, 0.060403, 0.041436, 0.041484, 0.021603, 0.034045, 0.047832, 0.039952, 0.040259]

    if plexi :
        #masseT = 382e-3 #masse
        rho_T =  814.4294729660583 
        L_x = 0.3979591836734694 #longueur de la table (m)
        L_y = 0.2620408163265306 #largeur de la table (m)
        h = 0.006010000000000001 #épaisseur de la table (m)
        E_nu = 1307742955.0439968  
        xinB = [0.071505, 0.037908, 0.038456, 0.040453, 0.013247, 0.020312,0.014682, 0.038635, 0.021269]

    return(rho_T,L_x,L_y,h,E_nu,xinB)