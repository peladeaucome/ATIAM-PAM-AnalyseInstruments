import numpy as np

def table(medium_1=False,medium_2=False,metal=False,plexi = False):
    if medium_1 : 
        rho_T = 677.9661016949153
        L_x = 0.31636363636363635
        L_y = 0.2466060606060606 #largeur de la table (m)
        h = 2.95e-3 
        E_nu = 3457372289.45938
        xinB = np.array([0.088747/2,0.053920/4,0.112417,0.052442,0.053975,0.013130,0.013130,0.061390,0.037500])/5


    if medium_2 :
        rho_T = 711.3592995846896
        L_x = 0.4206060606060606 
        L_y = 0.24818181818181817  #largeur de la table (m)
        h = 6.04e-3 
        E_nu = 2576534079.3792567
        xinB = np.array([0.112532,0.066335,0.054624,0.143033,0.057250,0.043180,0.038542,0.066675,0.036730])/5

    if metal : 
        rho_T = 2623.244373467073
        L_x = 0.2921212121212121 #longueur de la table (m)
        L_y = 0.20563636363636364  #largeur de la table (m)
        h = 6.01e-3 #épaisseur de la table (m)       
        E_nu = 8187642687.6106825
        xinB = np.array([0.063436*10,0.060403,0.041436,0.041484,0.021603,0.034045,0.047832,0.039952,0.040259])/5

    if plexi :
        rho_T =  814.4294729660583 
        L_x = 0.40848484848484845 #longueur de la table (m)
        L_y =  0.28284848484848485 #largeur de la table (m)
        h = 4.51e-3 #épaisseur de la table (m)
        E_nu = 2957903678.604968 
        xinB = np.array([0.071505,0.037908,0.038456,0.040453,0.013247,0.020312,0.020856,0.038635,0.021269])/5


    return(rho_T,L_x,L_y,h,E_nu,xinB)