def parametre_corde(article = False, acier = False):
    if article:
        T = 73.9 - 0.5
        rho_l = (3.61) * 10**(-3)
        Lc = 0.65
        r = 0.40 * 10e-4
        B =  4 * 10**(-5) 
    if acier: 
        T= 12

    return(T,rho_l,Lc,r,B)

    