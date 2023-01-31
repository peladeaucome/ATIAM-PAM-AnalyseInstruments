###c'est chaud entre ce qu'on a mesuré,  les bails de l'article et le reste
def parametre_corde(article = False, acier_1 = False):
    if article:
        T = 73.9 
        rho_l = (3.61) * 10**(-3)
        Lc = 0.65
        r = 0.40 * 10e-4
        B =  4 * 10**(-5) 
    if acier_1: 
        T = 87.27
        rho_l = 4.80e-5 ##bizare
        Lc = 0.65
        r = 0.28 * 1e-4
        B = 1 #jsp
    return(T,rho_l,Lc,r,B)

    