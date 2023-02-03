###c'est chaud entre ce qu'on a mesuré,  les bails de l'article et le reste
def parametre_corde(article = False, acier_1 = False,acier_2 = False):
    if article:
        T = 73.9 
        rho_l = (3.61) * 10**(-3)
        Lc = 0.65
        r = 0.40 * 10e-3
        B_E =  4 * 10**(-5)  ##ici c'est le coef B d'inharmonicité de la corde
        
    if acier_1 : 

        T = 87.27
        r = 0.28 * 1e-3
        Lc = 0.65
        rho = 7800 #(kg/m3) #de internet
        rho_l = rho *  (3.1415 * r**2) 

        B_E = 210e9 #ici module de young # trouvé sur internet

    if acier_2 :

        T = 79.34
        r = 0.36 * 1e-3
        Lc = 0.65
        rho = 7800 #(kg/m3) #de internet
        rho_l = rho * (3.1415 * r ** 2)
        print(rho_l)
        
        B_E = 210e9 #ici module de young # trouvé sur internet

    return(T,rho_l,Lc,r,B_E)

    