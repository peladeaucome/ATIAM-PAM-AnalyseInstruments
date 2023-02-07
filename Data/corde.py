###c'est chaud entre ce qu'on a mesuré,  les bails de l'article et le reste
from numpy import pi

def parametre_corde(article = False, acier_1 = False,acier_2 = False):
    if article :
        T = 73.9 
        rho_l = (3.61) * 10**(-3)
        Lc = 0.65
        #r = 0.40 * 10e-3
        B =  4 * 10**(-5)  ##ici c'est le coef B d'inharmonicité de la corde
        
    if acier_1 : ## corde de ré

        T = 93.947707
        r = 0.71 * 1e-3
        Lc = 0.65
        rho = 7800 #(kg/m3) #de internet
        rho_l = rho *  (pi * r**2) 
        print(rho_l)
        B = 0.0001943572734201584 #ici module de young # trouvé sur internet

    if acier_2 : ##corde de A (la)

        T = 93.2612415
        r = 0.97 * 1e-3
        Lc = 0.65
        rho = 7800 #(kg/m3) #de internet
        rho_l = rho * (pi * r ** 2) 
        print(rho_l)       
        B = 0.0002581 #ici module de young # trouvé sur internet

    return(T,rho_l,Lc,B)

    