import numpy as np
from scipy.sparse import dia_matrix, diags
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt


def ravel_index_from_true_indexes(x_idx, y_idx, Nx) :
    return y_idx*Nx + x_idx

def find_nearest_index(array, value, nearest_value=False):
    """
    # Inputs
    Takes an array and a value as arguments.

    # Ouputs
    Return the index of the nearest value. Also returns (nearest_idx, nearest_value) if nearest_value is True.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if nearest_value :
        return idx, array[idx]
    return idx

def omega_pq (p,q,h,E_nu,rho,Lx,Ly) :    #Calcul analytique des pulsations propres d'une plaque en appuis simple
    return np.sqrt(E_nu*h**2/(12*rho)) * ((p*np.pi/Lx)**2+(q*np.pi/Ly)**2)

def phi_pq (p,q,x,y, Lx, Ly) :  #Calcul analytique des déformées des modes d'une plaque en appuis simple
    """
    ## Inputs
    - p : numéro du mode selon x
    - q : numéro du mode selon y
    - x : arrayLike, vecteur des abscisses
    - y : arrayLike, vecteur des ordonnées

    ## Outputs
    - phi_pq : arrayLike, size (Nx,Ny), déformée du mode (p,q) en tous les points (x,y) du maillage
    """
    return np.sin(p*np.pi*x[:,np.newaxis]/Lx) * np.sin(q*np.pi*y[np.newaxis,:]/Ly)

def Bigidibig_matrice_totale(h = 2.8e-3, E_nu = 7291666666, rho = 400, Lx = 40e-2, Ly = 40e-2, T = 73.9, rho_l = 3.61 * 10**(-3), L = 0.65, B = 4*10**(-5), xinB = np.array([2.2,1.1,1.6,1.0,0.7,0.9,1.1,0.7,1.4])/100):
    """
    input:
    - h : hauteur de la table
    - E_nu : rapport de E sur (1 - nu**2)
    - rho : masse volumique de la table
    - Lx : longueur celon x de la table
    - Ly : longeur celon y de la table
    - T : tension de la corde
    - rho_l : masse linéique de la corde
    - L : longeur de la table
    - B : coef d'inharmonicité de la table
    - xinB : amortissement modaux des 9 premiers modes de la table 

    En premier ce code fait l'analyse modale de la table, en la supposant simplement supporté, puis de la corde, 
    puis mets tout ensemble pour entrer dans U-K

    output : 
     - M : matrice de toutes les masses modales (pour U-K)
     - M_inv : matrice inverse de toutes les masses modales (pour U-K)
     - C : matrice de tous les C modaux 
     - K : matrice de tous les K modaux
     - phiS_Nx_NmS : modes de la corde
     - phiB_NxNy_NmB : modes de la table
     - NmS : Nombre de mode de corde
     - NmB : Nombre de mode de la table
     - x : discrétisation en x de la table
     - y : discrétisation en y de la table
     - xS : discrétisation de la corde

    """
    ################## plaque : 

    ## Paramètres de discrétisation
    NB, MB = 3, 3 #Nombre de modes selon x, y
    NmB = NB * MB   #Nombre de modes total considéré dans le modèle de plaque

    dx, dy = 10e-3, 10e-3
    x, y = np.arange(0,Lx,dx), np.arange(0,Ly,dy)
    Nx, Ny = len(x), len(y)
    X_plate, Y_plate = np.meshgrid(x, y)
    X_ravel, Y_ravel = np.ravel(X_plate), np.ravel(Y_plate)

    ## Calcul des modes
 
    wnB = np.zeros(NmB)
    NmB_idx = np.zeros((2,NmB))   #Cette liste permet de remonter du mode contracté "i" au mode réel (n_i,m_i) en appelant NmB_idx[:,i]
    j = 0
    for n in range(1,NB+1) :
        for m in range(1,MB+1) :
            wnB[j] = omega_pq(n,m, h, E_nu,rho,Lx,Ly)
            NmB_idx[0,j] = n
            NmB_idx[1,j] = m
            j += 1

    ### Tri par ordre de fréquences croissantes
    tri_idx = np.argsort(wnB)

    wnB = wnB[tri_idx]    #On range les pulsations par ordre croissant 
    NmB_idx = NmB_idx[:,tri_idx]      #On ordonne les modes par ordre croissant

    ### Déformées
   
    phiB_NxNy_NmB = np.zeros((Nx*Ny,NmB)) #Matrice des déformées avec les 2 dimensions spatiales applaties en 1 dimension
    for mode in range (NmB) :
        n, m = NmB_idx[0,mode], NmB_idx[1,mode]
        phiB_NxNy_NmB[:,mode] = phi_pq(n, m , x, y, Lx, Ly).ravel()

    ### Masses modales
    MmB = np.zeros(NmB)
    for j in range(NmB) :
        PHI_j_Ny_Nx = np.reshape(phiB_NxNy_NmB[:,j],(Nx,Ny))      #Correspond à la déformée du mode j sur la plaque (en 2D)
        MmB[j] = rho * h * np.sum(np.sum(PHI_j_Ny_Nx**2,axis=1),axis=0)*dx*dy

    ### Normalisation des modes
    norme_deformee_NmB = np.sqrt(MmB)         #Ref : Modal Testing Theory, Practice and Application p.54, Eq. (2.25)
    phiB_NxNy_NmB = phiB_NxNy_NmB[:,:] / norme_deformee_NmB[np.newaxis,:]

    MB = np.ones(NmB)
    MB_inv = MB 
    CB = 2 * MmB * wnB * xinB
    KB = MmB * wnB ** 2
    
    ################################## cordes
    ## Paramètres physique

    ct = np.sqrt(T / rho_l) #célérité des ondes transverse (M/s)

    ## Paramètres de discrétisation
    NmS = 75  #Modes de cordes
    NnS = np.arange(1,NmS+1)

    NxS = 1000 #Discrétisation spatiale
    xS = np.linspace(0,L,NxS) #Vecteur de la corde

    ## Calcul des modes
    phiS_Nx_NmS = np.sin((2*NnS[np.newaxis,:]-1)*np.pi*xS[:,np.newaxis] / 2 / L) #Déformées d'une corde fixe/libre
    pnS = (2 * NnS - 1) * np.pi / (2 * L)
    fnS = (ct / 2 / np.pi) * pnS * (1 + pnS**2 * B / (2 * T)) #Fréquences propres de la corde (hz)

    wnS = 2*np.pi*fnS

    etaf, etaA, etaB = 7e-5, 0.9, 2.5e-2
    xinS = 1/2 * ( T * (etaf + etaA / 2 / np.pi / fnS) + etaB * B*pnS**2 ) / (T + B*pnS**2) #Amortissements modaux de la corde (ø)

    MmS = rho_l * L / 2  #Masses modales de la corde (kg)
    phiS_Nx_NmS = phiS_Nx_NmS / np.sqrt(MmS)

    ### Matrices modales
    MS = np.ones(NmS) #* MmS
    CS = MS * 2 * wnS * xinS
    KS = MS * wnS ** 2
    MS_inv = np.ones(NmS)# * (1/MmS)

    ################# bigmatrix :

    M_lin = np.concatenate((MS,MB))
    C_lin = np.concatenate((CS,CB))
    K_lin = np.concatenate((KS,KB))
    M_inv_lin = np.concatenate((MS_inv,MB_inv))

    M = diags(M_lin)
    M_inv = diags(M_inv_lin)
    K = diags(K_lin)
    C = diags(C_lin)

    return(M,M_inv, C,K, phiS_Nx_NmS,phiB_NxNy_NmB,NmS,NmB,x,y,xS)

def UK_params(M,M_inv,NmS, NmB, phiS_Nx_NmS,phiB_NxNy_NmB,xS,article = True, model = False, mode = 'A1',x =0, y = 0):
    phiSB = phiS_Nx_NmS[-1,:] #déformée de la corde au point du chevalet
    phiSF = phiS_Nx_NmS[1,:] #déformée de la corde au point d'appuis du doigt du guitariste

    if model : 
        Nx = len(x)
        Ny = len(y)
        #Point de couplage (par rapport à la table)
        xc, yc = x[int(24.5/40*Nx)], y[int(10/26 * Ny)]
        xc_idx, yc_idx = find_nearest_index(x, xc), find_nearest_index(y, yc)
        xyc = ravel_index_from_true_indexes(xc_idx, yc_idx, Nx)
        #print(xyc)
        #pour modèle de la plaque:
        phiBS = phiB_NxNy_NmB[xyc,:]

    if article : 
        phiBS = phiB_NxNy_NmB
        xyc = 0

    if mode == 'A1':
        Aa = np.block([
                        [phiSB.T, np.zeros(NmB)],
                        [phiSF.T, np.zeros(NmB)]
        ])
    if mode == 'A2':
        #print(phiBS.shape)
        Aa = np.block([
                        [phiSB.T, - phiBS],
                        [phiSF.T, np.zeros(NmB)]
        ])


    M_inv_demi = M_inv.sqrt()

    B = Aa @ M_inv_demi
    Bplus = B.T @ np.linalg.inv((B @ B.T))
    W = np.eye(NmS+NmB) - M_inv_demi @ Bplus @ Aa

    Z = - M.sqrt() @ Bplus @ Aa #pour calculer la force ensuite

    #print(W,Z)
    return(W,Z,xyc)

def Simu_config(xS,Fe, Tps = 3):
    """
    entrée :
    xS : discrétisation de la corde
    Fe : fréquence d'échantillonnage
    T ; temps d'acquisition
    
    sortie : 
    t : vecteur de temps
    FextS_NxS_Nt : Force dans la matrice spatiale

    """
    # Vecteur temps
    # Fe = int(2.2*max(fnS[-1], fnB[-1])) #Fréquence d'échantillonnage (hz) (on prends un peu plus que la limite pour respecter Shannon pour optimiser)
    # Fe = 44100
    # print(f"Fréquence d'échantillonage : {Fe} Hz")
    # T = 10 #Temps d'acquisition (s)
    # print(f"Temps d'acquisition : {T} s")
    t = np.linspace(0, Tps, Tps*Fe) #Vecteur temps
    Nt = len(t)
    L = 0.65 #a changer dans la def de corde si on change

    # Force extérieure appliquée à la corde
    Fext = np.zeros_like(t)


    fm = 0.187
    t1 = int(4*0.016*Fe) #indice du temps où l'on lâche la corde
    t2 = t1 + int(0.5*1e-3*Fe) #indice du temps où la force repasse à 0 (fin du glissement du plectre sur la corde) : à modéliser, int(1/2*1e-3*Fe) pour le moment #CF thèse Grégoire Derveaux, eq. 1.34

    Fext[:t1] = fm/2 * (1 - np.cos(np.pi*t[:t1]/t[t1]))
    Fext[t1:t2] = fm/2 * (1 + np.cos(np.pi*(t[t1:t2]-t[t1])/(t[t2]-t[t1])))

    #Fext[idx_fin:idx_zero] = np.linspace(1,0,idx_zero - idx_fin) * 0.187 #Dans ce cas, Fext est une rampe
    xe_idx = find_nearest_index(xS, 0.9*L)
    NxS = len(xS)

    FextS_NxS_Nt = np.zeros((NxS,Nt))
    FextS_NxS_Nt[xe_idx, : ] = Fext

    plot_fext = True
    if plot_fext :
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.plot(t,Fext,label="")
        ax1.grid()
        #ax1.legend()
        ax1.set_xlabel("Temps (s)")
        ax1.set_ylabel("Force (N)")
        ax1.set_title(rf"Force extérieure appliquée au point $x_e$={xS[xe_idx]:.1f}")

        fig.tight_layout()

        plt.show()
    return(t,FextS_NxS_Nt)

def lounch_simu_article(t,FextS_NxS_Nt,phiS_Nx_NmS,NmS,NmB,M_inv,C,K,Z,W):
    F_pro_cor = phiS_Nx_NmS.T @ FextS_NxS_Nt # projection de la force
    point_temp = len(t)
    #print(NmS,NmB,point_temp,FextS_NxS_Nt.shape)
    #comme il n'y a pas de force extérieur sur la table, leur projection sur la base modale vaut 0

    F_pro_tot = np.zeros((NmS+NmB,point_temp)) 
    F_pro_tot[:NmS,:] = F_pro_cor

    #initialisation du shéma numérique de résolution
    q_temp = np.zeros((NmS + NmB, point_temp))
    q_dd_temp = np.zeros((NmS + NmB, point_temp))
    q_d_temp = np.zeros(NmS + NmB)
    q_d_temp_demi = np.zeros_like(q_d_temp)
    q_pour_f = np.zeros_like(q_temp)

    #shéma
    h = t[1] - t[0] #step d'intégration

    for i in range(point_temp - 1):
        q_temp[:,i+1] = q_temp[:,i] + h * q_d_temp + 0.5 * h**2 * q_dd_temp[:,i]
        q_d_temp_demi = q_d_temp + 0.5 * h * q_dd_temp[:,i]
        F_temp = - C.dot(q_d_temp_demi) - K.dot(q_temp[:,i+1]) + F_pro_tot[:,i+1]
        q_pour_f[:,i+1] = M_inv.dot(F_temp)

        q_dd_temp[:,i+1] = W @  q_pour_f[:,i+1]

        q_d_temp = q_d_temp + 0.5 * h * (q_dd_temp[:,i] + q_dd_temp[:,i+1])

    Q = q_dd_temp
    Q_pos = q_temp

    F_c = Z @ q_pour_f

    return(Q,F_c,Q_pos)

def Calcul_force(F_c,NmS,phiS_Nx_NmS):
    FS = F_c[:NmS,:]
    FS_NxS_Nt = phiS_Nx_NmS[-1,:] @ FS
    return(FS_NxS_Nt)

def Calcul_accel(Q,NmS,xyc,phiB_Nxy_NmB):
    QB = Q[NmS :,:]
    #print(xyc)
    QB_nT = phiB_Nxy_NmB[xyc,:] @ QB
    return(QB_nT)

def Calcul_force_2(F_c,NmS,phiB_Nxy_NmB,xyc):
    FS = F_c[NmS:,:]
    FS_NxS_Nt_2 = phiB_Nxy_NmB[xyc,:] @ FS
    return(FS_NxS_Nt_2)

def Main(T,rho_l,L,B,h,E_nu,rhoT,Lx,Ly,xinB,Fe):
    """
    input : 
    - T : tension de la corde
    - rho_l : masse linéique de la corde
    - L : longueur de la corde
    - B : coefficient d'inharmonicité de la corde
    - h : hauteur de la table
    - E_nu : rapport E/(1-nu**2) de la table
    - rhoT : masse volumique de la table
    - Lx : largueur celon x de la table
    - Ly : largeur celon y de la table
    - xinB : coef d'amortissement des 9 premiers modes de la table
    - Fe : fréquence d'échantillonage

    retun:
    - La force exercé au chevalet par la corde
    """

    M, M_inv, C,K, phiS_Nx_NmS, phiB_NxNy_NmB, NmS,NmB,x,y,xS = Bigidibig_matrice_totale(h = h,E_nu= E_nu, rho= rhoT,Lx= Lx,Ly= Ly,T= T, rho_l= rho_l,L= L ,B= B,xinB =  xinB)
    W,Z,xyc = UK_params(M,M_inv,NmS, NmB, phiS_Nx_NmS,phiB_NxNy_NmB,xS,article = False, model = True, mode = 'A1',x=x, y=y)
    t,FextS_NxS_Nt = Simu_config(xS,Fe, Tps = 3)
    Q, F_c, Q_pos= lounch_simu_article(t,FextS_NxS_Nt,phiS_Nx_NmS,NmS,NmB,M_inv,C,K,Z,W)
    F = Calcul_force(F_c,NmS,phiS_Nx_NmS)
    #F_2 = Calcul_force_2(F_c,NmS,phiB_NxNy_NmB,xyc)   
    Q_nT = Calcul_accel(Q,NmS,xyc,phiB_NxNy_NmB)
    Q_posnT = Calcul_accel(Q_pos,NmS,xyc,phiB_NxNy_NmB)
    return(Q_nT,Q_posnT,F,phiB_NxNy_NmB)
