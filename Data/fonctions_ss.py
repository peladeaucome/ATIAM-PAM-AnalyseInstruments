import numpy as np
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

def omega_pq (p,q, h, E_nu, rho, Lx, Ly) :
        return np.sqrt(E_nu*h**2/(12*rho)) * ((p*np.pi/Lx)**2+(q*np.pi/Ly)**2)

def phi_pq (p,q,x,y, Lx, Ly) :  #Calcul analytique des déformées des modes d'une plaque en appuis simple
        return np.sin(p*np.pi*x/Lx)*np.sin(q*np.pi*y/Ly)

def bigdickenergy_ss(h, E_nu, rho, Lx, Ly, T, rho_l, L, E_c, I, xinB) :
    #============================================= TABLE =============================================
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
    #print(f"Fréquence du dernier mode de plaque calculé : {fnB[-1]:.0f} Hz")

    ### Déformées
    phiB_NxNy_NmB = np.zeros((Nx*Ny,NmB)) #Matrice des déformées avec les 2 dimensions spatiales applaties en 1 dimension
    for mode in range (NmB) :
        for point in range(Nx*Ny) :
            n = NmB_idx[0,mode]
            m = NmB_idx[1,mode]
            x_ = X_ravel[point]
            y_ = Y_ravel[point]

            phiB_NxNy_NmB[point,mode] = phi_pq(n, m , x_, y_,Lx,Ly)

    ### Masses modales
    MmB = np.zeros(NmB)
    for j in range(NmB) :
        PHI_j_Ny_Nx = np.reshape(phiB_NxNy_NmB[:,j],(Ny,Nx))      #Correspond à la déformée du mode j sur la plaque (en 2D)
        MmB[j] = rho*h* np.sum(np.sum(PHI_j_Ny_Nx**2,axis=1),axis=0)*dx*dy

    ### Normalisation des masses modales
    norme_deformee_NmB = np.sqrt(MmB)         #Ref : Modal Testing Theory, Practice and Application p.54, Eq. (2.25)
    phiB_NxNy_NmB /= norme_deformee_NmB[np.newaxis,:]

    MB = np.ones(NmB)
    MBinv = MB
    CB = np.ones(NmB)*2*MmB*wnB*xinB
    KB = np.ones(NmB)*MmB*wnB**2
    
    #============================================= CORDE =============================================
    ct = np.sqrt(T / rho_l) #célérité des ondes transverse (M/s)
    B = E_c * I

    ## Paramètres de discrétisation
    NmS = 75  #Modes de cordes
    NnS = np.arange(1,NmS+1)

    NxS = 1000 #Discrétisation spatiale
    xS = np.linspace(0,L,NxS) #Vecteur de la corde

    ## Calcul des modes
    phiS_Nx_NmS = np.sin((2*NnS[np.newaxis,:]-1)*np.pi*xS[:,np.newaxis] / 2 / L) #Déformées d'une corde fixe aux extrémités
    pnS = (2 * NnS - 1) * np.pi / (2 * L)
    fnS = (ct / 2 / np.pi) * pnS * (1 + pnS**2 * B / (2 * T)) #Fréquences propres de la corde (hz)
    #print(f"Fréquence du dernier mode de corde calculé : {fnS[-1]:.0f} Hz")
    wnS = 2*np.pi*fnS

    etaf, etaA, etaB = 7e-5, 0.9, 2.5e-2
    xinS = 1/2 * ( T*(etaf + etaA / 2 / np.pi / fnS) + etaB * B*pnS**2 ) / (T + B*pnS**2) #Amortissements modaux de la corde (ø)

    MmS = rho_l * L / 2  #Masses modales de la corde (kg)

    ### Matrices modales
    MS = np.ones(NmS)*MmS
    MSinv = np.ones(NmS)*1/MmS
    CS = MS * np.ones(NmS)*2*wnS*xinS
    KS = MS*np.ones(NmS)*wnS**2

    M_lin = np.concatenate((MS,MB))
    M_inv_lin = np.concatenate((MSinv,MBinv))
    M = np.diag(M_lin)
    M_inv = np.diag(M_inv_lin)

    return (M, M_inv, np.diag(MBinv), np.diag(MSinv), np.diag(MB),np.diag(MS), np.diag(KB),np.diag(KS), np.diag(CB),np.diag(CS), phiS_Nx_NmS, phiB_NxNy_NmB, NmS, NmB, x, y, xS)


def UK_params(M,M_inv,NmS, NmB, phiS_Nx_NmS,phiB_NxNy_NmB,xS, article = True, model = False, mode = 'A1',x =0, y = 0):
    phiSB = phiS_Nx_NmS[-1,:] #déformée de la corde au point du chevalet
    phiSF = phiS_Nx_NmS[int(len(xS)/4),:] #déformée de la corde au point d'appuis du doigt du guitariste

    if model : 
        Nx = len(x)
        Ny = len(y)
        #Point de couplage (par rapport à la table)
        xc, yc = x[int(24.5/40*Nx)], y[Ny//2]
        xc_idx, yc_idx = find_nearest_index(x, xc), find_nearest_index(y, yc)
        xyc = ravel_index_from_true_indexes(xc_idx, yc_idx, Nx)
        #print(xyc)
        #pour modèle de la plaque:
        phiBS = phiB_NxNy_NmB[xyc,:]

    if article : 
        phiBS = phiB_NxNy_NmB

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

    M_inv_demi = np.sqrt(M_inv)

    B = Aa @ M_inv_demi
    Bplus = B.T @ np.linalg.inv((B @ B.T))
    W = np.eye(NmS+NmB) - M_inv_demi @ Bplus @ Aa

    Z = - np.sqrt(M) @ Bplus @ Aa #pour calculer la force ensuite

    return(W,Z)

def Simu_config(xS,Fe, T = 3):
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
    t = np.linspace(0, T, T*Fe) #Vecteur temps
    Nt = len(t)
    L = 0.65

    # Force extérieure appliquée à la corde
    Fext = np.zeros_like(t)
    idx_deb = 0
    idx_fin = int(0.4*1e-3*Fe)
    Fext[idx_deb:idx_fin] = np.linspace(0,1,idx_fin - idx_deb) * 0.187 #Dans ce cas, Fext est une rampe

    xe_idx = find_nearest_index(xS, 0.9*L)
    NxS = len(xS)

    FextS_NxS_Nt = np.zeros((NxS,Nt))
    FextS_NxS_Nt[xe_idx, : ] = Fext

    plot_fext = False
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

import control
def launch_simu_ss(t, FextS_NxS_Nt, phiS_Nx_NmS, NmS, NmB, MBinv, MSinv, KS, KB, CS, CB, W) :
    """
    Ce code permet de configurer l'espace d'état du modèle et de lancer une simulation temporelle.

    # Entrées
    - Configuration de guitare préalablement établie dans "guitare_config.py"
    - Paramètres issue de la formulation d'Udwadia-Kalaba établis dans "UK_parameters.py"
    - Paramètres de simulation : force extérieure "FextS_NxS_Nt", vecteur temps de simulation "t"

    # Sortie
    - Vecteur Q de dimension (NmS+NmB, Nt) : représente l'évolution des participations modales au cours du temps de la position de la corde sur les NmS premières coordonnées, et de la table sur les NmB dernières coordonnées.
    """

    ABG = W @ np.block([
        [-MSinv @ KS, np.zeros((NmS,NmB))],
        [np.zeros((NmB, NmS)), -MBinv @ KB]
    ])
    ABD = W @ np.block([
        [-MSinv @ CS, np.zeros((NmS,NmB))],
        [np.zeros((NmB, NmS)), -MBinv @ CB]
    ])

    A = np.block([
        [np.zeros((NmS+NmB,NmS+NmB)) , np.eye(NmS+NmB)],
        [ ABG         , ABD      ]
    ])

    B = np.block([
        [np.zeros((NmS+NmB, NmS))],
        [W @ np.block([
            [MSinv],
            [np.zeros((NmB, NmS))]
            ])]
    ])

    #Pour observer la position
    C = np.block([
        [np.eye(NmS+NmB) ,  np.zeros((NmS+NmB,NmS+NmB))]
    ])

    D = 0

    sys = control.StateSpace(A,B,C,D)

    U = phiS_Nx_NmS.T @ FextS_NxS_Nt

    t, Q = control.forced_response(sys, T=t, U=U, X0=0)

    return Q