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
    """
    ## Inputs
    - p : numéro du mode selon x
    - q : numéro du mode selon y
    - x : arrayLike, vecteur des abscisses
    - y : arrayLike, vecteur des ordonnées

    ## Outputs
    - phi_pq : arrayLike, size (Nx,Ny), déformée du mode (p,q) en tous les points (x,y) du maillage
    """
    return np.sin(p*np.pi*x[:,np.newaxis]/Lx)*np.sin(q*np.pi*y[np.newaxis,:]/Ly)

def modal_config(table_dico, corde_dico, plot_deformee=False) :
    """
    Ce code permet de calculer les configurations modales d'une table et d'une corde donnée.

    ## Inputs
    - `table_dico` : dictionnaire contenant les paramètres de la table. Voir le fichier table_corrigé.py pour un exemple.
    - `corde_dico` : dictionnaire contenant les paramètres de la corde. Voir le fichier corde.py pour un exemple.
    - `plot_deformee` : booléen, si True, affiche la déformée de la table.

    ## Outputs
    - `M` : arrayLike, matrice de masse du système complet
    - `M_inv` : arrayLike, matrice inverse de M
    - `modal_config_table` : arrayLike, dictionnaire contenant toutes les informations importantes sur la configuration modale de la table
    - `modal_config_corde` : arrayLike, dictionnaire contenant toutes les informations importantes sur la configuration modale de la corde
    """

    #============================================= TABLE =============================================
    #Chargement des paramètres de table
    Lx, Ly, h = table_dico['L_x'], table_dico['L_y'], table_dico['h']
    E_nu, rho = table_dico['E_nu'], table_dico['rho_T']
    xinB = table_dico['xinB']

    ## Paramètres de discrétisation
    NB, MB = 3, 3 #Nombre de modes selon x, ys
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
    fnB = wnB/2/np.pi
    NmB_idx = NmB_idx[:,tri_idx]      #On ordonne les modes par ordre croissant
    #print(f"Fréquence du dernier mode de plaque calculé : {fnB[-1]:.0f} Hz")

    ### Déformées
    phiB_NxNy_NmB = np.zeros((Nx*Ny,NmB)) #Matrice des déformées avec les 2 dimensions spatiales applaties en 1 dimension
    for mode in range (NmB) :
        n, m = NmB_idx[0,mode], NmB_idx[1,mode]
        phiB_NxNy_NmB[:,mode] = phi_pq(n, m , x, y, Lx, Ly).ravel()

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

    ## Dictionnaire des paramètres de la table
    modal_config_table = {
        'Nx' : Nx,
        'Ny' : Ny,
        'dx' : dx,
        'dy' : dy,
        'x' : x,
        'y' : y,
        'wnB' : wnB,
        'fnB' : fnB,
        'NmB_idx' : NmB_idx,
        'phiB_NxNy_NmB' : phiB_NxNy_NmB,
        'MmB' : MmB,
        'MB' : np.diag(MB),
        'MB_inv' : np.diag(MBinv),
        'CB' : np.diag(CB),
        'KB' : np.diag(KB),
    }
    
    #============================================= CORDE =============================================
    #Chargement des paramètres de corde
    T, rho_l, L, B = corde_dico['T'], corde_dico['rho_l'], corde_dico['Lc'], corde_dico['B']

    ct = np.sqrt(T/rho_l) #célérité des ondes transverse (M/s)

    ## Paramètres de discrétisation
    NmS = 75  #Modes de cordes
    NnS = np.arange(1,NmS+1)

    NxS = 1000 #Discrétisation spatiale
    xS = np.linspace(0,L,NxS) #Vecteur de la corde

    ## Calcul des modes
    phiS_Nx_NmS = np.sin((2*NnS[np.newaxis,:]-1)*np.pi*xS[:,np.newaxis] / 2 / L) #Déformées d'une corde fixe aux extrémités
    pnS = (2 * NnS - 1) * np.pi / (2 * L)
    fnS = (ct / 2 / np.pi) * pnS * (1 + pnS**2 * B / (2 * T)) #Fréquences propres de la corde (hz)
    wnS = 2*np.pi*fnS

    etaf, etaA, etaB = 1*7e-5, 1*0.9, 1*2.5e-2 
    #etaA semble gérer plutot l'enveloppe temporelle générale à l'écoute, etaB semble ajuster plutôt l'amortissement des hautes fréquences
    #
    xinS = 1/2 * ( T*(etaf + etaA/2/np.pi/fnS) + etaB*B*pnS**2 ) / (T + B*pnS**2) #Amortissements modaux de la corde (ø)

    MmS = rho_l * L / 2  #Masses modales de la corde (kg)

    ### Matrices modales
    MS = np.ones(NmS)*MmS
    MSinv = np.ones(NmS)*1/MmS
    CS = MS * np.ones(NmS)*2*wnS*xinS
    KS = MS*np.ones(NmS)*wnS**2

    ## Dictionnaire des paramètres de la corde
    modal_config_corde = {
        'NxS' : NxS,
        'xS' : xS,
        'wnS' : wnS,
        'fnS' : fnS,
        'NmS' : NmS,
        'phiS_Nx_NmS' : phiS_Nx_NmS,
        'MmS' : MmS,
        'MS' : np.diag(MS),
        'MS_inv' : np.diag(MSinv),
        'CS' : np.diag(CS),
        'KS' : np.diag(KS),
    }

    M_lin = np.concatenate((MS,MB))
    M_inv_lin = np.concatenate((MSinv,MBinv))
    M = np.diag(M_lin)
    M_inv = np.diag(M_inv_lin)

    if plot_deformee :
        mode = 2
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        img = ax1.imshow(phiB_NxNy_NmB[:,mode].reshape(Nx,Ny).T,
            extent=[x[0], x[-1] , y[0], y[-1]] ,
            cmap="jet" ,
            interpolation = "bilinear",
            # aspect="auto" ,
            origin="lower")

        xc, yc = x[int(24.5/40*Nx)], y[int(10/26*Ny)]
        ax1.scatter(xc,yc,marker="x",color="k",s=100, label="Point de couplage avec la corde")
        ax1.legend()

        fig.colorbar(img,ax=ax1)
        ax1.set_xlabel("$x$")
        ax1.set_ylabel(r"$y$")
        ax1.set_title(fr"Defomée du mode {NmB_idx[:,mode]} de plaque ($f = {fnB[mode]:.0f}$ Hz)")

        fig.tight_layout()

        plt.show()

    # return (M, M_inv, np.diag(MBinv), np.diag(MSinv), np.diag(MB),np.diag(MS), np.diag(KB),np.diag(KS), np.diag(CB),np.diag(CS), phiS_Nx_NmS, phiB_NxNy_NmB, NmS, NmB, x, y, xS)
    return (M, M_inv, modal_config_table, modal_config_corde)

def UK_params(M, M_inv, modal_config_table, modal_config_corde, mode='A2', article=False) :
    """
    Cette fonction calcul les matrices utiles de la formulation d'Udwadia-Kalaba.

    ## Inputs
    - `M` : Matrice de masse totale
    - `M_inv` : Matrice inverse de masse totale
    - `modal_config_table` : dictionnaire des paramètres modaux de la table
    - `modal_config_corde` : dictionnaire des paramètres modaux de la corde
    - `mode` : si 'A2' (par défaut), on considère la vibration de la table. Si 'A1', on considère uniquement la vibration de la corde (alors fixe-fixe).
    - `article` : si True, on considère la vibration de la table comme dans l'article. Si False, on considère la vibration de la table comme dans le rapport de stage.

    ## Outputs
    - `W` : Matrice W de la formulation d'u-K
    - `Z` : Matrice Z de la formulation d'u-K
    - `xyc` : coordonnées du point de couplage entre la corde et la table
    """

    #Chargement des paramètres modaux utiles pour la table
    NmB = modal_config_table['NmB_idx'].shape[1]
    phiB_NxNy_NmB = modal_config_table['phiB_NxNy_NmB']
    x, y = modal_config_table['x'], modal_config_table['y']

    #Chargement des paramètres modaux utiles pour la corde
    NmS = modal_config_corde['NmS']
    phiS_Nx_NmS = modal_config_corde['phiS_Nx_NmS']

    #=========================================== Formulation UK =======================================================
    phiSB = phiS_Nx_NmS[-1,:] #déformée de la corde au point du chevalet
    phiSF = phiS_Nx_NmS[20,:] #déformée de la corde au point d'appuis du doigt du guitariste

    if article : 
        phiBS = phiB_NxNy_NmB

    else :
        Nx = len(x)
        Ny = len(y)
        
        #Point de couplage (par rapport à la table)
        xc, yc = x[int(24.5/40*Nx)], y[int(10/26*Ny)]
        xc_idx, yc_idx = find_nearest_index(x, xc), find_nearest_index(y, yc)
        xyc = ravel_index_from_true_indexes(xc_idx, yc_idx, Nx)
        phiBS = phiB_NxNy_NmB[xyc,:]

    if mode == 'A1':
        Aa = np.block([
                        [phiSB.T, np.zeros(NmB)],
                        [phiSF.T, np.zeros(NmB)]
        ])
    elif mode == 'A2':
        Aa = np.block([
                        [phiSB.T, - phiBS],
                        [phiSF.T, np.zeros(NmB)]
        ])

    M_inv_demi = np.sqrt(M_inv)
    B = Aa @ M_inv_demi
    Bplus = B.T @ np.linalg.inv((B @ B.T))
    W = np.eye(NmS+NmB) - M_inv_demi @ Bplus @ Aa
    Z = - np.sqrt(M) @ Bplus @ Aa #pour calculer la force ensuite
    return (W, Z, xyc)

def fext(xS, Fe, temps_desc=1*1e-3, xexc_ratio=(32.5+20)/65, L=65e-2, T=3, exc="gliss", plot_fext=False):
    """
    ## Inputs
    - `xS` : vecteur spatial de la corde
    - `Fe` : fréquence d'échantillonnage
    - `temps_desc` : temps de descente du plectre (en s), par défaut 1/2 ms
    - `xexc_ratio` : position de l'excitation par rapport à la longueur de la corde sous forme d'un ratio
    - `L` : longueur de la corde (en m)
    - `T` ; temps d'acquisition en secondes
    - `exc` : type d'excitation, par défaut glissé
    - `plot_fext` : booléen pour afficher la force extérieure appliquée à la corde. Par défaut False.
    
    ## Outputs
    - `t` : vecteur de temps
    - `FextS_NxS_Nt` : vecteur spatial de la force extérieure appliquée à la corde
    """

    #Vecteur temps
    t = np.linspace(0, T, T*Fe)
    Nt = len(t)

    #Force extérieure appliquée à la corde
    Fext = np.zeros_like(t)
    fm = 0.187 #d'après analyse force d'un plectre ulysse
    if exc == "rampe" :
        idx_deb = 0
        idx_fin = int(0.16*Fe)
        Fext[idx_deb:idx_fin] = np.linspace(0,1,idx_fin - idx_deb) * fm #Dans ce cas, Fext est une rampe

        idx_zero = idx_fin + 100
        Fext[idx_fin:idx_zero] = np.linspace(1,0,idx_zero - idx_fin) * fm #Dans ce cas, Fext est une rampe

    elif exc == "gliss" :
        t1 = int(100e-3*Fe) #indice du temps où l'on lâche la corde
        t2 = t1 + int(temps_desc*Fe) #indice du temps où la force repasse à 0 (fin du glissement du plectre sur la corde) : à modéliser, int(1/2*1e-3*Fe) pour le moment #CF thèse Grégoire Derveaux, eq. 1.34
        Fext[:t1] = fm/2 * (1 - np.cos(np.pi*t[:t1]/t[t1]))
        Fext[t1:t2] = fm/2 * (1 + np.cos(np.pi*(t[t1:t2]-t[t1])/(t[t2]-t[t1])))

    xe_idx = find_nearest_index(xS, xexc_ratio*L) #indice de la position du point d'application de la force extérieure (32.5cm = 12e frette + 20cm pour l'excitation)
    NxS = len(xS)

    FextS_NxS_Nt = np.zeros((NxS,Nt))
    FextS_NxS_Nt[xe_idx, : ] = Fext

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
    
    return (t,FextS_NxS_Nt)

import control
def launch_simu_ss(t, FextS_NxS_Nt, modal_config_table, modal_config_corde, W, obs="all") :
    """
    Ce code permet de configurer l'espace d'état du modèle et de lancer une simulation temporelle.

    ## Inputs
    - t : vecteur temps de la simulation
    - FextS_NxS_Nt : vecteur spatial de la force extérieure appliquée à la corde
    - modal_config_table : dictionnaire contenant les données modales de la table
    - modal_config_corde : dictionnaire contenant les données modales de la corde
    - W : matrice de la formulation d'UK
    - obs : type d'observation, par défaut "all" càd que le vecteur d'observation est le vecteur d'état complet

    ## Sortie
    - Vecteur Q de dimension `(NmS+NmB, Nt)` ou `(2*(NmS+NmB, Nt))` : représente l'évolution des participations modales au cours du temps de la position de la corde sur les NmS premières coordonnées, et de la table sur les NmB dernières coordonnées.
    - Vecteur U de la représentation d'état.
    """
    #Récupération des données modales utiles
    ## Table
    NmB = modal_config_table["NmB_idx"].shape[1]
    MBinv = modal_config_table["MB_inv"]
    KB, CB = modal_config_table["KB"], modal_config_table["CB"]

    ## Corde
    NmS = modal_config_corde["NmS"]
    MSinv = modal_config_corde["MS_inv"]
    KS, CS = modal_config_corde["KS"], modal_config_corde["CS"] 
    phiS_Nx_NmS = modal_config_corde["phiS_Nx_NmS"]

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

    D = 0
    U = phiS_Nx_NmS.T @ FextS_NxS_Nt

    if obs == "pos" :
        #Pour observer la position
        C = np.block([
            [np.eye(NmS+NmB) ,  np.zeros((NmS+NmB,NmS+NmB))]
        ])
        C = np.block([
            [np.zeros((NmS+NmB,NmS+NmB)),np.eye(NmS+NmB)]
        ])
        sys = control.StateSpace(A,B,C,D)

        t, Q = control.forced_response(sys, T=t, U=U, X0=0)
    if obs == "all" :
        C = np.eye(2*(NmS+NmB))
        sys = control.StateSpace(A,B,C,D)

        t, Q = control.forced_response(sys, T=t, U=U, X0=0)

    return Q, U

def Main_ss(table_dico, corde_dico, exc_ref, Fe, obs="acc", plot_fext=False, plot_deformee=False, debug=False):
    """
    ## Inputs :
    - `table_dico` : dictionnaire contenant les paramètres de la table voir "table_corrigé.py". Les paramètres de la table sont :
        - 'rho_T' : masse volumique de la table
        - 'L_x' : longueur de la table en x
        - 'L_y' : longueur de la table en y
        - 'h' : épaisseur de la table
        - 'E_nu': rapport E/(1-nu^2)
        - 'xinB' : arrayLike, amortissements de la table
    - `corde_dico` : dictionnaire contenant les paramètres de la corde voir "corde.py". Les paramètres de la corde sont :
        - 'T' : tension de la corde
        - 'rho_l' : masse linéique de la corde
        - 'Lc' : longueur de la corde
        - 'B' : coefficient d'inharmonicité
    - `exc_ref` : dictionnaire contenant les paramètres de l'excitation de référence. Les paramètres de l'excitation de référence sont :
        - 'xexc_ratio' : position de l'excitation de référence en pourcentage de la longueur de la corde.
        - 'temps_desc' : temps de descente de l'excitation de référence.
    - `Fe` : fréquence d'échantillonage
    - `obs` : acc par défaut
    - `plot_fext` : booléen, si True, affiche la force extérieure appliquée à la corde
    - `plot_deformee` : booléen, si True, affiche la table et la corde déformées
    - `debug` : booléen, si True, affiche les matrices A, B, C, D et les valeurs propres de A

    ## Outputs
    - arrayLike, size (Nt). Par défaut, renvoie le vecteur de la force exercée par le chevalet sur la corde. Si `obs`=="pos", renvoie l'évolution temporelle en position du chevalet.
    Si obs=="acc", renvoie l'évolution temporelle en accélération d'un point de la table correspondant au point où les mesures ont été effectuées.
    Si obs=="acc_mod", tuple of size (Nm, Nt) renvoie l'évolution temporelle de la corde et de la table en accélération.
    """

    #Calcul de la configuration modale de la guitare avec la table et la corde d'entrée
    (M, M_inv, modal_config_table, modal_config_corde) = modal_config(table_dico, corde_dico, plot_deformee=plot_deformee)
    
    #Récupération des données modales utiles
    ## Table
    NmB = modal_config_table["NmB_idx"].shape[1]
    MB_inv = modal_config_table["MB_inv"]
    KB, CB = modal_config_table["KB"], modal_config_table["CB"]
    phiB_NxNy_NmB = modal_config_table["phiB_NxNy_NmB"]
    x, y = modal_config_table["x"], modal_config_table["y"]

    ##Corde
    NmS = modal_config_corde["NmS"]
    MS_inv = modal_config_corde["MS_inv"]
    KS, CS = modal_config_corde["KS"], modal_config_corde["CS"]
    phiS_Nx_NmS = modal_config_corde["phiS_Nx_NmS"]
    xS = modal_config_corde["xS"]

    #Calcul des paramètres de la formulation d'Udwadia-Kalaba
    W,Z, xyc = UK_params(M, M_inv, modal_config_table, modal_config_corde, article=False, mode='A2')
    
    #Calcul de la force extérieure
    xexc_ratio = exc_ref["xexc_ratio"]
    temps_desc = exc_ref["temps_desc"]
    t, FextS_NxS_Nt = fext(xS, Fe, temps_desc=temps_desc, xexc_ratio=xexc_ratio, L=65e-2, T=3, exc="gliss", plot_fext=plot_fext)

    if debug :
        print("Debug mode : returning modal_config_table, modal_config_corde, W, Z, xyc, t, FextS_NxS_Nt")
        return modal_config_table, modal_config_corde, W, Z, xyc, t, FextS_NxS_Nt
    
    #Pour observer la force :
    if obs == "force" :
        Q, U = launch_simu_ss(t,FextS_NxS_Nt, modal_config_table, modal_config_corde, W, obs="all")
        # pos_chev_Nt = phiS_Nx_NmS[-1,:] @ Q[:NmS]

        BG = np.block([
            [-MS_inv @ KS, np.zeros((NmS,NmB))],
            [np.zeros((NmB, NmS)), -MB_inv @ KB]
        ])
        BD = np.block([
            [-MS_inv @ CS, np.zeros((NmS,NmB))],
            [np.zeros((NmB, NmS)), -MB_inv @ CB]
        ])
        mat_to_acc_u = np.block([
            [BG, BG]
        ])

        B_new = np.block([
        [np.block([
            [MS_inv],
            [np.zeros((NmB, NmS))]
            ])]
        ])

        acc_u_Nm_Nt = mat_to_acc_u @ Q + B_new @ U
        Fc_Nm_Nt = Z @ acc_u_Nm_Nt

        Force_au_chevalet_Nt = phiS_Nx_NmS[-1,:] @ Fc_Nm_Nt[:NmS,:]
        return Force_au_chevalet_Nt
    
    #Pour observer l'accélération :
    elif "acc" in obs :
        Q, U = launch_simu_ss(t,FextS_NxS_Nt, modal_config_table, modal_config_corde, W, obs="all")

        BG = np.block([
            [-MS_inv @ KS, np.zeros((NmS,NmB))],
            [np.zeros((NmB, NmS)), -MB_inv @ KB]
        ])
        BD = np.block([
            [-MS_inv @ CS, np.zeros((NmS,NmB))],
            [np.zeros((NmB, NmS)), -MB_inv @ CB]
        ])
        mat_to_acc_u = np.block([
            [BG, BG]
        ])

        B_new = np.block([
        [np.block([
            [MS_inv],
            [np.zeros((NmB, NmS))]
            ])]
        ])

        acc_u_Nm_Nt = mat_to_acc_u @ Q + B_new @ U
        acc_Nm_Nt = W @ acc_u_Nm_Nt

        if obs == "acc_mod" :
            return  phiS_Nx_NmS[:,:] @ acc_Nm_Nt[:NmS,:], phiB_NxNy_NmB[:,:] @ acc_Nm_Nt[NmS:,:], xyc, len(x), len(y)
        else :
            acc_table_Nt = phiB_NxNy_NmB[xyc,:] @ acc_Nm_Nt[NmS:,:]
            return acc_table_Nt

    #Pour observer la position :
    elif obs == "pos" :
        Q, _ = launch_simu_ss(t,FextS_NxS_Nt,FextS_NxS_Nt, modal_config_table, modal_config_corde, W, obs="pos")
        pos_chev_Nt = phiS_Nx_NmS[-10,:] @ Q[:NmS]
        return pos_chev_Nt