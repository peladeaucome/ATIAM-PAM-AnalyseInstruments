import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Enu_AS (mode_idx, Lx, Ly, h, rho, modal_analysis, NmB_idx) :
    n, m = NmB_idx[:,mode_idx]
    fnm = modal_analysis["freq"][mode_idx]
    Anm = ((n**2*Ly**2 + m**2*Lx**2)/(Lx**2*Ly**2))**2
    I = h**2*np.pi**4/12/rho
    Enu = (2*np.pi*fnm)**2/I/Anm
    return Enu

def corr_dim_AS(modal_analysis, Lx_init, Ly_init, h_init, rho, E, delta_l, delta_h=0, Nh=1, mode_idx=0, plot=False, nu_cible = 0.4, eps=3e-3) :
    """
    Prends une plaque d'entrée avec son analyse modale, retourne une plaque pour laquelle on corrige les dimensions pour approcher un appuis simple,
    avec des valeurs de E et nu qui sont celles d'un constructeur.

    ## Inputs (not optional)
    - `modal_analysis` : dictionnaire contenant une clef ["freq"] avec les fréquences propres en Hz
    - `Lx_init` : longueur réelle mesurée de la plaque (en m)
    - `Lx_init` : largeur réelle mesurée de la plaque (en m)
    - `h_init` : épaisseur réelle mesurée de la plaque (en m)
    - `rho` : masse volumique du matériau (kg/m3)
    - `E` : module d'Young du matériau (Pa)
    - `delta_l` : corrections maximales à appliquer sur les longueurs et largeurs (en m)

    ### Optional
    - `delta_h` : correction maximale à appliquer sur l'épaisseur (en m) (0 par défaut : pas de correction d'épaisseur)
    - `Nh` : nombre de test de correction d'epaisseur (1 par défaut)
    - `mode_idx` : choix du mode pour accorder la plaque (premier mode de l'analyse modal par défaut)
    - `plot` : si True, montre une cartographie des valeurs de nu en fonction des dimensions
    - `nu_cible` : valeur de nu à viser pour établir la correction (0.4 par défaut)
    - `eps` : tolérance pour comparer l'égalité entre 2 valeurs (3e-3 par défaut)

    ## Outputs (dans l'ordre)
    - `Lx_corr` : nouvelle longueur de plaque après correction
    - `Ly_corr` : nouvelle largeur de plaque après correction
    - `h_corr` : nouvelle épaisseur de plaque après correction
    - `nu_corr` : nu réel le plus proche de la cible demandé
    """

    _, NmB_idx = compute_AS_frequencies(Lx_init, Ly_init, h_init, 0.3, E, rho, Nmx=4, Nmy=4)

    Lys = np.linspace(Ly_init-delta_l, Ly_init+delta_l)
    # Lys = np.insert(Lys, 0,Ly_init)
    Lxs = np.linspace(Lx_init-delta_l, Lx_init+delta_l)
    # Lxs = np.insert(Lxs, 0,Lx_init)
    hs = np.linspace(h_init-delta_h, h_init+delta_h, Nh)
    # hs = np.insert(hs, 0,h_init)
    nus = np.zeros((len(Lxs),len(Lys), len(hs)))
    for i in range(len(Lxs)) :
        for j in range(len(Lys)) :
            for k in range(len(hs)) :
                Enu_AS_0 = Enu_AS(mode_idx, Lxs[i], Lys[j], hs[k], rho ,  modal_analysis=modal_analysis, NmB_idx = NmB_idx)
                if 1-E/Enu_AS_0 < 0 :
                    nu_cons0 = 0
                else :
                    nu_cons0 = np.sqrt(1-E/Enu_AS_0)
                nus[i,j,k] = nu_cons0
    if plot :
        for k in range(len(hs)) :
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            img = ax1.imshow(nus[:,:,k].T,
                extent=[Lxs[0], Lxs[-1] , Lys[0], Lys[-1]] ,
                cmap="jet" ,
                interpolation = "bilinear",
                aspect="auto" ,
                origin="lower")

            cbar = fig.colorbar(img,ax=ax1)
            img.set_clim(0, 1)
            cbar.set_label(r"$\nu$")
            ax1.set_ylabel("Ly (m)")
            ax1.set_xlabel(r"Lx (m)")
            ax1.set_title(fr"Epaisseur : {hs[k]*1e3:.2f} mm")

            fig.tight_layout()

            plt.show()
    #Détermination des correctoions nécessaires pour approcher une valeure classique de nu
    ps = np.argwhere(np.abs(nus-nu_cible) < eps)

    dico_init = {
        "Lx_init" : [Lx_init],
        "Ly_init" : [Ly_init],
        "h_init" : [h_init],
    }
    print(pd.DataFrame(dico_init))
    dico_corr = {
        "Lx_corr" : Lxs[ps[:,0]],
        "Ly_corr" : Lys[ps[:,1]],
        "h_corr" : hs[ps[:,2]],
    }
    print(pd.DataFrame(dico_corr))
    
    idx_choice = int(input("Choisir l'indice de la correction à appliqué :"))
    print("Correction choisie :", idx_choice)

    Lx_corr, Ly_corr, h_corr = Lxs[ps[idx_choice,0]], Lys[ps[idx_choice,1]], hs[ps[idx_choice,2]]
    nu_corr = nus[ps[idx_choice,0],ps[idx_choice,1],ps[idx_choice,2]]

    return Lx_corr, Ly_corr, h_corr, nu_corr

def compute_AS_frequencies(Lx, Ly, h, nu, E, rho, Nmx=3, Nmy=3) :
    """
    Calcul les modes d'une plaque simplement supportée pour une configuration donnée.

    ## Outputs
    - fnB : arrayLike, vecteur des fréquence propres (aplati)
    - NmB_idx : arrayLike, matrice permettant de remonté au réel indice n,m du mode aplatit i en appelant NmB_idx[:,i]
    """
    Nmtot = Nmx*Nmy
    ## Calcul des modes
    def omega_pq (p,q) :    #Calcul analytique des pulsations propres d'une plaque en appuis simple
        return np.sqrt(E*h**2/(12*rho*(1-nu**2))) * ((p*np.pi/Lx)**2+(q*np.pi/Ly)**2)

    wnB = np.zeros(Nmtot)
    NmB_idx = np.zeros((2,Nmtot))   #Cette liste permet de remonter du mode contracté "i" au mode réel (n_i,m_i) en appelant NmB_idx[:,i]
    j = 0
    for n in range(1,Nmx+1) :
        for m in range(1,Nmy+1) :
            wnB[j] = omega_pq(n,m)
            NmB_idx[0,j] = n
            NmB_idx[1,j] = m
            j += 1

    ### Tri par ordre de fréquences croissantes
    tri_idx = np.argsort(wnB)

    wnB = wnB[tri_idx]    #On range les pulsations par ordre croissant
    fnB = wnB/(2*np.pi)
    return fnB, NmB_idx

def corr_dim_AS_optim(modal_analysis, Lx_init, Ly_init, h_init, rho, E, delta_l, delta_h=0, Nh=1, mode_idx=0, plot=False, nu_cible = 0.4, eps=3e-3, optim_tol=20, NconfigMax=50, Nmatch=5) :
    """
    Prends une plaque d'entrée avec son analyse modale, retourne une plaque pour laquelle on corrige les dimensions pour approcher un appuis simple,
    avec des valeurs de E et nu qui sont celles d'un constructeur. Cette version fais une optimisation sur un nombre de configuration maximal donné,
    et renvoies les configurations les plus proches du tableau de fréquence trouvé par la mesure.

    ## Inputs (not optional)
    - `modal_analysis` : dictionnaire contenant une clef ["freq"] avec les fréquences propres en Hz
    - `Lx_init` : longueur réelle mesurée de la plaque (en m)
    - `Lx_init` : largeur réelle mesurée de la plaque (en m)
    - `h_init` : épaisseur réelle mesurée de la plaque (en m)
    - `rho` : masse volumique du matériau (kg/m3)
    - `E` : module d'Young du matériau (Pa)
    - `delta_l` : corrections maximales à appliquer sur les longueurs et largeurs (en m)

    ### Optional
    - `delta_h` : correction maximale à appliquer sur l'épaisseur (en m) (0 par défaut : pas de correction d'épaisseur)
    - `Nh` : nombre de test de correction d'epaisseur (1 par défaut)
    - `mode_idx` : choix du mode pour accorder la plaque (premier mode de l'analyse modal par défaut)
    - `plot` : si True, montre une cartographie des valeurs de nu en fonction des dimensions
    - `nu_cible` : valeur de nu à viser pour établir la correction (0.4 par défaut)
    - `eps` : tolérance pour comparer l'égalité entre 2 valeurs (3e-3 par défaut)
    - `optim_tol` : tolérance pour estimer que 2 fréquences sont "proches" entre la synthèse et l'analyse modale. Au delà, on estime que les fréquences n'étaient pas associé au même mode.
    - `NconfigMax` : nombre de config maximal a testé
    - `Nmatch` : nombre de fréquences sur lesquels on calcul la distance. Par défaut Nmatch=5 soit on analyse la distance sur les 5 premiers modes uniquement.

    ## Outputs (dans l'ordre)
    - `Lx_corr` : nouvelle longueur de plaque après correction
    - `Ly_corr` : nouvelle largeur de plaque après correction
    - `h_corr` : nouvelle épaisseur de plaque après correction
    - `nu_corr` : nu réel le plus proche de la cible demandé
    """

    _, NmB_idx = compute_AS_frequencies(Lx_init, Ly_init, h_init, 0.3, E, rho, Nmx=4, Nmy=4)

    Lys = np.linspace(Ly_init-delta_l, Ly_init+delta_l)
    # Lys = np.insert(Lys, 0,Ly_init)
    Lxs = np.linspace(Lx_init-delta_l, Lx_init+delta_l)
    # Lxs = np.insert(Lxs, 0,Lx_init)
    hs = np.linspace(h_init-delta_h, h_init+delta_h, Nh)
    # hs = np.insert(hs, 0,h_init)
    nus = np.zeros((len(Lxs),len(Lys), len(hs)))
    for i in range(len(Lxs)) :
        for j in range(len(Lys)) :
            for k in range(len(hs)) :
                Enu_AS_0 = Enu_AS(mode_idx, Lxs[i], Lys[j], hs[k], rho ,  modal_analysis=modal_analysis, NmB_idx = NmB_idx)
                if 1-E/Enu_AS_0 < 0 :
                    nu_cons0 = 0
                else :
                    nu_cons0 = np.sqrt(1-E/Enu_AS_0)
                nus[i,j,k] = nu_cons0
    if plot :
        for k in range(len(hs)) :
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            img = ax1.imshow(nus[:,:,k].T,
                extent=[Lxs[0], Lxs[-1] , Lys[0], Lys[-1]] ,
                cmap="jet" ,
                interpolation = "bilinear",
                aspect="auto" ,
                origin="lower")

            cbar = fig.colorbar(img,ax=ax1)
            img.set_clim(0, 1)
            cbar.set_label(r"$\nu$")
            ax1.set_ylabel("Ly (m)")
            ax1.set_xlabel(r"Lx (m)")
            ax1.set_title(fr"Epaisseur : {hs[k]*1e3:.2f} mm")

            fig.tight_layout()

            plt.show()
    #Détermination des correctoions nécessaires pour approcher une valeure classique de nu
    ps = np.argwhere(np.abs(nus-nu_cible) < eps)

    Nconfig = ps.shape[0]
    if Nconfig > NconfigMax :
        Nconfig = NconfigMax

    config_distances = np.zeros(Nconfig)
    for i in range(Nconfig) :
        Lx_corr, Ly_corr, h_corr = Lxs[ps[i,0]], Lys[ps[i,1]], hs[ps[i,2]]
        nu_corr = nus[ps[i,0],ps[i,1],ps[i,2]]
        fn,_ = compute_AS_frequencies(Lx_corr, Ly_corr, h_corr, nu_corr, E, rho)
        for j in range(Nmatch) :
            for k in range(len(modal_analysis["freq"])) :
                deltaf = np.abs(modal_analysis["freq"][k] - fn[j])
                if deltaf < optim_tol :
                    config_distances[i] += deltaf
                else :
                    config_distances[i] += 50 #pour ne pas dépendre ensuite de la distance : on estime que de toute facon ces 2 pics ne sont pas comparables, et cela permet d'impacter fortement quand il y a aucun match

    best_config_idx = np.argmin(config_distances)
    Lx_corr, Ly_corr, h_corr = Lxs[ps[best_config_idx,0]], Lys[ps[best_config_idx,1]], hs[ps[best_config_idx,2]]
    nu_corr = nus[ps[best_config_idx,0],ps[best_config_idx,1],ps[best_config_idx,2]]
    
    return Lx_corr, Ly_corr, h_corr, nu_corr
