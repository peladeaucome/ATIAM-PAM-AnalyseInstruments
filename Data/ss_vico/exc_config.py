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