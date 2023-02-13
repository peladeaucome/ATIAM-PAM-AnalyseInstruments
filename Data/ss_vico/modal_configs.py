from guit_config import h, E_nu, rho, Lx, Ly, T, rho_l, L, B, xinB



def bigdickenergy_ss(h, E_nu, rho, Lx, Ly, T, rho_l, L, B, xinB) :
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
    fnB = wnB/2/np.pi
    # print(fnB)
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