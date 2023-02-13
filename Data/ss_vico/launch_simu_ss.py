import control
def launch_simu_ss(t, FextS_NxS_Nt, phiS_Nx_NmS, NmS, NmB, MBinv, MSinv, KS, KB, CS, CB, W, obs="all") :
    """
    Ce code permet de configurer l'espace d'état du modèle et de lancer une simulation temporelle.

    ## Entrées
    - Configuration de guitare préalablement établie dans "guitare_config.py"
    - Paramètres issue de la formulation d'Udwadia-Kalaba établis dans "UK_parameters.py"
    - Paramètres de simulation : force extérieure "FextS_NxS_Nt", vecteur temps de simulation "t"
    - `obs` : si obs=="pos", le vecteur de sortie est de taille NmB+NmS et correspond aux participations en position.
    Si obs=="all", le vecteur de sortie est de taille 2*(NmB+NmS) et correspond aux participations en position et en vitesse concaténées.

    ## Sortie
    - Vecteur Q de dimension `(NmS+NmB, Nt)` ou `(2*(NmS+NmB, Nt))` : représente l'évolution des participations modales au cours du temps de la position de la corde sur les NmS premières coordonnées, et de la table sur les NmB dernières coordonnées.
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

        t, Q, Qout, inputs = control.forced_response(sys, T=t, U=U, X0=0)
    if obs == "all" :
        C = np.eye(2*(NmS+NmB))
        sys = control.StateSpace(A,B,C,D)

        t, Q, state = control.forced_response(sys, T=t, U=U, X0=0, return_x=True, squeeze=False)

    return Q, U, state