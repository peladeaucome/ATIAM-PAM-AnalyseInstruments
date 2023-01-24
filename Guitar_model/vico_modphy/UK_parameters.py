"""
Ce code permet de paramètrer les matrices utiles dans la formulation d'Udwadia-Kalaba. 
Il prend en entrée la configuration de la guitare établie au préalable dans "guitare_config.py".

- Aa1, Aa2 : a1 si on néglige les vibrations de la table au point de couplage, a2 si on les prends en compte
- Bplus : voir article
- W : voir article
"""

from guitare_config import *
import utils_modphy as u

phiSB = phiS_Nx_NmS[-1,:] #déformée de la corde au point du chevalet
phiSF = phiS_Nx_NmS[250,:] #déformée de la corde au point d'appuis du doigt du guitariste

#Point de couplage (par rapport à la table)
xc, yc = x[int(24.5/40*Nx)], y[Ny//2]
xc_idx, yc_idx = u.find_nearest_index(x, xc), u.find_nearest_index(y, yc)
xyc = u.ravel_index_from_true_indexes(xc_idx, yc_idx, Nx)
#pour modèle de la plaque:
#phiBS = phiB_NxNy_NmB[xyc,:]

#pour valeur numérique article:
phiBS = phiB_NxNy_NmB

Aa1 = np.block([
                [phiSB.T, np.zeros(NmB)],
                [phiSF.T, np.zeros(NmB)]
])

Aa2 = np.block([
                [phiSB.T, -phiBS],
                [phiSF.T, np.zeros(NmB)]
])

# Choix concernant le mouvement de la table au point de couplage
Aa = Aa2

B = Aa @ np.linalg.inv(M**(1/2))
Bplus = B.T @ np.linalg.inv((B @ B.T))
W = np.eye(NmS+NmB) - np.linalg.inv(M**(1/2)) @ Bplus @ Aa

Z = - M ** (1/2) @ Bplus @ Aa #pour calculer la force ensuite