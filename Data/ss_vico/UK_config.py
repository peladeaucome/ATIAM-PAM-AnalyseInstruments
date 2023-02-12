from modal_configs import M,M_inv,NmS, NmB, phiS_Nx_NmS,phiB_NxNy_NmB, x, y, xS
from utils_data import find_nearest_index, ravel_index_from_true_indexes

mode = "A1"

phiSB = phiS_Nx_NmS[-1,:] #déformée de la corde au point du chevalet
phiSF = phiS_Nx_NmS[20,:] #déformée de la corde au point d'appuis du doigt du guitariste
# phiSF = np.ones(NmS)*1e-10 #déformée de la corde au point d'appuis du doigt du guitariste

if model : 
    Nx = len(x)
    Ny = len(y)
    #Point de couplage (par rapport à la table)
    xc, yc = x[int(24.5/40*Nx)], y[Ny//2]
    xc_idx, yc_idx = find_nearest_index(x, xc), find_nearest_index(y, yc)
    xyc = ravel_index_from_true_indexes(xc_idx, yc_idx, Nx)
    print(xc_idx, yc_idx)
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

return (W,Z, xyc)