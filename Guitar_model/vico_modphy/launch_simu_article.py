"""
Ce code permet de configurer l'espace d'état du modèle et de lancer une simulation temporelle
avec le shéma numérique de résolution de l'article (page 15 en annexe de 
Dynamical computation of constrained flexible systems using a modal Udwadia-Kalaba
formulation: Application to musical instruments)

# Entrées
- Configuration de guitare préalablement établie dans "guitare_config.py"
- Paramètres issue de la formulation d'Udwadia-Kalaba établis dans "UK_parameters.py"
- Paramètres de simulation : force extérieure "FextS_NxS_Nt", vecteur temps de simulation "t"

# Sortie
- Vecteur Q de dimension (NmS+NmB, Nt) : représente l'évolution des participations modales au cours du temps de la position de la corde sur les NmS premières coordonnées, et de la table sur les NmB dernières coordonnées.
"""

from guitare_config import *
from UK_parameters import *
from simu_config import t, FextS_NxS_Nt

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
M_inv = np.linalg.inv(M)

#shéma
for i in range(point_temp-1):
    h = t[i+1] - t[i]   #step d'intégration
    q_temp[:,i+1] = q_temp[:,i] + h * q_d_temp + 0.5 * h**2 * q_dd_temp[:,i]
    q_d_temp_demi = q_d_temp + 0.5 * h * q_dd_temp[:,i]

    F_temp = - C @ q_d_temp_demi - K @ q_temp[:,i+1] + F_pro_tot[:,i+1]
    q_u_dd_temp = M_inv @ F_temp
    q_pour_f[:,i+1] = q_u_dd_temp

    q_dd_temp[:,i+1] = W @ q_u_dd_temp

    q_d_temp = q_d_temp + 0.5 * h * (q_dd_temp[:,i] + q_dd_temp[:,i+1])
Q = q_temp

F_c = Z @ q_pour_f