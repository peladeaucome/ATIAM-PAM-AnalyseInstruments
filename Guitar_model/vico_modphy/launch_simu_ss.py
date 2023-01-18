"""
Ce code permet de configurer l'espace d'état du modèle et de lancer une simulation temporelle.

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

import control #C'est un module à télécharger qui permet de simuler des espaces d'état : https://python-control.readthedocs.io/en/0.9.1/

MSinv = np.linalg.inv(MS)
MBinv = np.linalg.inv(MB)

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
    [np.zeros((NmS+NmB, NxS))],
    [W @ np.block([
        [MSinv @ phiS_Nx_NmS.T],
        [np.zeros((NmB, NxS))]
        ])]
])

C = np.block([
    [np.eye(NmS+NmB) ,  np.zeros((NmS+NmB,NmS+NmB))]
])
#C permet d'observer la position avec les NmS premieres contributions des modes de la corde et ensuite les NmB de la plaque

D = 0

sys = control.StateSpace(A,B,C,D)

U = FextS_NxS_Nt

t, Q = control.forced_response(sys, T=t, U=U, X0=0)