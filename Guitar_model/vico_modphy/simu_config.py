"""
Ce code permet de fixer les paramètres de la simulation temporelle du modèle physique.

# Entrées
- Configuration spatiale de la corde : .xS, .L, .NxS

# Sorties
- Vecteur temps de la simulation : t
- Vecteur force extérieure au cours du temps, par point de la corde : FextS_NxS_Nt
- Tracé de la force au cours du temps si plot_fext == True
"""

import numpy as np
import utils_modphy as u
from guitare_config import xS, L, NxS, fnS, fnB
import matplotlib.pyplot as plt


# Vecteur temps
Fe = int(2.2*max(fnS[-1], fnB[-1])) #Fréquence d'échantillonnage (hz) (on prends un peu plus que la limite pour respecter Shannon pour optimiser)
Fe = 16000
# print(f"Fréquence d'échantillonage : {Fe} Hz")
T = 3 #Temps d'acquisition (s)
# print(f"Temps d'acquisition : {T} s")
t = np.linspace(0, T, T*Fe) #Vecteur temps
Nt = len(t)

# Force extérieure appliquée à la corde
Fext = np.zeros_like(t)
idx_deb = 0
idx_fin = 1*Fe
Fext[idx_deb:idx_fin] = np.linspace(0,1,idx_fin - idx_deb) * 5 #Dans ce cas, Fext est une rampe

xe_idx = u.find_nearest_index(xS, 0.9*L)

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