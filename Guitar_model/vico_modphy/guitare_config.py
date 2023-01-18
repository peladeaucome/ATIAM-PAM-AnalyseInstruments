"""
Ce code permet de configurer la corde et la table d'harmonie utilisée par le modèle physique.
On paramètre les caractéristiques physiques des 2 structures et on calcul les propriétés modales associées.
La plaque est supposée simplement appuyée, la normalisation choisie est celle des masses modales.

- Fréquences propres : fnS (corde), fnB (table)
- Amortissements modaux : xinS, xinB
- Masses modales : MmS, MmB
- Déformées modales : phiS_Nx_NmS, phiB_NxNy_NmB
- Matrices modales (masse, amortissement, raideur) : (MS,CS,KS) , (MB,CB,KB)
- Matrices modales concaténées : M, C, K
"""

import numpy as np

#=========================================== CONFIG CORDE =====================================================================================================

## Paramètres physique
L = 0.65 #longueur de corde (m)
f1 = 110 #freq de la corde (hz)
T = 73.9 #tension de la corde (N)
rho_l = 3.61 * 10**(-3) #masse linéique (kg/m)
ct = np.sqrt(T/rho_l) #célérité des ondes transverse (M/s)
B = 4*10**(-5) #coefficient d'inarmonicité : B = E*I (N*m**2)

## Paramètres de discrétisation
NmS = 100 #Modes de cordes
NnS = np.arange(1,NmS+1)

NxS = 1000 #Discrétisation spatiale
xS = np.linspace(0,L,NxS) #Vecteur de la corde

## Calcul des modes
phiS_Nx_NmS = np.sin((2*NnS[np.newaxis,:]-1)*np.pi*xS[:,np.newaxis]/2/L) #Déformées d'une corde fixe aux extrémités
pnS = (2 * NnS - 1) * np.pi / (2 * L)
fnS = (ct / 2 / np.pi) * pnS * (1 + pnS**2 * B / (2 * T)) #Fréquences propres de la corde (hz)
print(f"Fréquence du dernier mode de corde calculé : {fnS[-1]:.0f} Hz")
wnS = 2*np.pi*fnS

etaf, etaA, etaB = 7e-5, 0.9, 2.5e-2
xinS = 1/2 * ( T*(etaf + etaA/2/np.pi/fnS) + etaB*B*pnS**2 ) /(T + B*pnS**2) #Amortissements modaux de la corde (ø)

MmS = rho_l * L / 2  #Masses modales de la corde (kg)

### Matrices modales
MS = np.diag(np.array([MmS]*NmS))
CS = MS * np.diag(2*wnS*xinS)
KS = MS*np.diag(wnS**2)

#=========================================== CONFIG PLAQUE =====================================================================================================

## Paramètres physique
h = 3e-3 #Epaisseur de  la plaque (m)
nu = 0.2 #Coeff de poisson (Pa)
E = 7e9 #Module de Young (Pa)
rho = 400 #Masse volumique (kg/m3)
D = E*h**3/(12*(1-nu**2)) #Raideur de la plaque
eta = 0.005 #Amortissement interne à la plaque
Lx, Ly, Lz = 530e-3, 200e-3, h #Dimensions (m)

## Paramètres de discrétisation
NB = 7          #Nombre de modes selon x
MB = 7          #Nombre de modes selon y
NmB = NB*MB      #Nombre de modes total considérés dans le modèle de plaque

Nx = 40
Ny = 40
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
x = np.linspace(0,Lx,Nx)
y = np.linspace(0,Ly,Ny)
X_plate, Y_plate = np.meshgrid(x, y)
X_ravel, Y_ravel = np.ravel(X_plate), np.ravel(Y_plate)

## Calcul des modes
def omega_pq (p,q) :    #Calcul analytique des pulsations propres d'une plaque en appuis simple
    return np.sqrt(E*h**2/(12*rho*(1-nu**2))) * ((p*np.pi/Lx)**2+(q*np.pi/Ly)**2)

wnB = np.zeros(NmB)
NmB_idx = np.zeros((2,NmB))   #Cette liste permet de remonter du mode contracté "i" au mode réel (n_i,m_i) en appelant NmB_idx[:,i]
j = 0
for n in range(1,NB+1) :
    for m in range(1,MB+1) :
        wnB[j] = omega_pq(n,m)
        NmB_idx[0,j] = n
        NmB_idx[1,j] = m
        j += 1

### Tri par ordre de fréquences croissantes
tri_idx = np.argsort(wnB)

wnB = wnB[tri_idx]    #On range les pulsations par ordre croissant
fnB = wnB/(2*np.pi)
print(f"Fréquence du dernier mode de plaque calculé : {fnB[-1]:.0f} Hz")
xinB = np.array([eta/2]*NmB)
NmB_idx = NmB_idx[:,tri_idx]      #On ordonne les modes par ordre croissant

### Déformées
def phi_pq (p,q,x,y) :  #Calcul analytique des déformées des modes d'une plaque en appuis simple
    return np.sin(p*np.pi*x/Lx)*np.sin(q*np.pi*y/Ly)

phiB_NxNy_NmB = np.zeros((Nx*Ny,NmB)) #Matrice des déformées avec les 2 dimensions spatiales applaties en 1 dimension
for mode in range (NmB) :
    for point in range(Nx*Ny) :
        n = NmB_idx[0,mode]
        m = NmB_idx[1,mode]
        x_ = X_ravel[point]
        y_ = Y_ravel[point]

        phiB_NxNy_NmB[point,mode] = phi_pq(n, m , x_, y_)

### Masses modales
MmB = np.zeros(NmB)
for j in range(NmB) :
    PHI_j_Ny_Nx = np.reshape(phiB_NxNy_NmB[:,j],(Ny,Nx))      #Correspond à la déformée du mode j sur la plaque (en 2D)
    MmB[j] = rho*h* np.sum(np.sum(PHI_j_Ny_Nx**2,axis=1),axis=0)*dx*dy

### Normalisation des masses modales
norme_deformee_NmB = np.sqrt(MmB)         #Ref : Modal Testing Theory, Practice and Application p.54, Eq. (2.25)
phiB_NxNy_NmB = phiB_NxNy_NmB[:,:] / norme_deformee_NmB[np.newaxis,:]

### Matrices modales
MB = np.diag(MmB)
CB = np.diag(2*MmB*wnB*xinB)
KB = np.diag(MmB*wnB**2)

# Matrices concaténées par bloc
M = np.block([
              [MS               , np.zeros((NmS,NmB))],
              [np.zeros((NmB,NmS)), MB               ]
])

K = np.block([
              [KS               , np.zeros((NmS,NmB))],
              [np.zeros((NmB,NmS)), KB               ]
])

C = np.block([
              [CS               , np.zeros((NmS,NmB))],
              [np.zeros((NmB,NmS)), CB               ]
])