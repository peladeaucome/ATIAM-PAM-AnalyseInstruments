"""
Ce code permet de créer 2 objets (des matrices) dans lesquelles on peut accéder à toutes les données utiles mesurées.
Ces matrices contiennent toutes les configurations testées.

- Nta : nombre de table
- Ntc : nombre de type de corde (pour l'instant 1 seul : acier)
- Nc : nombre de corde (6)
- Np : nombre de plectre (4)

## Outputs

### Mesures en jeu
- `jeu_Nta_Ntc_Nc_Np` : arrayLike. 
Un élément de cette matrice correspond à un dictionnaire avec les clefs utiles mesurées lors de l'expérience, pour une configuration de table, corde, plectre.

Clefs pour les mesures en jeu :
    - "mat_table" : str, matériau de la table
    - "mat_corde" : str, matériau de la corde
    - "acc" : arrayLike, accélération mesurée
    - "fs" : fréquence d'échantillonnage
    - "temps" : arrayLike, vecteur temps associé

### Mesures des tables
- `deforme_Nta_Npos`: arrayLike.
Un élément de cette matrice correspond à un dictionnaire avec les clefs utiles mesurées lors de l'expérience, pour une configuration d'accéléro, table.

Clefs pour les mesures des tables :
- "mat_table" : str, matériau de la table
- "pos" : str, position 1 ou 2 de l'accéléro
- "acc" : arrayLike, accélération mesurée
- "fs" : fréquence d'échantillonnage
- "temps" : arrayLike, vecteur temps associé
- "mar" : arrayLike, force du marteau mesurée
- "RI" : arrayLike, réponse impulsionnelle du système
- "FRF" : arrayLike, FRF de la table
- "freq" : vecteur fréquence associé
"""

import numpy as np
from scipy.io import loadmat
from scipy.signal import oaconvolve
import os
import sys
import mymodule2 as mm
sys.path.append("../")

Nta = 4 #4 type de tables
Ntc = 1 #1 seul type de corde pour l'instant : acier
Nc = 6 #6 cordes
Np = 4 #4 plectres différents

jeu_Nta_Ntc_Nc_Np = np.zeros((Nta,Ntc,Nc,Np), dtype=object) #Initialisation de la matrice pour les mesures en jeu

Npos = 2 #Nombre de position de l'accéléro pour la déformé
deforme_Nta_Npos = np.zeros((Nta, Npos), dtype=object)


path_to_folder = "../../../Mesures/Mesures_2023.01.27_LAM/"
temp_path = path_to_folder

#On parcours les différentes tables utilisées lors des expériences
itable = 0 #itable est le compteur réel du nombre de table (car it rajoute DS_Store et autres fichiers non souhaités)
            #on ajoute 1 a itable seulement après avoir sélectionner proprement les tables
for it, table in enumerate(os.listdir(path_to_folder)) :
    if table[0] != "." :
        temp_path = path_to_folder + table + "/"

        #On parcours le fichier avec les différentes cordes et les déformées de chaque plaque
        for itc, type_corde in enumerate(os.listdir(temp_path)) : #Types de corde
            filename, file_extension = os.path.splitext(type_corde)

            #Pour les fichiers en jeu
            if file_extension == "" and type_corde[0] != "." : #Pour sélectionner uniquement les fichiers, et pas .DS_store
                if "Nylon" not in type_corde : #Pour le moment on enlève les cordes nylons
                    temp_path += filename + "/"

                    #On parcours le dernier fichier avec tous les "C_1_P_1.mat" de chaque corde et chaque plectre
                    for ic, corde in enumerate(os.listdir(temp_path)) :
                        filename, file_extension = os.path.splitext(corde)
                        if file_extension == ".mat" :
                            #on extrait les indices de plectre et de corde du fichier
                            icorde = int(filename[2])
                            iplectre = int(filename[-1])
                            #chargement des données matlab
                            mat = loadmat(temp_path + corde)
                            #création du dictionnaire "utile" de ce fichier
                            data_dict = {}
                            data_dict["mat_table"] = table
                            data_dict["mat_corde"] = type_corde
                            data_dict["acc"] = mat["acc_t"].reshape(-1)
                            data_dict["fs"] = int(mat["fs"].reshape(-1))
                            data_dict["temps"] = mat["time"].reshape(-1)
                            #assignation du dictionnaire dans le grand tableau avec tout
                            jeu_Nta_Ntc_Nc_Np[itable,0,icorde-1,iplectre-1] = data_dict #on pourra rendre le "0" variable qd on prendre d'autres types de cordes en compte
                    temp_path = path_to_folder + table + "/"
            
            #Pour les fichiers "Deforme_P1.mat => déformées de la table
            elif file_extension == ".mat" and "deforme".upper() in filename.upper() :
                #on extrait l'indice de la position testée
                idx_pos = int(filename[-1])-1
                #chargement des données matlab
                mat = loadmat(temp_path + type_corde)
                #création du dictionnaire "utile" associé à ce fichier
                data_dict = {}
                data_dict["mat_table"] = table
                data_dict["pos"] = f"pos {idx_pos+1}"
                # data_dict["acc"] = mat["acc_t"].reshape(-1)
                data_dict["fs"] = int(mat["fs"].reshape(-1))
                # data_dict["temps"] = mat["time"].reshape(-1)
                # data_dict["mar"] = mat["mar_t"].reshape(-1)
                RI = oaconvolve(mat["acc_t"].reshape(-1),mat["mar_t"].reshape(-1))
                
                #Nettoyage de la RI (couper le début et la fin)
                data_dict["tRI"], data_dict["RI"] = mm.clean_RI(RI, data_dict["fs"], method="max", cut_end=1.2)

                data_dict["FRF"] = mat["FRF"].reshape(-1)
                data_dict["freq"] = mat["freq"].reshape(-1)

                #assignation du dictionnaire temporaire au tableau final
                # print(itable, idx_pos)
                deforme_Nta_Npos[itable, idx_pos] = data_dict
        itable += 1

