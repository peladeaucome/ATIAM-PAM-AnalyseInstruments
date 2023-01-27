"""
Ce code répértorie les différentes cordes et tables disponibles expérimentalement. Elles servent ainsi de "références" pour la synthèse.
On défini également les ratios de position du chevalet.
"""


#============================ CORDES =======================================================

corde_acier = {
    "id" : "acier",
    "L" : 65e-2, #longueur (m)
    "d" : 0.3e-3, #diamètre (m)
    "f0" : 128, #fréquence fondamentale accordée sur la guitare (Hz),
    "T" : ..., #tension de la corde (N)
    "E" : 210e9, #module d'Young (Pa)
    "rho_m" : 7850, #masse volumique (kg/m3)
}

corde_nylon = {
    "id" : "acier",
    "L" : 65e-2, #longueur (m)
    "d" : 0.3e-3, #diamètre (m)
    "f0" : 128, #fréquence fondamentale accordée sur la guitare (Hz),
    "T" : ..., #tension de la corde (N)
    "E" : 210e9, #module d'Young (Pa)
    "rho_m" : 7850, #masse volumique (kg/m3)
}

corde_article = {
    "id" : "article",
    "L" : 65e-2, #longueur (m)
    "d" : 0.3e-3, #diamètre (m)
    "f0" : 128, #fréquence fondamentale accordée sur la guitare (Hz),
    "T" : ..., #tension de la corde (N)
}

#============================ TABLES =======================================================

table_composite = {
    "id" : "composite",
    "Lx" : 40e-2, #longueur de la table (m)
    "Ly" : 25.9e-2, #largeur de la table (m)
    "h" : 2.8e-3, #épaisseur de la table (m)
    "rho" : 236e-3/(40e-2*25.9e-2*2.8e-3), #masse volumique (kg/m3)
    "E" : 2.4e9, #Module d'Young (Pa) (basé sur http://www.vidal-panneaux.fr/mdf.html)
    "nu" : 0.02, #coefficient de Poisson (Pa) (basé sur le premier mode calculé)
}

table_plexi = {
    "id" : "plexi",
    "Lx" : 40e-2, #longueur de la table (m)
    "Ly" : 25.9e-2, #largeur de la table (m)
    "h" : 3.5e-3, #épaisseur de la table (m)
    "rho" : 362e-3/(40e-2*25.9e-2*2.8e-3), #masse volumique (kg/m3)
    "E" : ..., #Module d'Young (Pa) 
    "nu" : ..., #coefficient de Poisson (Pa)
}

table_acier = {
    "id" : "acier",
    "Lx" : 40e-2, #longueur de la table (m)
    "Ly" : 25.9e-2, #largeur de la table (m)
    "h" : 6e-3, #épaisseur de la table (m)
    "rho" : 1588e-3/(40e-2*25.9e-2*2.8e-3), #masse volumique (kg/m3)
    "E" : 210e9, #Module d'Young (Pa)
    "nu" : 0.33, #coefficient de Poisson (Pa)
}

#============================ COUPLAGE AU CHEVALET =======================================================
#Cela permettra de définir le point d'attache entre la corde et le chevalet dans le modèle (dans UK_parameters.py)
CB_ratio_x = 24.7/40
CB_ratio_y = 10.5/25.9