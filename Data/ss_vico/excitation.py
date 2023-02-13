"""
Ce code définit un dictionnaire avec les paramètres d'excitation de la corde de référence.
C'est un peu overkill car il y a que 2 valeurs pour l'instant mais ça permet de garder une cohérence avec les autres dictionnaires, et d'ajouter des paramètres si jamais.
"""

exc_ref = {
    "xexc_ratio" : (32.5+20)/65, #position de l'excitation visée lors des mesures
    "temps_desc" : 1/2*1e-3, #décroissance de référence (selon des observations sur les simulations)
}