from numpy import pi
#Sous forme de dictionnaire
corde_article = {
    'T' : 73.9,
    'rho_l' : 3.61*1e-3,
    'Lc' : 0.65,
    #'r' : 0.40 * 10e-3,
    'B' : 4e-5,
}

corde_acier_1 = {
    'T' : 93.947707,
    'r' : 0.71/2 * 1e-3,
    'Lc' : 0.65,
    'rho_l' : 7800 * pi * 0.71/2 * 1e-3**2,
    'B' : 0.0001943572734201584,
}

corde_acier_2 = {
    'T' : 93.2612415,
    'r' : 0.97/2 * 1e-3,
    'Lc' : 0.65,
    'rho_l' : 7800 * pi * 0.97/2 * 1e-3**2,
    'B' : 0.0002581,
}