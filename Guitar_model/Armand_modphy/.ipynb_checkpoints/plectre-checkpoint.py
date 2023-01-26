import numpy as np
from scipy.misc import derivative
import scipy.integrate as integrate
from plectre_parameters import *

def parameters(T_corde, rho_l_corde, L_plectre, E_plectre, I_plectre):
    Z = np.sqrt(T_corde * rho_l_corde)
    alpha = (2 * Z * (L_plectre**3)) / (3 * E_plectre * I_plectre)
    return Z,alpha


def g(t, slope=slope):
    return slope*t


def deplacement_main_linear(duree, Fe, slope):
    temps = np.linspace(0, duree, duree*Fe)
    D = g(temps, slope)
    return temps, D

def derivation(t, g):
    derivee = derivative(g, t, dx = 1e-6)
    return derivee

def derivation_compete(temps, g):
    derivee_call = np.zeros(len(temps))
    for t_0 in range(len(temps)):
        derivee_all[t_0] = derivative(g, t_0, dx = 1e-6)
    return derivee_all

def force(temps, Z, alpha, f, L):
    F = np.zeros(len(temps))
    P_lim = (4 * Z * L * f) / (3 * alpha)        
    print("La froce de relâchement est de "+str(P_lim)+"N")
    i=0
    for t_0 in temps:
        I = integrate.quad(lambda t: derivation(t, g) * np.exp(t / alpha), 0, t_0)
        force = (2 * Z)/alpha * (np.exp(-t_0 / alpha)) * I[0]
        if force > P_lim:
            t_rel = t_0
            print("Relâchement de la corde à "+str(t_0)+"s")
            break
        else:
            F[i] = force
            i+=1
    return F, t_rel