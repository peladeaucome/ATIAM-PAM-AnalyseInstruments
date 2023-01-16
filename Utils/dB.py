import numpy as np

def dB20(x):
    return 20*np.log10(np.abs(x))


def dB10(x):
    return 10*np.log10(np.abs(x))