import numpy as np
import scipy.signal as sig

def filter_n_decimate(filterLength_smp : int, numBands : int, inputSignal : np.array):
    """
    args :
        - filterLength : [int]
            Digital filter length, in samples
        - numBands : [int]
            Number of frequency channels in which the signal is to be separated
        - inputSignal : [(1, N) np.array]
            Input signal 
    returns :
        - outputSignals : [numBands, N) np.array]
            Output signal"""
    signalLength_smp = np.shape(inputSignal) #Input signal length
    outputSignal = np.zeros((signalLength_smp, numBands))

    for i in range(numBands):
        outputSignal[:,i] = inputSignal

    return (outputSignal)