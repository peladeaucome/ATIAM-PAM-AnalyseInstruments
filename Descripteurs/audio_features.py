######################################
# Author : Come Peladeau
# V1.0
# 14/02/2023
######################################
import numpy as np
import numpy.typing as npt

def compute_spectralCentroid(
    x_fft:npt.ArrayLike,
    f:npt.ArrayLike
    ):
    """
    Computes the spectral centroid of the signal from its fft based on
    [1, p.135].
    
    Parameters :
    ------------
    `x_fft` : array-like
        FFT of the signal.
    `f` : array-like
        Frequencies.
    
    Returns :
    ---------
    `spectral_centroid` : float
        Spectral centroid of the signal.
    
    References :
    ------------
    [1] Klapuri, Anssi, and Manuel Davy, eds. Signal Processing Methods for
    Music Transcription. New York: Springer, 2006.
    """
    x_fft_norm = np.abs(x_fft)
    x_fft_norm /= np.sum(x_fft_norm)
    spectral_centroid:float = np.sum(x_fft_norm*f)/np.sum(x_fft_norm)
    return spectral_centroid

def compute_spectralBandwidth(
    x_fft:npt.ArrayLike,f:npt.ArrayLike,
    order:float = 2, centroid:float = None
    ):
    """
    Computes the spectral bandwidth of the signal from its fft based on
    [1, p.135].
    
    Parameters :
    ------------
    `x_fft` : array-like
        FFT of the signal.
    `f` : array-like
        Frequencies.
    `order` : float
        Order of the distance function used.
    `centroid` : float
        Pre-computed centroid. If `None`, the centroid will be computed inside
        this function.
    
    Returns :
    ---------
    `spectral_bandwidth` : float
        Spectral bandwidth of the signal.
    
    References :
    ------------
    [1] Klapuri, Anssi, and Manuel Davy, eds. Signal Processing Methods for
    Music Transcription. New York: Springer, 2006.
    """
    if centroid == None:
        centroid = compute_spectralCentroid(x_fft, f)
    
    x_fft_norm = np.abs(x_fft)
    x_fft_norm /= np.sum(x_fft_norm)

    spectral_bandwidth:float = np.power(
        np.sum(x_fft_norm*np.power(np.abs(f-centroid), order)),
        1/order
    )
    return spectral_bandwidth

def compute_RMS(x:npt.ArrayLike):
    """
    Computes the RMS level of a signal

    Parameters :
    ------------
    `x` : array-like
        input signal.
    
    Returns :
    ---------
    `RMS` : float
        RMS (root mean square) level of the signal.
    """
    RMS:float = np.sqrt(np.mean(np.square(x)))
    return RMS

def compute_ZCR(x:npt.ArrayLike, sr = 1):
    """
    Computes the zero-crossings rate of the input signal.
    
    Parameters :
    ------------
    `x` : array-like
        Input signal.
    
    Returns :
    ---------
    `ZCR` : float
        Zero-crossings rate of the signal.
    """
    crossings = 0
    for i in range(len(x)-1):
        if x[i+1]*x[i]<0:
            crossings+=1
    return crossings/len(x)*sr

def compute_features(
    x:npt.ArrayLike,
    sr:float,
    features_list:tuple,
    **kwargs):
    """
    Computes efficiently multiple audio feautures

    Parameters :
    ------------
    `x` : array-like
        Input signal.
    `sr` : float
        Sample rate.
    `features_list` : tuple of strings
        tuple or list containing the keys of the features to be computed.
    
    Returns :
    ---------
    `features_dict` : dict
        Dictionnary containing the requested features values.
    
    
    Supported feature keys :
    ------------------------
    `centroid` : spectral centroid.
    `bandwidth` : spectral bandwidth.
    `ZCR` : Zero-crossings rate.
    `RMS` : RMS level.

    Keyword arguments :
    -------------------
    `bandwidth_order` : float
        Order of the distance function used to compute the spectral bandwidth.
    """
    
    features_dict = {}

    if 'centroid' in features_list or 'bandwidth' in features_list:
        x_fft = np.fft.rfft(x)
        f = np.fft.rfftfreq(len(x), 1/sr)

    centroid = None
    if 'centroid' in features_list:
        centroid = compute_spectralCentroid(x_fft, f)
        features_dict['centroid'] = centroid
    
    if 'bandwidth' in features_list:
        if 'bandwidth_order' in kwargs:
            bandwidth_order = kwargs['bandwidth_order']
        else:
            bandwidth_order = 2
        
        bandwidth = compute_spectralBandwidth(
            x_fft = x_fft, 
            f = f,
            order = bandwidth_order,
            centroid = centroid)
        features_dict['bandwidth'] = bandwidth
    
    if 'ZCR' in features_list:
        ZCR = compute_ZCR(x, sr)
        features_dict['ZCR'] = ZCR
    
    if 'RMS' in features_list:
        RMS = compute_RMS(x)
        features_dict['RMS'] = RMS
    return features_dict
