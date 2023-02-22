######################################
# Author : Come Peladeau
# V1.0
# 08/02/2023
######################################

import numpy as np
import scipy
import numpy.typing as npt

def phi1(
    impulse_response:npt.ArrayLike,
    num_bands:int,
    **kwargs
    ):
    """
    phi_1 function defined in [1, p. 365]. Estimates the flatness of the
    filter-bank response

    Parameters :
    ------------
    `impulse_response` : array-like
        Impulse response of the filter
    `num_bands` : int
        Number of bands
    
    Returns :
    ---------
    `output` : float

    References :
    ------------
    [1] Vaidyanathan, P. P. Multirate Systems and Filter Banks. Prentice-Hall
    Signal Processing Series. Englewood Cliffs, N.J: Prentice Hall, 1993.

    """
    if 'n_fft' in kwargs:
        n_fft = kwargs['n_fft']
    else:
        n_fft = 4096
    
    H = np.fft.rfft(impulse_response, n=n_fft)
    
    out = np.square(np.abs(H[:n_fft//(2*num_bands)])) # 1st term
    out += np.square(np.abs(H[n_fft//(2*num_bands)-1::-1]))
    out -= 1
    return np.sum(np.square(out))/n_fft*num_bands
    #return np.sum(np.square(np.abs(out)))

def phi2(
    impulse_response:npt.ArrayLike,
    num_bands:int,
    tol_transitionBand:float,
    **kwargs
    ):
    """
    phi_2 function defined in [1, p. 365]. Estimates the stop-band attenuation
    of each filter

    Parameters :
    ------------
    `impulse_response` : array-like
        Impulse response of the filter
    `num_bands` : int
        Number of bands
    `tol_transitionBand` : float
        tolerance of the transition band (normalized frequency)
    
    Returns :
    ---------
    `output` : float

    References :
    ------------
    [1] Vaidyanathan, P. P. Multirate Systems and Filter Banks. Prentice-Hall
    Signal Processing Series. Englewood Cliffs, N.J: Prentice Hall, 1993.

    """
    if 'n_fft' in kwargs:
        n_fft = kwargs['n_fft']
    else:
        n_fft = 4096
    
    H = np.fft.rfft(impulse_response, n=n_fft)
    n_start = n_fft/(4*num_bands)
    n_start += np.ceil(tol_transitionBand*n_fft)
    n_start = int(n_start)
    out = np.sum(np.square(np.abs(H[n_start:])))/(n_fft*(num_bands-1)/num_bands)
    return out
    #return np.sum(np.square(np.abs(H[n_start:])))

def phi(
    impulse_response_half:npt.ArrayLike, *args
    ):
    """
    Complete phi function defined in [1, p. 365]

    Parameters :
    ------------
    `impulse_response` : array-like
        Impulse response of the filter
    
    kwargs :
    --------
    `num_bands` : int
        Number of bands
    `tol_transitionBand` : float
        tolerance of the transition band (normalized frequency)
    `alpha` : float
        Balance between the two otpimization criterions
    
    Returns :
    ---------
    `output` : float

    References :
    ------------
    [1] Vaidyanathan, P. P. Multirate Systems and Filter Banks. Prentice-Hall
    Signal Processing Series. Englewood Cliffs, N.J: Prentice Hall, 1993.

    """
    num_bands = args[0]
    tol_transitionBand = args[1]
    alpha = args[2]
    N = 2*len(impulse_response_half)-1

    impulse_response = np.zeros(N)
    impulse_response[:N//2+1] = impulse_response_half[::-1]
    impulse_response[N//2:] = impulse_response_half[::1]

    out = alpha*phi1(
        impulse_response=impulse_response,
        num_bands=num_bands,
    )
    out += (1-alpha)*phi2(
        impulse_response=impulse_response,
        num_bands=num_bands,
        tol_transitionBand=tol_transitionBand
    )
    return out

def PQMF_prototype(num_bands, num_taps, tol_transitionBand, alpha = .5):

    prototype_init = np.sinc(
        (np.arange(num_taps)-num_taps//2)/(4*(num_bands-1))
        )*np.blackman(num_taps)
    prototype_init/=np.sum(prototype_init)
    prototype_half_init = prototype_init[num_taps//2:]

    result = scipy.optimize.minimize(
        fun = phi,
        x0 = prototype_half_init,
        args = (num_bands, tol_transitionBand, alpha),
    )
    prototype_half = result['x']
    prototype = np.zeros(num_taps)
    prototype[:num_taps//2+1] = prototype_half[::-1]
    prototype[num_taps//2:] = prototype_half[::1] 
    return prototype

def nearPR_CMFB(num_bands:int, num_taps:int = None, tol_transitionBand:float = None):
    """
    Returns near perfect reconstruction cosine-modulated filter bank
    impulse responses

    Parameters :
    ------------
    `num_bands` : int
        Number of bands in which the signal needs to be splitted
    `num_taps` : int
        Length of the filters, in samples
    `tol_transitionBand` : float
        Tolerance of the transition band of the filters

    Returns :
    ---------
    `filters` : array-like (num_bands, 10*num_bands-1)
        Impulse responses of the filters
    """
    # Number of filter taps
    if num_taps == None:
        num_taps = num_bands*8-1
    # Width of the transition band
    if tol_transitionBand == None:
        tol_transitionBand = .2/num_bands

    # Design of the protoype filter
    prototype_filter_IR = PQMF_prototype(
        num_bands = num_bands,
        num_taps = num_taps,
        tol_transitionBand = tol_transitionBand,
        alpha = .5)
    
    # Modulating the protoype filter to get the filter bank
    Analysis_filters = np.zeros((num_bands, num_taps))
    Synthesis_filters = np.zeros((num_bands, num_taps))
    for band_idx in range(num_bands):
        theta = np.power(-1, band_idx)*np.pi/4
        Analysis_filters[band_idx] = prototype_filter_IR*np.cos(
            np.pi*(np.arange(num_taps) - num_taps/2)*(band_idx+.5)/num_bands + theta
            ) *2
        Synthesis_filters[band_idx] = prototype_filter_IR*np.cos(
            np.pi*(np.arange(num_taps) - num_taps/2)*(band_idx+.5)/num_bands - theta
            ) *2
    return Analysis_filters, Synthesis_filters