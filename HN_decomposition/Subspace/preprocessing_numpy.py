import numpy as np
import librosa
import scipy.signal as sig
import scipy.linalg
import numpy.typing as npt

def rankFilter_stft(x_stft:npt.ArrayLike, rankFilter_bins:int, rankFilter_rank:float = .5):
    """
    Returns the noise PSD estimation

    args :
        - x_psd : array-like
            input STFT (PSD)
        - num_bins : int
            number of frequency bins over which the quantile filter is performed
        - rank : float
            rank of the quantile filter. Should be between 0 and 1
    
    returns :
        - X_filt : array-like
            STFT of the estimated noise
    """
    x_psd = np.square(np.abs(x_stft))
    rank_filtered_stft = np.zeros(np.shape(x_psd))
    for f in range(np.shape(x_psd)[0]):
        f_lowerBound = max(0, f-rankFilter_bins//2)
        f_upperBound = min(np.shape(x_psd)[0]-1, f+rankFilter_bins//2)
        rank_filtered_stft[f,:] = np.quantile(x_psd[f_lowerBound:f_upperBound], q = rankFilter_rank, axis = 0)
    return rank_filtered_stft


def compute_ARFilter(noise_psd:npt.ArrayLike, ARFilter_length:int):
    """
    Computes the auto-regressive filter coefficients to match a given noise power spectral density,
    based on the assumption that the noise is obtained by filtering white noise with an AR filter.
    Parameters :
    ------------
    `noise_psd` : array-like
        power spectral density of the noise.
    `ARFilter_length` : int
        number of coefficients of the AR filter.
    
    returns :
        - ARFilter_a : array-like
            Coeffeicients of the AR filter (the initial '1' is added so that this function is suitable to be used 
            as-is with the scipy.signal.lfilter function : scipy.signal.lfilter([1], ARFilter_a, x)).
    """
    # Computing the autocorrelation vector
    noise_autocorr_vec = np.fft.irfft(noise_psd)[:ARFilter_length+1]
    noise_autocorr_vec /= noise_autocorr_vec[0]
    # Autocorrelation vector
    r = noise_autocorr_vec[1: ARFilter_length+1]
    # Solving the Yule-Walker equations to get the AR filter coefficients
    ARFilter_a = np.concatenate((np.ones(1), -scipy.linalg.solve_toeplitz(noise_autocorr_vec[:ARFilter_length], r)))
    return ARFilter_a


def window_signal(x:npt.ArrayLike, window_length:int, hop_length:int):
    """
    Windows the input signal x

    Parameters :
    ------------
    `x` : array-like
        input signal
    `window_length` : int
        length of the windows
    `hop_length` : int
        number of samples gap between adjacent windows
    `window_type` : str
        window type. Default : hann
    
    returns :
    `xChopped` : array-like
        each time frame of the original signal
    """
    
    num_windows = int(np.ceil((len(x)-window_length)/hop_length))+1
    res = len(x) - (num_windows-1)*hop_length - window_length
    if res==0:
        num_windows-=1
        res = window_length

    # Initializing the output arrays
    xChopped = np.zeros((num_windows, window_length), dtype = 'complex128')

    for t in range(num_windows):
        if t == num_windows-1:
            xChopped[t, 0:res] = x[t*hop_length:t*hop_length+window_length]
        else:
            xChopped[t] = x[t*hop_length:t*hop_length+window_length]
    return xChopped


def whiten_signal(x:npt.ArrayLike, n_fft:int, rankFilter_bins:int, rankFilter_rank:int, ARFilter_length:int, threshold:float = 1e-6, window_type:str = 'hann'):
    """
    Applies a filter to the input signal so that the underlying noise has a flat spectrum.

    args :
        - x : array-like
            input signal
        - rankFilter_bins : int
            size of the rank filter
        - rankFilter_rank : float or int
            rank of the rank filter, which must be between 0 and 1
        - ARFilter_length : int
            size of he auto-regressive filter
        - threshold : float
            RMS threshold below which the signal is left unfiltered to avoid computation problems
        - window_type : str
            window type. Default : hann
    
    returns :
        - xWhitened : array-like, same size as x
            x which has been 'whitened'
        - ARFilters : array-like
            coefficients of the AR-filter
    """
    if window_type==None:
        window_type=='hann'
    

    _, x_psd = sig.welch(x, window = window_type, nperseg = n_fft, noverlap = 3*n_fft//4)
    noise_psd = np.zeros(np.shape(x_psd))
    for f in range(np.shape(x_psd)[0]):
        f_lowerBound = max(0, f-rankFilter_bins//2)
        f_upperBound = min(np.shape(x_psd)[0]-1, f+rankFilter_bins//2)
        noise_psd[f] = np.quantile(x_psd[f_lowerBound:f_upperBound], q = rankFilter_rank, axis = 0)
    
    # Initializing the output arrays
    xWhitened = np.zeros((np.shape(x)[0]), dtype = 'complex128')
    ARFilter = np.zeros((ARFilter_length+1))
    
    if np.std(x)>threshold:
        ARFilter = compute_ARFilter(noise_psd, ARFilter_length)
        #Filtering x
        xWhitened = sig.lfilter(ARFilter, [1], x)
    else:
        xWhitened = x
    return xWhitened, ARFilter


def compute_stft_from_windowed(xWhitened:npt.ArrayLike ,window_type:str ='hann'):
    """
    Computes the STFT from the whitened signal array

    Parameters :
    ------------
    `xWhitened` : array-like
        Whitened signal array
    `window_type` : str
        window type. Default : hann
    
    Returns :
    ---------
    `xWhitened_stft` : array-like
        STFT of the whitened signal
    """
    if window_type==None:
        window_type=='hann'
    xWhitened_stft = np.zeros((np.shape(xWhitened)[1]//2+1, np.shape(xWhitened)[0]), dtype = 'complex128')
    window = sig.get_window(window_type, np.shape(xWhitened)[1])
    for t in range(np.shape(xWhitened_stft)[1]):
        xWhitened_stft[:,t] = np.fft.rfft(xWhitened[t]*window)
    return xWhitened_stft


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
        num_taps = 16*num_bands-1
    # Width of the transition band
    if tol_transitionBand == None:
        tol_transitionBand = 1/(16*num_bands)

    # Design of the protoype filter
    prototype_filter_IR = sig.remez(
        numtaps = num_taps,
        bands = [
            0,
            1/(4*(num_bands-1)) - tol_transitionBand,
            1/(4*(num_bands-1)) + tol_transitionBand,
            1/2],
        fs = 1,
        type = 'bandpass',
        desired = [1, 0],
        weight= [1,1]
    )

    # Modulating the protoype filter to get the filter bank
    Analysis_filters = np.zeros((num_bands, num_taps))
    Synthesis_filters = np.zeros((num_bands, num_taps))
    for band_idx in range(num_bands):
        theta = np.power(-1, band_idx)*np.pi/4
        Analysis_filters[band_idx] = prototype_filter_IR*np.cos(
            np.pi*(np.arange(num_taps) - (num_taps)/2)*(band_idx+.5)/num_bands + theta
            ) *2
        Synthesis_filters[band_idx] = prototype_filter_IR*np.cos(
            np.pi*(np.arange(num_taps) - num_taps/2)*(band_idx+.5)/num_bands - theta
            ) *2
    return Analysis_filters, Synthesis_filters