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
    args :
        - noise_psd : array-like
            power spectral density of the noise.
        - ARFilter_length : int
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
    #ARFilter_a = np.concatenate((np.ones(1), -np.dot(scipy.linalg.inv(R), r)))
    return ARFilter_a

def window_and_whiten_signal(x:npt.ArrayLike, window_length:int, hop_length:int, rankFilter_bins:int, rankFilter_rank:int, ARFilter_length:int, threshold:float = 1e-6, window_type:str = 'hann'):
    """
    Whitens each window of x

    args :
        - x : array-like
            input signal
        - window_length : int
            length of the windows
        - hop_length : int
            number of samples gap between adjacent windows
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
            x which has been 'whitened' on each window
        - xChopped : array-like
            each time frame of the original signal
        - ARFilters : array-like
            coefficients of the AR-filters for each time frame
    """
    x_stft = librosa.stft(
        np.real(x),
        n_fft = window_length,
        hop_length=hop_length,
        center=False,
        window=window_type
    )

    #_, _, x_stft = sig.stft(
    #    x,
    #    fs=1,
    #    window = window_type,
    #    nperseg = window_length,
    #    noverlap = window_length-hop_length,
    #    nfft = window_length,
    #    detrend = False,
    #    return_onesided = True,
    #    scaling = 'spectrum'
    #)
    #print(x_stft.shape)
    #x_stft = x_stft.T
    #print(x_stft.shape)
    
    noise_psd = rankFilter_stft(
        x_stft = x_stft,
        rankFilter_bins = rankFilter_bins,
        rankFilter_rank = rankFilter_rank
    )
    
    # Initializing the output arrays
    xWhitened = np.zeros((np.shape(x_stft)[1], window_length), dtype = 'complex128')
    xChopped = np.zeros((np.shape(x_stft)[1], window_length), dtype = 'complex128')
    ARFilters = np.zeros((np.shape(x_stft)[1], ARFilter_length+1))

    for t in range(np.shape(x_stft)[1]):
        x_windowed = x[t*hop_length:t*hop_length+window_length]
        if np.std(x_windowed)>threshold:
            ARFilter_a = compute_ARFilter(noise_psd[:,t], ARFilter_length)
            ARFilters[t] = ARFilter_a
            #Filtering x
            xWhitened[t] = sig.lfilter(ARFilter_a, [1], x_windowed)
        else:
            xWhitened[t] = x_windowed
        xChopped[t] = x_windowed
    return xWhitened, xChopped, ARFilters


def window_signal(x:npt.ArrayLike, window_length:int, hop_length:int):
    """
    Windows the input signal x

    args :
        - x : array-like
            input signal
        - window_length : int
            length of the windows
        - hop_length : int
            number of samples gap between adjacent windows
        - window_type : str
            window type. Default : hann
    
    returns :
        - xChopped : array-like
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
    args :
        - xWhitened : array-like
            Whitened signal array
        - window_type : str
            window type. Default : hann
    returns
        - xWhitened_stft : array-like
            STFT of the whitened signal
    """
    if window_type==None:
        window_type=='hann'
    xWhitened_stft = np.zeros((np.shape(xWhitened)[1]//2+1, np.shape(xWhitened)[0]), dtype = 'complex128')
    window = sig.get_window(window_type, np.shape(xWhitened)[1])
    for t in range(np.shape(xWhitened_stft)[1]):
        xWhitened_stft[:,t] = np.fft.rfft(xWhitened[t]*window)
    return xWhitened_stft
