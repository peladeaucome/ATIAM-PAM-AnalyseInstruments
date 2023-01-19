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

def whiten_signal(x:npt.ArrayLike, n_fft:int, hop_length:int, rankFilter_bins:int, rankFilter_rank:int, ARFilter_length:int, threshold:float = 1e-6, window_type:str = 'hann'):
    """
    Whitens each window of x
    args :
        - x : array-like
            input signal
        - n_fft : int
            number of samples each fft is computed over
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
            coeeficients of the ARFilters for each time frame
    """
    x_stft = librosa.stft(
        x,
        n_fft = n_fft,
        hop_length=hop_length,
        center=False,
        window=window_type
    )
    noise_psd = rankFilter_stft(
        x_stft = x_stft,
        rankFilter_bins = rankFilter_bins,
        rankFilter_rank = rankFilter_rank
    )
    
    # Initializing the output arrays
    xWhitened = np.zeros((n_fft, np.shape(x_stft)[1]))
    xChopped = np.zeros((n_fft, np.shape(x_stft)[1]))
    ARFilters = np.zeros((ARFilter_length+1, np.shape(x_stft)[1]))
    for t in range(np.shape(x_stft)[1]):
        x_windowed = x[t*hop_length:t*hop_length+n_fft]
        if np.std(x_windowed)>threshold:
            ARFilter_a = compute_ARFilter(noise_psd[:,t], ARFilter_length)
            ARFilters[:,t] = ARFilter_a
            #Filtering x
            xWhitened[:,t] = sig.lfilter(ARFilter_a, [1], x_windowed)
        else:
            xWhitened[:,t] = x_windowed
        xChopped[:,t] = x_windowed
    return xWhitened, xChopped, ARFilters


def compute_stft_from_whitened(xWhitened:npt.ArrayLike ,window_type:str ='hann'):
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
    xWhitened_stft = np.zeros((np.shape(xWhitened)[0]//2+1, np.shape(xWhitened)[1]), dtype = 'complex128')
    window = sig.get_window(window_type, np.shape(xWhitened)[0])
    for t in range(np.shape(xWhitened_stft)[1]):
        xWhitened_stft[:,t] = np.fft.rfft(xWhitened[:,t]*window)
    return xWhitened_stft
