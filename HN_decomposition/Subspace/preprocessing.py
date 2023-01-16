import numpy as np
import librosa
import scipy.signal as sig
import scipy.linalg

def rankFilter_stft(x_psd, rankFilter_bins, rankFilter_rank = .5):
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
    rank_filtered = np.zeros(np.shape(x_psd))
    for f in range(np.shape(x_psd)[0]):
        f_lowerBound = max(0, f-rankFilter_bins//2)
        f_upperBound = min(np.shape(x_psd)[0]-1, f+rankFilter_bins//2)
        rank_filtered[f,:] = np.quantile(x_psd[f_lowerBound:f_upperBound], q = rankFilter_rank, axis = 0)

    return rank_filtered

def whiten_spectrum(x, n_fft, rankFilter_bins, rankFilter_rank, ARFilter_length, threshold = 1e-6):
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
    
    returns :
        - xWhitened : array-like, same size as x
            x which has been 'whitened' on each window
    """
    x_stft = librosa.stft(x, n_fft = n_fft, hop_length=n_fft, center=False)
    x_stft_rankFiltered = rankFilter_stft(
        x_psd = np.square(np.abs(x_stft)),
        rankFilter_bins = rankFilter_bins,
        rankFilter_rank = rankFilter_rank
    )
    
    # Initializing the output vector
    xWhitened = np.zeros(len(x))
    for t in range(np.shape(x_stft)[1]):
        x_windowed = x[t*n_fft:(t+1)*n_fft]
        if np.std(x_windowed)>threshold:
            # Computing the autocorrelation vector
            noise_autocorr_vec = np.fft.irfft(x_stft_rankFiltered[:,t])[:ARFilter_length+1]
            # Autocorrelation matrix to solve the Yule-Walker Equation
            R = scipy.linalg.toeplitz(noise_autocorr_vec[:ARFilter_length])
            r = noise_autocorr_vec[1: ARFilter_length+1]
            # Solving the Yule-Walker equations to get the AR filter coefficients
            AR_filt = np.concatenate((np.ones(1), -np.dot(scipy.linalg.inv(R), r)))
            #Filtering x
            xWhitened[t*n_fft:(t+1)*n_fft] = sig.lfilter(AR_filt, [1], x_windowed)
        else:
            xWhitened[t*n_fft:(t+1)*n_fft] = x_windowed
    return xWhitened