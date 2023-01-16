def filter_spectrum(X, num_bins, rank):
    """
    Returns the whitened STFT
    args :
        - X : input STFT
        - num_bins : number of frequency bins over which the quantile filter is performed
        - rank : rank of the quantile filter
    returns :
        - X_filt : filtered STFT
    """
    filtered = np.zeros(np.shape(X))
    for t in range(np.shape(X)[1]):
        for f in range(np.shape(X)[0]):
            f_lowerBound = max(0, f-num_bins//2)
            f_upperBound = min(np.shape(X)[0]-1, f+num_bins//2)
            filtered[f,t] = np.quantile(X[f_lowerBound:f_upperBound,t], q = rank)
    return filtered
