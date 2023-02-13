def compute_spectralCentroid(x_fft, f):
    x_fft_norm = np.abs(x_fft)
    x_fft_norm /= np.sum(x_fft_norm)
    return np.sum(x_fft_norm*f)/np.sum(x_fft_norm)

def compute_spectralBandwidth(x_fft, f, order, centroid = None):
    """
    from [1, p.135]

    References :
    ------------
    [1] Klapuri, Anssi, and Manuel Davy, eds. Signal Processing Methods for
    Music Transcription. New York: Springer, 2006.
    """
    if centroid == None:
        centroid = compute_spectralCentroid(x_fft, f)
    
    x_fft_norm = np.abs(x_fft)
    x_fft_norm /= np.sum(x_fft_norm)
    return np.power(np.sum(x_fft_norm*np.power(np.abs(f-centroid), order)), 1/order)

def compute_RMS(x):
    return np.sqrt(np.mean(np.square(x)))

def compute_ZCR(x):
    crossings = 0
    for i in range(len(x)-1):
        if x[i+1]*x[i]<0:
            crossings+=1
    return crossings/len(x)

def compute_features(x, sr, features_list, **kwargs):
    if 'bandwidth_order' in kwargs:
        bandwidth_order = kwargs['bandwidth_order']
    else:
        bandwidth_order = 2
    
    out = []

    x_fft = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1/sr)
    centroid = None
    if 'centroid' in features_list:
        centroid = compute_spectralCentroid(x_fft, f)
        out.append(centroid)
    
    if 'bandwidth' in features_list:
        bandwidth = compute_spectralBandwidth(
            x_fft = x_fft, 
            f = f,
            order = bandwidth_order,
            centroid = centroid)
        out.append(bandwidth)
    
    if 'ZCR' in features_list:
        ZCR = compute_ZCR(x)
        out.append(ZCR)
    
    if 'RMS' in features_list:
        RMS = compute_RMS(x)
        out.append(RMS)
    
    return out
