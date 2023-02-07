import numpy as np
import scipy.signal as sig
import scipy.linalg
import numpy.typing as npt
import scipy

def LeastSquare(x:npt.ArrayLike, damp:npt.ArrayLike, redFreq:npt.ArrayLike):
    """
    Estimates the complex amplitudes of the signal poles given their frequencies
    and damping (or growing) factors using the least-squares method :

    Parameters :
    ------------
    `x` : array-like
        Input signal
    `damp` : array-like
        damping (or growing) factors
    `redFreq` : array-like
        Normlized frequencies
    
    Returns :
    ---------
    `complexAmp` : array-like, complex
        complex amplitudes of the poles

    """
    #Estimation des amplitudes et des phases
    N=len(x)
    num_poles=len(redFreq)
    t = np.arange(0,N)

    # Computing the Vandermonde matrix V

    log_V = t[:,None]*(damp+2j*np.pi*redFreq)
    V = np.exp(log_V)

    # Computing the complex vector of amplitudes and initial phases
    complexAmp = np.linalg.pinv(V)@x

    # Separating the amplitudes and phases
    # amp = np.abs(complexAmp)
    # phase =np.angle(complexAmp)
    
    return complexAmp

def ESPRIT(x:npt.ArrayLike,num_poles:int):
    """
    Performs the ESPRIT algorithm over the input signal
    
    args :
        - x : array-like [signal_length]
            input signal
        - num_poles : int
            number of poles to find

    returns:
        - poles : array-like
            Complex poles of the signal
        - complexAmp : array-like
            Complex amplitudes of the signal
        - Lambda : array-like
            eigen-values of the autocorrelation matrix
    """
    mat_size = len(x)//2
    #mat_size = num_poles*4
    N = len(x)
    l = N-mat_size+1

    X=scipy.linalg.hankel(x[:mat_size],x[mat_size-1:N])

    # Computing the autocorrelation matrix
    Rxx=(1/l)*(X@(np.conjugate(X.T)))

    # Diagonilizing the autocorelation matrix
    U1,Lambda,U2 = np.linalg.svd(Rxx)
    
    # Creating a base of the signal space
    W=U1[:,:num_poles]

    W_down=W[:-1,:]
    W_up=W[1:,:]


    # Creating a matrix phi
    phi = np.linalg.pinv(W_down)@W_up

    # Computing its singular values (which are the signal poles)
    poles = np.linalg.eig(phi)[0]
    damp = np.log(np.abs(poles))
    redFreq = (1/(2*np.pi))*np.angle(poles)
    
    complexAmp = LeastSquare(x, damp, redFreq)
    return poles, complexAmp, Lambda


def norm2(x):
        return np.sum(np.square(np.abs(x)))

def ddiag(x):
    return np.diag(np.diag(x))

def HRHATRAC(xChopped:npt.ArrayLike, num_poles:int, beta:float = 1,
                   mu_L:float = .5, mu_V:float = .5):
    """
    Performs the HRAHATRAC algorithm [1] with the exponential window Fast
    Approximate Power Iteration subspace-tracker [2].
   
    args :
        - x : array-like
            input signal
        - num_poles : int
            dimension of the signal space 
        - beta : float
            forgetting factor
        - mu_L : float

        - mu_V : float

    returns:
        - poles_list : array-like
            List of the poles for each time frame
        - complexAmp_list : array-like
            List of the domplex amplitudes of the signal poles
        
    References :

    [1] David, B., R. Badeau, and G. Richard. 'HRHATRAC Algorithm for Spectral
    Line Tracking of Musical Signals.' In 2006 IEEE International Conference on
    Acoustics Speed and Signal Processing Proceedings, 3:III-45-III-48.
    Toulouse, France: IEEE, 2006. https://doi.org/10.1109/ICASSP.2006.1660586.

    [2] Badeau, R., B. David, and G. Richard. 'Fast Approximated Power Iteration
    Subspace Tracking.' IEEE Transactions on Signal Processing 53, no. 8
    (August 2005): 2931-41. https://doi.org/10.1109/TSP.2005.850378.
    """

    # Getting the number of samples in each window and the number of samples
    num_windows, window_size  = np.shape(xChopped)

    #Initializing output arrays
    poles_list = np.matrix(np.zeros((num_windows, num_poles)),
                                    dtype= 'complex128')
    #complexAmp_list = np.matrix(np.zeros((num_poles, num_windows),
    #                                     dtype = 'complex128'))

    # Initializing
    W = np.matrix(np.concatenate((np.eye(num_poles, dtype = 'complex128'),
                                  np.zeros((window_size- num_poles, num_poles),
                                           dtype = 'complex128')), axis = 0),
                  dtype = 'complex128')
    
    Psi = np.matrix(W[:-1,:].H@W[1:,:], dtype=  'complex128')
    
    Z = np.matrix(np.eye(num_poles, dtype = 'complex128'))
    poles = np.zeros(num_poles, dtype = 'complex128')

    for window_idx in range(num_windows):
        x = np.matrix(xChopped[window_idx], dtype = "complex128").T

        # FAPI
        y = W.H@x
        h = Z@y
        g = h/(beta+(y.H@h))
        eps2 = norm2(x)-norm2(y)
        norm2g = norm2(g)
        tau = eps2/(1+eps2*norm2g + np.sqrt(1+eps2*norm2g))
        eta = 1-tau*norm2g
        yprim = eta*y + tau*g
        hprim = Z.H@yprim
        eps_bf = (tau/eta)*(Z@g - (hprim.H@g)[0,0]*g)
        Z = (Z - g@hprim.H + eps_bf@g.H)/beta
        e = eta*x - W@yprim
        W1 = np.matrix(W, dtype = 'complex128')
        W += e@g.H
        
        ## Tracking the spectral matrix
        e_minus = W1[:-1,:].H@e[1:,:]
        e_plus =  W1[1:,:].H@e[:-1,:]
        e_plusprim =  e_plus + g*((e[1:,:].H@e[:-1,:])[0,0])
        
        nu = W[-1].H
        Psi += e_minus@g.H + g@e_plusprim.H
        
        phi = Psi.H@nu
        Phi = Psi + nu@phi.H/(1 - norm2(nu))

        # Updating the poles
        if window_idx == 0:
            poles, V = np.linalg.eig(Phi)
            Lambda = np.matrix(np.diag(poles), dtype = 'complex128')
            Lambda_inv = np.matrix(np.diag(1/poles), dtype = 'complex128')
        else:
            Lambda = (1-mu_L)*Lambda + mu_L*ddiag(np.linalg.inv(V)@Phi@V)
            Lambda_inv = np.matrix(np.diag(1/np.diag((Lambda))),
                                   dtype='complex128')
            E_V = V - Phi@V@Lambda_inv
            V = (1-mu_V)*V + mu_V*(Phi@V@Lambda_inv + Phi.H@E_V@Lambda_inv.H)
            # Normalizing the columns of V to avoid numerical instability
            V = V/np.sqrt(np.sum(np.square(np.abs(V)), axis = 0))
        poles_list[window_idx] = np.diag(Lambda)
    return poles_list

def HN_FAPI(xChopped:npt.ArrayLike, num_poles:int, beta:float,
              xChoppedWhitened:npt.ArrayLike = None ,hop_length:int = 64,
              window_type:str = 'hann'):
    """
    Performs an harmonic+noise decomposition by projecting onto the signal space
    estimated using the Fast Approximated Power Iteration (FAPI) algorithm
    
    args :
        - xChopped : array-like
            input signal
        - num_poles : int
            dimension of the signal space onto which the signal is projected
        - beta:float
            Forgetting factor of the HRHATRAC algorithm
        - xChoppedWhitened : array-like
            preprocessed input siganl
        - hop_length : int
            hop length between consecutive time frames
        - window_type : str
            type of window to use. Uses `scipy.signal.get_window`.

    returns:
        - xHarmo : array-like
            projection of the input signal onto the signal subspace
        - xNoise : array-like
            projection of the input signal onto the noise subspace
    """

    # Dimensions 
    num_windows, window_length = np.shape(xChopped)
    mat_size = window_length//2
    l = window_length-mat_size+1
    
    # Squared Window
    window2 = np.square(scipy.signal.get_window(window_type, window_length))
    
    # Initializing output arrays
    xHarmo = np.zeros(
        hop_length*num_windows + window_length,
        dtype = 'complex128'
    )
    xNoise = np.zeros(
        hop_length*num_windows + window_length,
        dtype = 'complex128'
    )

    
    # Initializing FAPI
    W = np.matrix(np.concatenate((np.eye(num_poles, dtype = 'complex128'),
                                  np.zeros(
                                    (window_length - num_poles, num_poles),
                                           dtype = 'complex128')), axis = 0),
                  dtype = 'complex128')
    Z = np.matrix(np.eye(num_poles, dtype = 'complex128'))

    for window_idx, x in enumerate(xChopped):
        # Input Array
        xWhitened = np.matrix(
            xChoppedWhitened[window_idx],
            dtype = "complex128").T
        
        # FAPI algorithm
        y = W.H@xWhitened
        h = Z@y
        g = h/(beta+(y.H@h))
        eps2 = norm2(xWhitened)-norm2(y)
        norm2g = norm2(g)
        tau = eps2/(1+eps2*norm2g + np.sqrt(1+eps2*norm2g))
        eta = 1-tau*norm2g
        yprim = eta*y + tau*g
        hprim = Z.H@yprim
        eps_bf = (tau/eta)*(Z@g - (hprim.H@g)[0,0]*g)
        Z = (Z - g@hprim.H + eps_bf@g.H)/beta
        e = eta*xWhitened - W@yprim
        W += e@g.H
        projection_matrix_harmo = W@(W.H)
        projection_matrix_noise = np.eye(window_length)-projection_matrix_harmo

        # Overlap-Add method
        start_idx = window_idx*hop_length
        end_idx = window_idx*hop_length+window_length

        xHarmo[start_idx:end_idx] += window2*np.array(
            np.dot(projection_matrix_harmo, x[:,None]).T)[0]
        xNoise[start_idx:end_idx] += window2*np.array(
            np.dot(projection_matrix_noise, x[:,None]).T)[0]
    return xHarmo, xNoise

def HN_ESPRIT(xChopped:npt.ArrayLike, num_poles:int,
              xChoppedWhitened:npt.ArrayLike = None ,hop_length:int = 64,
              window_type:str = 'hann'):
    """
    Performs an harmonic+noise decomposition by projecting onto the signal
    estimated by digonalizing the correlation matrix of the signal.
    
    Args :
    ------
    `xChopped` : array-like
        Input signal
    `num_poles` : int
        Dimension of the signal space onto which the signal is projected
    `xChoppedWhitened` : array-like
        Preprocessed input siganl
    `hop_length` : int
        Hop length between consecutive time frames
    `window_type` : str
        Type of window to use. Uses `scipy.signal.get_window`.

    Returns :
    ---------
    `xHarmo` : array-like
        Projection of the input signal onto the signal subspace
    `xNoise` : array-like
        Projection of the input signal onto the noise subspace
    """
    num_windows, window_length = np.shape(xChopped)

    #if xChoppedWhitened == None:
    #    xChoppedWhitened=xChopped
    
    window2 = np.square(scipy.signal.get_window(window_type, window_length))
    
    xHarmo = np.zeros(hop_length*num_windows + window_length, dtype = 'complex128')
    xNoise = np.zeros(hop_length*num_windows + window_length, dtype = 'complex128')
    mat_size = window_length//2

    l = window_length-mat_size+1
    for window_idx, x in enumerate(xChopped):
        xWhitened = xChoppedWhitened[window_idx]
        X=scipy.linalg.hankel(xWhitened[:mat_size],xWhitened[mat_size-1:window_length])
        # Computing the autocorrelation matrix
        Rxx=(1/l)*(X@(np.conjugate(X.T)))

        # Diagonilizing the autocorelation matrix
        U1,Lambda,U2 = np.linalg.svd(Rxx)
        # Creating a base of the signal space
        W= np.matrix(U1[:,:num_poles])
        projection_matrix_harmo = W@W.H
        projection_matrix_noise = np.eye(mat_size)-projection_matrix_harmo

        xHarmo_chopped = np.zeros(window_length, dtype = 'complex128')
        xNoise_chopped = np.zeros(window_length, dtype = 'complex128')

        xHarmo_chopped[0:mat_size] = x[:mat_size]@projection_matrix_harmo.T
        xNoise_chopped[0:mat_size] = x[:mat_size]@projection_matrix_noise.T
        xHarmo_chopped[mat_size:] = x[mat_size:]@projection_matrix_harmo.T
        xNoise_chopped[mat_size:] = x[mat_size:]@projection_matrix_noise.T

        xHarmo[window_idx*hop_length:window_idx*hop_length+window_length] += window2*xHarmo_chopped
        xNoise[window_idx*hop_length:window_idx*hop_length+window_length] += window2*xNoise_chopped
    return xHarmo, xNoise


def HN_FAPI(xChopped:npt.ArrayLike, num_poles:int, beta:float,
              xChoppedWhitened:npt.ArrayLike = None ,hop_length:int = 64,
              window_type:str = 'hann'):
    """
    Performs an harmonic+noise decomposition by projecting onto the signal space
    estimated using the Fast Approximated Power Iteration (FAPI) algorithm
    
    args :
        - xChopped : array-like
            input signal
        - num_poles : int
            dimension of the signal space onto which the signal is projected
        - xChoppedWhitened : array-like
            preprocessed input siganl
        - hop_length : int
            hop length between consecutive time frames
        - window_type : str
            type of window to use. Uses `scipy.signal.get_window`.

    returns:
        - xHarmo : array-like
            projection of the input signal onto the signal subspace
        - xNoise : array-like
            projection of the input signal onto the noise subspace
    """

    # Dimensions 
    num_windows, window_length = np.shape(xChopped)
    mat_size = window_length//2
    l = window_length-mat_size+1
    
    # Squared Window
    window2 = np.square(scipy.signal.get_window(window_type, window_length))
    
    # Initializing output arrays
    xHarmo = np.zeros(
        hop_length*num_windows + window_length,
        dtype = 'complex128'
    )
    xNoise = np.zeros(
        hop_length*num_windows + window_length,
        dtype = 'complex128'
    )

    
    # Initializing FAPI
    W = np.matrix(np.concatenate((np.eye(num_poles, dtype = 'complex128'),
                                  np.zeros(
                                    (window_length - num_poles, num_poles),
                                           dtype = 'complex128')), axis = 0),
                  dtype = 'complex128')
    Z = np.matrix(np.eye(num_poles, dtype = 'complex128'))

    for window_idx, x in enumerate(xChopped):
        # Input Array
        xWhitened = np.matrix(
            xChoppedWhitened[window_idx],
            dtype = "complex128").T
        
        # FAPI algorithm
        y = W.H@xWhitened
        h = Z@y
        g = h/(beta+(y.H@h))
        eps2 = norm2(xWhitened)-norm2(y)
        norm2g = norm2(g)
        tau = eps2/(1+eps2*norm2g + np.sqrt(1+eps2*norm2g))
        eta = 1-tau*norm2g
        yprim = eta*y + tau*g
        hprim = Z.H@yprim
        eps_bf = (tau/eta)*(Z@g - (hprim.H@g)[0,0]*g)
        Z = (Z - g@hprim.H + eps_bf@g.H)/beta
        e = eta*xWhitened - W@yprim
        W += e@g.H
        projection_matrix_harmo = W@(W.H)
        projection_matrix_noise = np.eye(window_length)-projection_matrix_harmo

        # Overlap-Add method
        start_idx = window_idx*hop_length
        end_idx = window_idx*hop_length+window_length

        xHarmo[start_idx:end_idx] += window2*np.array(
            np.dot(x, projection_matrix_harmo.T))[0]
        xNoise[start_idx:end_idx] += window2*np.array(
            np.dot(x, projection_matrix_noise.T))[0]
    return xHarmo, xNoise

        
        
def ESTER(
    x:npt.ArrayLike,
    start_idx:int = 0,
    window_length:int = 512,
    max_poles:int = 255,
    factor:float = 10.
    ):
    """
    Returns the optimal amount of signal poles to estimate with the ESPRIT
    algorithm using the ESTimation ERror (ESTER) [1] criterion
    
    Args :
    ------
    `x` : `array-like`
        Input signal
    `window_length` : `int`
        Length of the window on which the ESTER criterion is computed
    `max_poles` : `int`
        Max number of poles that can be estimated
    `factor` : `float`
        Threshold for the ESTER criterion
    
    Returns :
    ---------

    
    References :
    ------------

    [1] Badeau, R., B. David, and G. Richard. “A New Perturbation Analysis for
    Signal Enumeration in Rotational Invariance Techniques.” IEEE Transactions
    on Signal Processing 54, no. 2 (February 2006): 450-58.
    https://doi.org/10.1109/TSP.2005.861899.
"""
    
    mat_size = window_length//2
    l = window_length-mat_size+1

    #Starting after the attack in order to have most of the modes
    #start_idx = np.argmax(np.abs(scipy.signal.hilbert(x))) + 100
    
    x_windowed = x[start_idx:start_idx+window_length]

    # Initialization
    ## Signal matrix
    X = np.matrix(scipy.linalg.hankel(
            x_windowed[:mat_size],
            x_windowed[mat_size-1:window_length]
        ))

    ## Computing the autocorrelation matrix
    Rxx=(1/l)*(X@X.H)

    ## Diagonilizing the autocorelation matrix
    U1,Lambda,_ = np.linalg.svd(Rxx)
    
    ## Creating a base of the signal space
    W = np.matrix(U1[:,:max_poles])
    J_list = np.zeros(max_poles+1)
    J_global_max = 0

    phi = np.array([[0]], dtype = np.complex128)
    Xi = np.array([[]], dtype= np.complex128)
    for num_poles in range(0, max_poles):
        # Update of the auxiliary matrix Psi
        W_down = W[:-1,:num_poles]
        W_up = W[1:,:num_poles]

        w_down = W[:-1,num_poles]
        w_up = W[1:,num_poles]

        nu = W[-1,:num_poles].T
        mu = W[-1, num_poles]
        
        phi_r = W_down.H@w_up
        phi_l = W_up.H@w_down
        phi_lr = (w_down.H@w_up)[0,0]

        # Update of the auxiliary matrix Xi
        eps = w_up - W_down@phi_r - phi_lr*w_down

        Xi = np.concatenate((Xi - w_down@phi_l.H, eps), axis = 1)

        # Computaion of E from Xi
        #if num_poles == 1:
        #    phi = phi_r.H@nu + mu*phi_lr
        #else:
        phi = np.concatenate((phi + mu*phi_l, phi_r.H@nu + mu*phi_lr), axis = 0)

        E = Xi - ((W_down@nu)@phi.H)/(1-norm2(nu))

        J = 1/norm2(E)
        J_list[num_poles+1] = J

    J_max_norm = np.amax(J_list/factor)
    for i in range(1,len(J_list)-1):
        if J_list[i]>J_max_norm and J_list[i]>J_list[i-1] and J_list[i]>J_list[i+1]:
            estimated_order = i
    return estimated_order, J_list



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


def multiband_HN(
    x,
    window_length:int = 32,
    hop_length:int = None,
    window_type:str = 'hann',
    ester_factor:int = 10,
    num_bands:int = 16, **kwargs):
    """
    Performs the full H+N decomposition using sub-band filtering, per band noise
    whitenening.
    Parameters :
    ------------
    `x` : array-like
        Input signal
    `window_length` : int
        Length of the windows used for the overlap-add method
    `ester_factor` : float
        used for he automatic pole number computation. The higher the factor is,
        the higher the number of poles used will be.
    `num_bands` : int
        number of frequency bands
    
    Returns :
    ---------
    `xHarmo` : array-like
        Projection of `x` onto the signal space
    `xNoise` : array-like
        Projection of `x` onto the noise space
    """
    
    # kwargs
    if hop_length==None:
        hop_length = window_length//4
    
    if "bandFilters_taps" in kwargs:
        bandFilters_taps = kwargs["bandFilters_taps"]
    else:
        bandFilters_taps = None
    
    if "bandFilters_tol" in kwargs:
        bandFilter_tol = kwargs["bandFilters_tol"]
    else:
        bandFilter_tol = None
    
    if "window_type" in kwargs:
        window_type = kwargs["window_type"]
    else:
        window_type = 'hann'
    
    if "tracking_method" in kwargs:
        tracking_method = kwargs["tracking_method"]
        if tracking_method=="FAPI":
            if "FAPI_beta" in kwargs:
                FAPI_beta = kwargs["FAPI_beta"]
            else:
                FAPI_beta = .95
    else:
        tracking_method = "classic"

    Analysis_filters, Synthesis_filters = nearPR_CMFB(
        num_bands = num_bands,
        num_taps = bandFilters_taps,
        tol_transitionBand = bandFilter_tol
    )


    max_poles = 25*window_length//64

    ## Initializing output vectors
    xHarmo = np.zeros(len(x), dtype = np.complex128)
    xNoise = np.zeros(len(x), dtype = np.complex128)
    for band_idx in range(num_bands):

        xDecimated = sig.lfilter(Analysis_filters[band_idx], [1], x)[::num_bands]

        start_idx = np.argmax(np.abs(sig.hilbert(xDecimated)))
        xWhitened, _ = whiten_signal(
            x = xDecimated,
            n_fft = 512,
            rankFilter_bins = 300,
            rankFilter_rank = .3,
            ARFilter_length = 12
        )
    
        num_poles, _ = ESTER(
            x = xDecimated,
            start_idx=start_idx,
            window_length=window_length,
            max_poles = max_poles,
            factor = ester_factor
        )
        xChoppedWhitened = window_signal(
            x = xWhitened,
            window_length = window_length,
            hop_length = hop_length
        )

        xChopped = window_signal(
            x = xDecimated,
            window_length = window_length,
            hop_length = hop_length
        )
        
        if tracking_method == "classic":
            xHarmo_band, xNoise_band = HN_ESPRIT(
                xChoppedWhitened = xChoppedWhitened,
                num_poles = num_poles,
                xChopped = xChopped,
                hop_length = hop_length,
                window_type = window_type)
        elif tracking_method == "FAPI":
            xHarmo_band, xNoise_band = HN_FAPI(
                xChoppedWhitened = xChoppedWhitened,
                num_poles = num_poles,
                xChopped = xChopped,
                hop_length = hop_length,
                beta = FAPI_beta,
                window_type = window_type)

        
        xHarmoInsert = np.zeros(len(x), dtype = np.complex128)
        xHarmoInsert[::num_bands] = xHarmo_band[:len(x)//num_bands+1]
        
        xNoiseInsert = np.zeros(len(x), dtype = np.complex128)
        xNoiseInsert[::num_bands] = xNoise_band[:len(x)//num_bands+1]

        xHarmo += sig.lfilter(Synthesis_filters[band_idx], [1], xHarmoInsert)*num_bands
        xNoise += sig.lfilter(Synthesis_filters[band_idx], [1], xNoiseInsert)*num_bands

    return xHarmo, xNoise


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
        n_fft = 2048
    
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
        n_fft = 2048
    
    H = np.fft.rfft(impulse_response, n=n_fft)
    n_start = n_fft/(4*num_bands)
    n_start += np.ceil(tol_transitionBand*n_fft)
    n_start = int(n_start)
    return np.sum(np.square(np.abs(H[n_start:])))/(n_fft*(num_bands-1)/num_bands)
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
    out+=(1-alpha)*phi2(
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
        num_taps = num_bands*8 - (1*(num_bands%2==0))
    # Width of the transition band
    if tol_transitionBand == None:
        tol_transitionBand = .3/num_bands

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
