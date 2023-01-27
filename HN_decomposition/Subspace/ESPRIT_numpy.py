import numpy as np
import numpy.typing as npt
import scipy

def LeastSquare(x:npt.ArrayLike, damp:npt.ArrayLike, redFreq:npt.ArrayLike):
    #Estimation des amplitudes et des phases
    N=len(x)
    num_poles=len(redFreq)
    t = np.arange(0,N)
    k=np.arange(0,num_poles)

    # Computing the Vandermonde matrix V

    log_V = t[:,None]*(damp+2j*np.pi*redFreq)
    V = np.exp(log_V)

    # Computing the complex vector of amplitudes and initial phases
    complexAmp =np.linalg.pinv(V)@x

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
    #mat_size = num_poles*2
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
        return np.sum(np.square(x))

def HRHATRAC_numpy(xChopped:npt.ArrayLike, num_poles:int, beta:float = 1, mu_L:float = .5, mu_V = .5):
    """
    Performs the HRAHATRAC algorithm [1] with the exponential window Fast Approximate Power Iteration subspace-tracker [2].
   
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
    [1] David, B., R. Badeau, and G. Richard. 'HRHATRAC Algorithm for Spectral Line Tracking of Musical Signals.'
    In 2006 IEEE International Conference on Acoustics Speed and Signal Processing Proceedings,
    3:III-45-III-48. Toulouse, France: IEEE, 2006. https://doi.org/10.1109/ICASSP.2006.1660586.

    [2] Badeau, R., B. David, and G. Richard. 'Fast Approximated Power Iteration Subspace Tracking.'
    IEEE Transactions on Signal Processing 53, no. 8 (August 2005): 2931-41. https://doi.org/10.1109/TSP.2005.850378.
    """

    # Getting the number of samples in each window and the number of samples
    window_size, num_windows = np.shape(xChopped)

    #Initializing output arrays
    poles_list = np.matrix(np.zeros((num_poles, num_windows)), dtype= 'complex128')
    complexAmp_list = np.matrix(np.zeros((num_poles, num_windows), dtype = 'complex128'))

    # Initializing
    W = np.matrix(np.concatenate((np.eye(num_poles, dtype = 'complex128'),np.zeros((window_size- num_poles, num_poles), dtype = 'complex128')), axis = 0), dtype = 'complex128')
    Psi = np.matrix(W[:-1,:].H@W[1:,:], dtype=  'complex128')
    
    Z = np.matrix(np.eye(num_poles, dtype = 'complex128'))
    poles = np.zeros(num_poles, dtype = 'complex128')

    for window_idx in range(num_windows):
        x = np.matrix(xChopped[:,window_idx], dtype = "complex128").T

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

        if window_idx < 1:
            poles, V = np.linalg.eig(Phi)
            Lambda = np.matrix(np.diag(poles), dtype = 'complex128')
            Lambda_inv = np.matrix(np.diag(1/poles), dtype = 'complex128')
        else:
            Lambda = (1-mu_L)*Lambda + mu_L*np.diag(np.diag(np.linalg.inv(V)@Phi@V))
            Lambda_inv = np.matrix(np.diag(1/np.diag(Lambda)), dtype='complex128')
            E_V = V - Phi@V@Lambda_inv
            V = (1-mu_V)*V + mu_V*(Phi@V@Lambda_inv + Phi.H@E_V@Lambda_inv.H)
            V = V/np.sqrt(np.sum(np.square(np.abs(V)), axis = 0)) # Normalization
        poles_list[:,window_idx] = np.diag(Lambda)[:,None]
    return poles_list