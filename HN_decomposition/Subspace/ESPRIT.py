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
    amp = np.abs(complexAmp)
    phase =np.angle(complexAmp)
    
    return complexAmp



def ESPRIT(x:npt.ArrayLike,num_poles:int):
    """
    Performs the ESPRIT algorithm over the input signal
    args :
        - x : array-like
            input signal
        - signalSpace_dim : int
            dimension of the signal space 
    returns:
        - delta : array-like
            dampening factors
        - f
    """
    window_size = len(x)//2

    N = len(x)
    l = N-window_size+1

    X=scipy.linalg.hankel(x[:window_size],x[window_size-1:N])

    # Computing the autocorrelation matrix
    Rxx=(1/l)*(X@(np.conjugate(X.T)))

    # Diagonilising the autocorelation matrix
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