import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt

def find_nearest_index(array, value, nearest_value=False):
    """
    # Inputs
    Takes an array and a value as arguments.

    # Ouputs
    Return the index of the nearest value. Also returns (nearest_idx, nearest_value) if nearest_value is True.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def compute_tfct_from_file(file,Nhop,Nwin,Nfft) :
    """
    # Inputs
    Takes a path to an audio WAV file, Nhop, Nwin, Nfft to compute STFT.

    # Ouputs
    Returns the 2D numpy matrix of the STFT, fs the samplerate frequency, the time series associated with the STFT, and frequencies

    return (xmat, fs, time, frequencies)
    """
    fs,sig = sp.wavfile.read(file)
    sig=sig/(2**(16-1))
    
    Nsig=len(sig)
    L=int((Nsig-Nwin+Nhop)/Nhop)
    if Nfft//2 == Nfft/2: #si pair
        I=int(Nfft/2 + 1)
    else: #si impair
        I=int((Nfft-1)/2 + 1)
    
    f=np.fft.rfftfreq(Nsig,1/fs)
    f = np.linspace(f[0],f[-1],L)
    xmat=np.zeros((I,L),dtype=complex)
    for j in range(L) :
        vec_sig = sig[j*Nhop : j*Nhop + Nwin]
        vec_sig_ham = vec_sig*np.hamming(Nwin)
        spectre_sig = np.fft.rfft(vec_sig_ham,Nfft)
        xmat[:I,j] = spectre_sig[0:I]
    time=np.linspace(0,len(sig)*1/fs,L)
    return xmat,fs,time,f #xmat est complexe

def compute_tfct_from_data(fs,sig,Nhop,Nwin,Nfft) :
    """
    # Inputs
    Takes a path to an audio WAV file, Nhop, Nwin, Nfft to compute STFT.

    # Ouputs
    Returns the 2D numpy matrix of the STFT, fs the samplerate frequency, the time series associated with the STFT, and frequencies

    return (xmat, fs, time, frequencies)
    """
    
    Nsig=len(sig)
    L=int((Nsig-Nwin+Nhop)/Nhop)
    if Nfft//2 == Nfft/2: #si pair
        I=int(Nfft/2 + 1)
    else: #si impair
        I=int((Nfft-1)/2 + 1)
    
    f=np.fft.rfftfreq(Nsig,1/fs)
    f = np.linspace(f[0],f[-1],L)
    xmat=np.zeros((I,L),dtype=complex)
    for j in range(L) :
        vec_sig = sig[j*Nhop : j*Nhop + Nwin]
        vec_sig_ham = vec_sig*np.hamming(Nwin)
        spectre_sig = np.fft.rfft(vec_sig_ham,Nfft)
        xmat[:I,j] = spectre_sig[0:I]
    time=np.linspace(0,len(sig)*1/fs,L)
    return xmat,fs,time,f #xmat est complexe

def plot_tfct(file,Nhop,Nwin,Nfft) :
    fs,sig = sp.wavfile.read(file)
    sig=sig/(2**(16-1))
    Nsig=len(sig)
    L=int((Nsig-Nwin+Nhop)/Nhop)
    if Nfft//2 == Nfft/2: #si pair
        I=int(Nfft/2 + 1)
    else: #si impair
        I=int((Nfft-1)/2 + 1)
    
    f=np.fft.rfftfreq(Nsig,1/fs)
    f = np.linspace(f[0],f[-1],L)
    xmat=np.zeros((I,L),dtype=complex)
    for j in range(L) :
        vec_sig = sig[j*Nhop : j*Nhop + Nwin]
        vec_sig_ham = vec_sig*np.hamming(Nwin)
        spectre_sig = np.fft.rfft(vec_sig_ham,Nfft)
        xmat[:I,j] = spectre_sig[0:I]
    time=np.linspace(0,len(sig)*1/fs,L)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    img = ax1.imshow(20*np.log10(np.abs(xmat)),
        extent=[time[0], time[-1] , f[0], f[-1]] ,
        #cmap=cmap ,
        interpolation = "bilinear",
        aspect="auto" ,
        origin="lower")

    fig.colorbar(img,ax=ax1)
    # ax1.legend()
    ax1.set_xlabel("")
    ax1.set_ylabel(r"Temps [s]")
    ax1.set_title(r"Fréquence [Hz]")

    fig.tight_layout()

def itfct(xmat,rate, Nwin, Nhop) :
    """
    Compute the temporal audio waveform from a TFCT/STFT matrix.

    #Inputs
    **xmat** : STFT matrix of format (freq indexes, time indexes) = (Nfft/2+1, Ntrame)
    **rate** : samplerate for the audio reconstruction
    **Nwin** : size of the synthesis window : should be the same as the analysis window with which the STFT was computed
    **Nhop** : size of the hopping : should be the same as the one used to compute the STFT

    #Returns
    **yvect** : audio waveform output of complex samples
    **time** : time vector associated with the audio waveform

    """


    nl,L = xmat.shape
    Nfft = 2*(nl-1) # car on dis que nfft sera pair
    # Nwin = Nfft
    # Nhop = int(Nwin/4)
    yvect = np.zeros((L-1)*Nhop+Nwin,dtype=complex)
    xmat2 = np.zeros((int(2*nl),L),dtype=complex) #xmat2 va contenir les trames complétées
    for i in range(int(Nfft/2)-2,1,-1) :
        for j in range(L) :
            xmat2[Nfft-i,j] = np.conjugate(xmat[i,j])

    #On calcul les tfd inverses et on stock dans la matrice ixmat
    ixmat = []
    for i in range(L) :
        ixl = np.fft.ifft(xmat2[:,i],Nfft)*np.hamming(Nfft)
        ixmat.append(ixl)
    ixmat = np.array(ixmat) #matrice dont la ligne i correspond à la TFD inverse (donc le signal temporel), associé à la ie colonne de xmat
    for i in range(L) :
        yvect[i*Nhop:i*Nhop+Nwin] = yvect[i*Nhop:i*Nhop+Nwin]+ixmat[i]
    
    #Normalisation
    K=0
    for i in range(Nwin) :
        K += np.hamming(Nfft)[i]/Nhop
    time = np.linspace(0,len(yvect)*1/rate,len(yvect))

    return yvect/K,time #yvect est un complexe

def fft(data,fe) :
    """
    # Inputs
    - data : time samples data series spaced at fe Hz

    # Ouputs
    Returns (frequencies, fft_array) as a tuple of numpy arrays.
    """
    Y = np.fft.rfft(data)
    fy = np.fft.rfftfreq(len(data),1/fe)
    return fy, Y

from scipy.linalg import hankel
def ESPRIT(x, n ,K, return_pos = True) :
    """
    ## Inputs
    - `x` : signal portion use to compute the poles
    - `n` : should be len(x)//2 by default
    - `K` : number of poles to compute with this method

    ## Outputs
    - `fk` : list of K normalized frequencies of the poles, sorted by increasing frequency
    - `deltak` : list of K damping ratios of the poles, with the same sort as frequencies
    - if `return_pos` is set to False, it returns also negative poles originally found by the method
    """
    X = hankel(x[:n],x[n-1:])
    l = X.shape[1]

    Rxx = 1/l * X @ X.conj().T

    U1, Lambda, U2 = np.linalg.svd(Rxx)

    W1 = U1[:,0:K] #On a singular matrix si on prend la partie imaginaire
    W1_down = W1[:W1.shape[0]-1,:]
    W1_up = W1[1:,:]

    PHI1 = (np.linalg.inv((W1_down.conj().T @ W1_down)) @ W1_down.conj().T) @ W1_up

    zk, _ = np.linalg.eig(PHI1)
    deltak = np.log(np.abs(zk))
    fk = 1/np.pi/2 * np.angle(zk)
    
    p = np.argsort(fk)
    fk = fk[p]
    deltak = deltak[p]

    if return_pos :
        deltak = deltak[fk>=0]
        fk = fk[fk>=0]
        return fk, deltak
    else :
        return fk, deltak

def LeastSquares(x, delta, f) :
    """
    ## Inputs
    - `x` : signal
    - `delta` : amortissements
    - `f` : fréquences (normalisées)

    ## Outputs
    - `ak` : amplitudes de chaque pic
    - `phik` : phases de chaque pic
    """
    N = len(x)
    K = len(delta)
    zk = np.exp(delta + 1j*2*np.pi*f)
    VNbis = np.zeros((N,K), dtype=np.complex128)
    for i in range(N) :
        for j in range(K) :
            VNbis[i,j] = zk[j]**i

    alphak = np.linalg.pinv(VNbis) @ x[:,np.newaxis]
    ak = np.abs(alphak)
    phik = np.angle(alphak)

    return ak, phik

def remove_init_noise(sig, fs, method="mean", ratio_parameter = 3, len_noise = 1000) :
    """
    Renvoie le signal tronqué avec le vecteurs temps associé

    # Inputs
    - sig : signal audio à tronquer
    - fs : fréquence d'échantillonnage (utile pour regénérer le vecteur temps)
    - ratio_parameter : paramètre à adapter pour couper plus ou moins le bruit. 6 par défaut

    # Ouputs
    - time : nouveau vecteur temps
    - sig : nouveau signal audio tronqué
    """
    sig /= np.max(np.abs(sig))
    noise = sig[:len_noise]
    if method == "mean" :
        noise_threshold = np.mean(np.abs(noise))
    elif method == "max" :
        noise_threshold = np.max(np.abs(noise))

    i=0
    while np.abs(sig[i]) < ratio_parameter*noise_threshold :
        if i < len(sig)-1 :
            i += 1
        elif i == len(sig) :
            print("Pas de coupe trouvée...")
            i = 0

    #i correspond au premier indice au dessus du seuil
    sig = sig[i:]
    time = np.arange(len(sig))/fs
    return time, sig

def clean_RI(ri, fs, tol_from_max=0.2, cut_end=None, method="mean", ratio_parameter = 2) :
    """
    Ce code permet premièrement de couper le début d'une réponse impulsionnelle (jusqu'à l'impact).
    Il peut aussi couper la fin en définissant une longueur souhaitée.

    ## Inputs
    - `ri` : arrayLike, signal brut de la réponse impulsionnelle.
    - `fs` : int, fréquence d'échantillonnage
    - `tol_from_max` : float, temps de tolérance autour du maximum pour définir le bruit que l'on coupe.
    - `cut_end` : None par défaut => pas de coupe à la fin. Sinon, mettre la durée en seconde souhaitée pour la taille de la RI.
    - `method` : méthode utilisée dans remove_init_noise. "mean" par défaut, "max" est une option.
    - `ratio_parameter` : paramètre permettant d'ajuster la coupe du bruit initial (2 par défaut).

    ## Outputs
    - `tRI` : arrayLike, nouveau vecteur temps.
    - `RI` : réponse impulsionnelle tronquée.
    """

    len_noise_RI = np.argmax(ri) - int(tol_from_max*fs)
    tRI, RI = remove_init_noise(ri, fs, len_noise=len_noise_RI, method=method, ratio_parameter=ratio_parameter)

    if cut_end != None :
        idx_to_cut = int(fs*cut_end)
        if np.max(np.abs(RI[:idx_to_cut])) < np.max(np.abs(ri)) :
            print("La première étape n'a pas fonctionnée, veuillez adapter les paramètres")
            return tRI,RI
        else :
            RI = RI[:idx_to_cut]
            tRI = tRI[:idx_to_cut]
    return tRI, RI


from scipy.signal import butter, sosfilt
def bandpass_filter(sig, lowcut, highcut, fs, order=5) :
    """
    Ce code filtre un signal temporel d'entrée par un passe bande.

    ## Inputs
    - `sig` : arrayLike, signal a filtrer.
    - `lowcut` : fréquence de coupure basse du filtre (non normalisés).
    - `highcut` : fréquence de coupure haute du filtre (non normalisés).
    - `fs` : fréquence d'échantillonnage du signal.
    - `order`: optionnel : ordre du filtrage généré.

    ## Outputs
    - `y` : signal temporel filtré
    """

    sos = butter(5, np.array([lowcut,highcut]),btype="bandpass", output="sos", fs=fs)
    y = sosfilt(sos, sig)
    return y

def normalise_array(sig) :
    """
    Renvoies le tableau normalisé à 1
    """
    return sig/np.max(np.abs(sig))