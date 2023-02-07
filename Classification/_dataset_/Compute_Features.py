import numpy as np
import librosa
import Classification._dataset_.Normalize as n



def spectral_slope(data, sr, n_fft=2048, hop_length=None, win_length=None, 
                   window='hann', center=True, pad_mode='constant'):

    S, phase = librosa.magphase(librosa.stft(y=data, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, 
                                             center=center, pad_mode=pad_mode))
    slope = np.zeros((1,S.shape[1]))
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    for k in range(S.shape[1]):
        fit = np.polyfit(freq, S[:,k], 1)
        slope[0,k] =fit[0]
    return slope


def compute_features(data, sr, S=None, n_fft=2048, frame_length=2048, hop_length=512, win_length=None, window='hann',
                     center=True, pad_mode='constant', freq=None, fmin=200.0, n_bands=6, quantile=0.02, linear=False, centroid=None, norm=True,
                     p=2, amin=1e-10, power=2.0,roll_percent=0.85,use="SVM",start=0,stop=-1,normalisation=True):
    
    features = []
    # Without contrast and spectral slope in a first place
    name = ['spectral_centroid', 'spectral_bandwidth', 'spectral_flatness', 'spectral_rolloff','zero_crossing_rate', 'rms']
    cent = librosa.feature.spectral_centroid(y=data, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, freq = freq,
                                             win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    spec_bw = librosa.feature.spectral_bandwidth(y=data, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, 
                                                 center=center, pad_mode=pad_mode, freq=freq, centroid=centroid, norm=norm, p=p)
    contrast = librosa.feature.spectral_contrast(y=data, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center,
                                                 pad_mode=pad_mode, freq=freq, fmin=fmin, n_bands=n_bands, quantile=quantile, linear=linear)
    
    flatness = librosa.feature.spectral_flatness(y=data, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center,
                                                 pad_mode=pad_mode, amin=amin, power=power)
    
    rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, 
                                                center=center, pad_mode=pad_mode, freq=freq, roll_percent=roll_percent)
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length, center=center)
    
    rms = librosa.feature.rms(y=data, S=S, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
    
    """ slope = spectral_slope(data, sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                   window=window, center=center, pad_mode=pad_mode) """
    
    if use == "SVM":
        if normalisation==True:
            #print(np.shape(cent[:,0,start:stop].mean(axis=-1).reshape(1,-1)))
            dict_features = {}
            dict_features['spectral_centroid'] = n.normalize(cent[:,0,start:stop].mean(axis=-1).reshape(1,-1))[0]
            dict_features['spectral_bandwidth'] = n.normalize(spec_bw[:,0,start:stop].mean(axis=-1).reshape(1,-1))[0]
            #dict_features['spectral_contrast'] = contrast.mean(axis=-1)
            dict_features['spectral_flatness'] = n.normalize(flatness[:,0,start:stop].mean(axis=-1).reshape(1,-1))[0]
            dict_features['spectral_rolloff'] = n.normalize(rolloff[:,0,start:stop].mean(axis=-1).reshape(1,-1))[0]
            dict_features['zero_crossing_rate'] = n.normalize(zcr[:,0,start:stop].mean(axis=-1).reshape(1,-1))[0]
            dict_features['rms'] = n.normalize(rms[:,0,start:stop].mean(axis=-1).reshape(1,-1))[0]
            #dict_features['spectral_slope'] = slope
            
            return dict_features
        else :
            dict_features = {}
            dict_features['spectral_centroid'] = cent[:,0,start:stop].mean(axis=-1)
            dict_features['spectral_bandwidth'] = spec_bw[:,0,start:stop].mean(axis=-1)
            #dict_features['spectral_contrast'] = contrast.mean(axis=-1)
            dict_features['spectral_flatness'] = flatness[:,0,start:stop].mean(axis=-1)
            dict_features['spectral_rolloff'] = rolloff[:,0,start:stop].mean(axis=-1)
            dict_features['zero_crossing_rate'] = zcr[:,0,start:stop].mean(axis=-1)
            dict_features['rms'] = rms[:,0,start:stop].mean(axis=-1)
            #dict_features['spectral_slope'] = slope

            return dict_features
    else:
        dict_features = {}
        dict_features['spectral_centroid'] = cent[:,0,start:stop]
        dict_features['spectral_bandwidth'] = spec_bw[:,0,start:stop]
        #dict_features['spectral_contrast'] = contrast.mean(axis=-1)
        dict_features['spectral_flatness'] = flatness[:,0,start:stop]
        dict_features['spectral_rolloff'] = rolloff[:,0,start:stop]
        dict_features['zero_crossing_rate'] = zcr[:,0,start:stop]
        dict_features['rms'] = rms[:,0,start:stop]
        #dict_features['spectral_slope'] = slope
        
        return dict_features