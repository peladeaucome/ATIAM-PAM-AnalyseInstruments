import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import librosa
import torchaudio.transforms as ta

def load_data_SVM(path,
              fs=40000,
              resample = False,
              resample_rate=32768):
    
    
    dataset= []
    label_num = {}

    # Dans le cas d'une utilisation SVM on veut deux listes
    data_list = []
    label_list = []
    parameters_list = []
    name_list = []
    
    for n,label in enumerate(os.listdir(path)):
        #load du pickel avec les parametres
        with open(path+label+"/parametres.pkl", "rb") as handle:
            parameters_dict = pickle.load(handle)

        # creation d'un dictionnaire avec les labels et leur numero
        label_num[label] = n

        # On parcours les wav du dossier et on les met dans un dictionnaire
        for j,wav in enumerate(os.listdir(path+label+"/Wav")):
            # On load les wav
            key_name = str(j)+".wav"
            parameters_add = np.zeros(115)
            
            data,_ = librosa.load(path+label+"/Wav/"+wav,sr=fs)
            if resample:
                data = librosa.resample(data,orig_sr = fs,target_sr=resample_rate)

            for i,key in enumerate(parameters_dict[key_name]):
                parameters_add[i] = parameters_dict[key_name][key]
            data_list.append(data)
            label_list.append(n)
            parameters_list.append(parameters_add)
            name_list.append(label+"_"+str(j))
            
    
    
    data_list = np.asarray(data_list)
    label_list = np.asarray(label_list)
    parameters_list = np.asarray(parameters_list)
    dataset=(data_list,label_list,parameters_list,name_list) 
    return dataset,label_num





def load_data_deep(path,
                   fs=40000,
                   resample = False,
                   resample_rate=32768,
                   device = "cpu",
                   spec_type = "MEL",):
    

    dataset= []
    label_num = {}
    label_Deep = np.zeros(len(os.listdir(path)))
    
    for n,label in enumerate(os.listdir(path)):
        #load du pickel avec les parametres
        with open(path+label+"/parametres.pkl", "rb") as handle:
            parameters_dict = pickle.load(handle)

        # creation d'un dictionnaire avec les labels et leur numero
        label_num[label] = n

        # On parcours les wav du dossier et on les met dans un dictionnaire
        for j,wav in enumerate(os.listdir(path+label+"/Wav")):
            # On load les wav
            key_name = str(j)+".wav"
            label_Deep = np.zeros(len(os.listdir(path)))
            data,_ = librosa.load(path+label+"/Wav/"+wav,sr=fs)
            if resample:
                data = librosa.resample(data,orig_sr = fs,target_sr=resample_rate)
            
            # Traitement des donnees
            if spec_type == "MEL":
                data = MEL(data,resample_rate)
            else :
                data = CQT(data,resample_rate)
            data = data[None,:,:]
            
            data = torch.from_numpy(data).float().to(device)
            
            # Save des parametres dans le model
            parameters_add = np.zeros(len(parameters_dict[key_name]))
            for i,key in enumerate(parameters_dict[key_name]):
                if key=="acc_t":
                    continue
                else :
                    parameters_add[i] = parameters_dict[key_name][key]
                    
     
            # Save des labels
            label_Deep[n] = 1
            deep_label = torch.from_numpy(label_Deep).float().to(device)

            # Append au dataset
            dataset.append((data,deep_label,parameters_add,n))
           
    return dataset,label_num



def load_mes(path,
              resample = False,
              resample_rate=16384,
              device = "cpu",
              spec_type = "MEL"):
    
    dataset= []
    label_num = {}
    if ".DS_Store" in os.listdir(path):
        os.remove(path+"/.DS_Store")
    label_Deep = np.zeros(len(os.listdir(path)))
    parameters = np.zeros(10)
    for n,label in enumerate(os.listdir(path)):
        if label == ".DS_Store":
            continue
        # creation d'un dictionnaire avec les labels et leur numero
        label_num["{}".format(n)] = label
        # On parcours les wav du dossier et on les met dans un dictionnaire
        for j,wav in enumerate(os.listdir(path+"/"+label+"/Wav")):
            # On load les wav
            label_Deep = np.zeros(len(os.listdir(path)))
            data,fs = librosa.load(path+"/"+label+"/Wav/"+wav,sr = 51200)
            if resample:
                data = librosa.resample(data,orig_sr = 51200,target_sr=resample_rate)
            data = data[:3*resample_rate]

            if spec_type == "MEL":
                data = MEL(data,resample_rate)
            else :
                data = CQT(data,resample_rate)
            
            data = torch.from_numpy(data).float().to(device)
            data = data[None,:,:]
            label_Deep[n] = 1
            deep_label = torch.from_numpy(label_Deep).float().to(device)
            dataset.append((data,deep_label,parameters,n))
    return dataset,label_num




def CQT(data,fs):
    if fs == 16384:
        cqt = np.abs(librosa.cqt(data, 
                                 sr=fs,
                                 hop_length = 128 , 
                                 n_bins= 80 * 5, 
                                 bins_per_octave=12* 5, 
                                 fmin=librosa.note_to_hz('C2'), 
                                 sparsity=0.01, 
                                 window='hann',
                                 pad_mode='constant'))
        return cqt
    else:
        cqt = np.abs(librosa.cqt(data, sr=fs))
        print("\n     !!!!! You will have shape problems dumbass !!!!!\n")
        return cqt
    




def Create_Dataset(dataset, 
                   valid_ratio = 0.1,
                   num_threads = 0,
                   batch_size  = 2,
                   mesure = False):
    # Load the dataset for the training/validation sets
    train_valid_dataset =  dataset
    if mesure:
        # Split it into training and validation sets
        nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset) )
        nb_valid =  int(valid_ratio * len(train_valid_dataset)+1)
    else :
        # Split it into training and validation sets
        nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset) )
        nb_valid =  int(valid_ratio * len(train_valid_dataset))

    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid],generator = torch.Generator().manual_seed(42))

    # Prepare 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_threads)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads)


    print("The train set contains {} samples, in {} batches".format(len(train_loader.dataset), len(train_loader)))
    print("The validation set contains {} samples, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))

    return train_loader,valid_loader


#############################################################
#############################################################
#############################################################
#############################################################
# Function for computing a mel spectrogram with the good shape for our problem
def MEL(data,fs):
    if fs == 16384:
        mel = librosa.feature.melspectrogram(y=data,sr = fs,hop_length=386)
        return mel
    else : 
        mel = librosa.feature.melspectrogram(y=data,sr = fs)
        print("\n     !!!!! You will have shape problems dumbass !!!!!\n")
        return mel

""" def CQT(data,fs):
    if fs == 16384:
        cqt = np.abs(librosa.cqt(data, sr=fs, n_bins=64, bins_per_octave=12, hop_length=386, fmin=librosa.note_to_hz('C2'), sparsity=0.01, window='hann'))
        return cqt
    else:
        cqt = np.abs(librosa.cqt(data, sr=fs))
        print("\n     !!!!! You will have shape problems dumbass !!!!!\n")
        return cqt """


# Load data pour le nouveau dataset
def load_data_cordes(path,
                     fs=32768,
                     resample = False,
                     resample_rate=16384,
                     device = "cpu"):
    dataset= []
    label_num = {}
    label_Deep = np.zeros(len(os.listdir(path)))
    random = np.zeros(20)
    for n,label in enumerate(os.listdir(path)):
        # creation d'un dictionnaire avec les labels et leur numero
        label_num[label] = n

        # On parcours les wav du dossier et on les met dans un dictionnaire
        for j,wav in enumerate(os.listdir(path+label+"/Wav")):
            # On load les wav
            label_Deep = np.zeros(len(os.listdir(path)))
            data,_ = librosa.load(path+label+"/Wav/"+wav,sr=fs)
            if resample:
                data = librosa.resample(data,orig_sr = fs,target_sr = resample_rate)
            data = MEL(data,resample_rate)
            data = data[None,:,:]
            data = torch.from_numpy(data).float().to(device)
            label_Deep[n] = 1
            deep_label = torch.from_numpy(label_Deep).float().to(device)
            dataset.append((data,deep_label,random,n))
    return dataset,label_num

class AudioTransform(torch.nn.Module):
    def __init__(self, n_fft=1023, device = 'cpu'):
        super().__init__()
        self.n_fft = n_fft
        window_length = 400
        #self.window = torch.hann_window(window_length).to(device)
        self.transf = ta.Spectrogram(n_fft = n_fft, win_length=window_length, hop_length=window_length//4, pad=window_length*2+200, power=2, normalized=False).to(device)
        
    def forward(self, wav: torch.Tensor): #-> torch.Tensor:
        # Convert to power spectrogram
        spec = self.transf(wav)
        return spec

def load_data(path,
              dataset_type = "list",
              fs=40000,
              resample = False,
              resample_rate=32768,
              features = False,
              use="Deep",
              device = "cpu"):
    
    if dataset_type=="list":
        dataset= []
    if dataset_type=="dict":
        dataset= {}
    label_num = {}

    # Dans le cas d'une utilisation SVM on veut deux listes
    data_list = []
    label_list = []
    parameters_list = []
    name_list = []
    label_Deep = np.zeros(len(os.listdir(path)))
    
    for n,label in enumerate(os.listdir(path)):
        #load du pickel avec les parametres
        with open(path+label+"/parametres.pkl", "rb") as handle:
            parameters_dict = pickle.load(handle)

        # creation d'un dictionnaire avec les labels et leur numero
        label_num[label] = n

        # On parcours les wav du dossier et on les met dans un dictionnaire
        for j,wav in enumerate(os.listdir(path+label+"/Wav")):
            # On load les wav
            key_name = str(j)+".wav"
            label_Deep = np.zeros(len(os.listdir(path)))
            data,_ = librosa.load(path+label+"/Wav/"+wav,sr=fs)
            if resample:
                data = librosa.resample(data,orig_sr = fs,target_sr=resample_rate)


            if dataset_type=="list":
                parameters_add = np.zeros(len(parameters_dict[key_name]))
                if use=="Deep":
                    for i,key in enumerate(parameters_dict[wav]):
                        parameters_add[i] = parameters_dict[wav][key]
                    dataset.append((data,parameters_add,n))
                    
                if use=="SVM":
                    for i,key in enumerate(parameters_dict[wav]):
                        parameters_add[i] = parameters_dict[wav][key]
                    data_list.append(data)
                    label_list.append(n)
                    parameters_list.append(parameters_add)
                    name_list.append(label+"_"+str(j))

                if use=="Deep_classif":
                    for i,key in enumerate(parameters_dict[key_name]):
                        if i==0:
                            continue
                        else :
                            parameters_add[i] = parameters_dict[key_name][key]
                    
                    
                    data = MEL(data,resample_rate)
                    data = torch.from_numpy(data).float().to(device)
                    data = data[None,:,:]
                    label_Deep[n] = 1
                    deep_label = torch.from_numpy(label_Deep).float().to(device)
                    dataset.append((data,deep_label,parameters_add,n))
            if dataset_type=="dict":# On les met dans un dictionnaire
                name=label+"_"+str(j)
                dataset[name] ={"data":data,"parameters":parameters_dict[wav],"label":label}
    if use=="SVM":
        data_list = np.asarray(data_list)
        label_list = np.asarray(label_list)
        parameters_list = np.asarray(parameters_list)
        dataset=(data_list,label_list,parameters_list,name_list) 
    return dataset,label_num

