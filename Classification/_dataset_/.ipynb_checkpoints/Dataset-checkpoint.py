import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import librosa
import _dataset_.Compute_Features as cf

def load_data(path,dataset_type = "list",
              fs=40000,
              resample = False,
              resample_rate=32768,
              features = False,
              use="Deep"):
    
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
    for n,label in enumerate(os.listdir(path)):
        #load du pickel avec les parametres
        with open(path+label+"/parametres.pickle", "rb") as handle:
            parameters_dict = pickle.load(handle)

        # creation d'un dictionnaire avec les labels et leur numero
        label_num[label] = n

        # On parcours les wav du dossier et on les met dans un dictionnaire
        for j,wav in enumerate(os.listdir(path+label+"/Wav")):
            # On load les wav
            data,_ = librosa.load(path+label+"/Wav/"+wav,sr=fs)
            if resample:
                data = librosa.resample(data,orig_sr = fs,target_sr=resample_rate)


            if dataset_type=="list":
                parameters_add = np.zeros(len(parameters_dict[wav]))
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
            if dataset_type=="dict":# On les met dans un dictionnaire
                name=label+"_"+str(j)
                dataset[name] ={"data":data,"parameters":parameters_dict[wav],"label":label}
    if use=="SVM":
        data_list = np.asarray(data_list)
        label_list = np.asarray(label_list)
        parameters_list = np.asarray(parameters_list)
        dataset=(data_list,label_list,parameters_list,name_list)
    return dataset,label_num

def Create_Dataset(dataset, 
                   valid_ratio = 0.1,
                   num_threads = 0,
                   batch_size  = 2):
    # Load the dataset for the training/validation sets
    train_valid_dataset =  dataset
    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset) )
    nb_valid =  int(valid_ratio * len(train_valid_dataset))

    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])

    # Prepare 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_threads)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads)


    print("The train set contains {} samples of length 32768, in {} batches".format(len(train_loader.dataset), len(train_loader)))
    print("The validation set contains {} samples of length 32768, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))

    return train_loader,valid_loader
