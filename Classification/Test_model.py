import _dataset_.Dataset as dataset
import _dataset_.Compute_Features as cf
import numpy as np
from torch.utils.data import Dataset
import pickle
import os
import librosa
import _dataset_.Compute_Features as cf
import torchaudio.transforms as ta
import A100_Deep_classif.configs.config as config
import matplotlib.pyplot as plt
import A100_Deep_classif.models.CNN_MLP 
import torch
import torch.nn as nn
from scipy.io import loadmat
import torchmetrics as tm
import A100_Deep_classif.train.Train

##################### Charge Model #####################
def charge_model(model_name,algo, path_main, best = True):
    main_config = config.load_config("{}/{}/runs/{}/{}_train_config.yaml".format(path_main,algo, model_name, model_name))

    sample_batch = [torch.rand(20,1,512,512),torch.rand(20,main_config.model.nb_classes),torch.rand(20,11),torch.rand(20)]

    model = A100_Deep_classif.models.CNN_MLP.CNN_MLP(data = sample_batch,
                                                     nb_classes = main_config.model.nb_classes,
                                                     ratios_CNN = main_config.model.ratios_CNN,
                                                     channel_size = main_config.model.channel_size,
                                                     size_MLP = main_config.model.size_MLP
                                                     )
    if best:
        ckpt = torch.load("{}/{}/runs/{}/{}_best.pt".format(path_main,algo, model_name, model_name))
    else:
        ckpt = torch.load("{}/{}/runs/{}/{}.pt".format(path_main,algo, model_name, model_name))
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model,main_config


best = True
model_name = "Dataset_1"
path_main = "./Classification"
algo = "A100_Deep_classif"
classes = ["acier", "medium_1", "medium_2", "plexi"]

model,main_config = charge_model(model_name,algo, path_main, best = best)


##################### Charge Data #####################
name_dataset = "Dataset_Mesures_HR"
path_dataset = "{}/_dataset_/Dataset/{}".format(path_main, name_dataset)
list_dataset,label_num = dataset.load_mes(path_dataset,resample=True,resample_rate=16384)

train_loader,valid_loader = dataset.Create_Dataset(dataset= list_dataset,
                                                   valid_ratio = main_config.dataset.valid_ratio,
                                                   num_threads = main_config.dataset.num_thread,
                                                   batch_size = main_config.dataset.batch_size)


##################### Compute Accuracy #####################
accuracy = torch.Tensor([0])

def compute_acc(inputs,classes):
    with torch.no_grad():
        output = model(inputs)
        pred_class = classes[torch.argmax(output, dim = 1)]
        accuracy = tm.Accuracy(task = "multiclass", num_classes = 4)
        acc = accuracy(pred_class, classes)
        return acc

for n, batch in enumerate(valid_loader):
    labels = batch[1]
    inputs = batch[0]
    classes = batch[2]
    with torch.no_grad():
        accuracy_add = compute_acc(inputs,classes)
    accuracy += accuracy_add
for n, batch in enumerate(train_loader):
    labels = batch[1]
    inputs = batch[0]
    classes = batch[2]
    with torch.no_grad():
        accuracy_add = compute_acc(inputs,classes)
    accuracy += accuracy_add
accuracy = accuracy/(len(valid_loader)+len(train_loader))
print("\nAccuracy : {}\n".format(accuracy))


