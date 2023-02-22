import _dataset_.Dataset as dataset
import A000_SVM.configs.config as config
import A000_SVM.SVM as SVM
import A000_SVM.train.Train
import _dataset_.Compute_Features as cf
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import matplotlib.pyplot as plt
############################################################# A MODIF #########################
path_main = "./Classification/A000_SVM"
path_dataset = "./Classification/_dataset_/Dataset/Dataset_4/"
############################################################# A MODIF #########################


# Permet de load toute la configuration
main_config = config.load_config("{}/config.yaml".format(path_main))


# Import du dataset
list_dataset,label_num = dataset.load_data_SVM(path=path_dataset,
                                           fs=32768,  
                                           resample=True,
                                           resample_rate=16384)
list_classes = list(label_num.keys())

print("Dataset loaded !")
print(list_dataset[0].shape, list_dataset[1].shape)
""" 
dict_features = cf.compute_features(data = list_dataset[0],
                           sr = main_config.dataset.fs,
                           use = "SVM",
                           normalisation=True)

print("Dataset loaded and features computed !")


# Creation session tensorboard et save de la config
writer = SummaryWriter("{}/runs/{}".format(path_main, main_config.model.model_name))


writer.add_text('SVM_Parameters', str(main_config.model))
writer.add_text('Train_parameters', str(main_config.train))
writer.add_text('Dataset_config', str(main_config.dataset))

# Save le config file pour pouvoir le rouvrir par la suite (save dans le dossier de logs runs/model_name)
config_path = "{}/runs/{}".format(path_main, main_config.model.model_name)   
config_name = "{}/{}_train_config.yaml".format(config_path, main_config.model.model_name)   
config.save_config(main_config , config_name)

SVM_train = A000_SVM.train.Train.train(features = dict_features,
                                       labels = list_dataset[1],
                                       classes_names = list_classes,
                                       valid_ratio = main_config.dataset.valid_ratio,
                                       feature_use = main_config.model.feature_used,
                                       kernel_svm = main_config.model.kernel_svm,
                                       C_svm = main_config.model.C_svm,
                                       path_main = path_main,
                                       plot_title = main_config.model.plot_title,
                                       model_name = main_config.model.model_name,
                                       writer = writer,
                                       step=main_config.model.step,
                                       n_features=main_config.model.n_features)
SVM_train.train_step() """




