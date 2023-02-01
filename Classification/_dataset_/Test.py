import Dataset as dataset
import Compute_Features as cf
import numpy as np

path_dataset = "./Apprentissage/_dataset_/Dataset_Test/"
############################################################# A MODIF #########################


# Import du dataset
list_dataset,label_num = dataset.load_data(path=path_dataset,
                                           dataset_type="list",
                                           fs=40000,
                                           resample=False,
                                           resample_rate=32768,
                                           use="SVM")

print(np.shape(list_dataset[0]))