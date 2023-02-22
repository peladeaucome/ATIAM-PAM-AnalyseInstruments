import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
# Import des fichiers du modele
import A100_Deep_classif.configs.config as config
import _dataset_.Dataset as dataset
import A100_Deep_classif.models.New 
import A100_Deep_classif.train.New_train
import time

############################################################# A MODIF #########################
path_main = "./Classification/A100_Deep_classif"
path_dataset = "./Classification/_dataset_/Dataset/Dataset_4/"
############################################################# A MODIF #########################


# Definit le device sur lequel on va train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Permet de load toute la configuration
main_config = config.load_config("{}/config.yaml".format(path_main))

start = time.time()
###### Load dataset ######
list_dataset,label_num = dataset.load_data_deep(path=path_dataset,
                                                fs=main_config.dataset.fs,
                                                resample=main_config.dataset.resample,
                                                resample_rate=main_config.dataset.resample_rate,
                                                device = 'cpu',
                                                spec_type = 'CQT')

train_loader,valid_loader = dataset.Create_Dataset(dataset= list_dataset,
                                                   valid_ratio = main_config.dataset.valid_ratio,
                                                   num_threads = main_config.dataset.num_thread,
                                                   batch_size = main_config.dataset.batch_size)


###### Load dataset Mesures HR ######
name_dataset = "Dataset_Mesures/"
path_dataset = "./Classification/_dataset_/Dataset/{}".format( name_dataset)
list_dataset_mes, label_num_mes = dataset.load_mes(path_dataset,
                                                   resample=True,
                                                   resample_rate=16384,
                                                   device = 'cpu',
                                                   spec_type = 'CQT')



train_loader_mes,valid_loader_mes = dataset.Create_Dataset(dataset = list_dataset_mes,
                                                           valid_ratio = 0.75,
                                                           num_threads = main_config.dataset.num_thread,
                                                           batch_size = main_config.dataset.batch_size,
                                                           mesure = True)

stop = time.time()
print("\n                       Dataset loaded in {} seconds \n".format(stop-start))

# Creation session tensorboard et save de la config
checkpoint = "{}".format(main_config.model.model_name)
writer = SummaryWriter("{}/runs/".format(path_main) + checkpoint)


writer.add_text('VAE_raw_parameters', str(main_config.model))
writer.add_text('Train_parameters', str(main_config.train))
writer.add_text('Dataset_config', str(main_config.dataset))


# Save le config file pour pouvoir le rouvrir par la suite (save dans le dossier de logs runs/model_name)
config_path = "{}/runs/{}".format(path_main, main_config.model.model_name)   
config_name = "{}/{}_train_config.yaml".format(config_path, main_config.model.model_name)   
config.save_config(main_config , config_name)

sample_batch = next(iter(train_loader))

model = A100_Deep_classif.models.New.New_model().to(device)

print("\n")
print(summary(model,(sample_batch[0].size()[1],sample_batch[0].size()[2],sample_batch[0].size()[3])))
print("\n")

model_train = A100_Deep_classif.train.New_train.train(model = model,
                              train_loader = train_loader,
                              valid_loader = valid_loader,
                              train_loader_mes = train_loader_mes,
                              valid_loader_mes = valid_loader_mes,
                              num_epochs = main_config.train.epochs,
                              lr = main_config.train.lr,
                              loss = main_config.train.loss,
                              writer = writer,
                              save_ckpt = main_config.train.save_ckpt,
                              add_fig = main_config.train.add_fig,
                              model_name = main_config.model.model_name,
                              path_main = path_main,
                              device = device)

model_train.train_step()



""" output,linear,cnn = model(sample_batch[0].to(device))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(cnn[0,0,:,:].detach().cpu().numpy())
plt.show()

plt.figure()
plt.imshow(cnn[0,1,:,:].detach().cpu().numpy())
plt.show()

plt.figure()
plt.imshow(cnn[1,0,:,:].detach().cpu().numpy())
plt.show()

plt.figure()
plt.imshow(cnn[1,1,:,:].detach().cpu().numpy())
plt.show()

plt.figure()
plt.plot(output.detach().cpu().numpy())
plt.show()

 """