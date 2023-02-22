import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
# Import des fichiers du modele
import B000_Deep.configs.config as config
import _dataset_.Dataset as dataset
import B000_Deep.models.CNN_MLP 
import B000_Deep.train.Train
import time

############################################################# A MODIF #########################
path_main = "./Classification/B000_Deep"
path_dataset = "./Classification/_dataset_/Dataset/Dataset_1/"
############################################################# A MODIF #########################


# Definit le device sur lequel on va train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Permet de load toute la configuration
main_config = config.load_config("{}/config.yaml".format(path_main))

start = time.time()
# Import du dataset
list_dataset,label_num = dataset.load_data(path=path_dataset,
                                           dataset_type=main_config.dataset.dataset_type,
                                           fs=main_config.dataset.fs,
                                           resample=main_config.dataset.resample,
                                           resample_rate=main_config.dataset.resample_rate)


train_loader,valid_loader = dataset.Create_Dataset(dataset= list_dataset,
                                                   valid_ratio = main_config.dataset.valid_ratio,
                                                   num_threads = main_config.dataset.num_thread,
                                                   batch_size = main_config.dataset.batch_size)
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
sample_batch[0] = sample_batch[0][:,None,:]

model = B000_Deep.models.CNN_MLP.CNN_MLP(data = sample_batch,
                               output_dim = main_config.model.output_dim,
                               ratios_CNN = main_config.model.ratios_CNN,
                               channel_size = main_config.model.channel_size,
                               size_MLP = main_config.model.size_MLP
                               ).to(device)

print("\n")
print(summary(model,(sample_batch[0].size()[1],sample_batch[0].size()[2])))
print("\n")

model_train = B000_Deep.train.Train.train(model = model,
                              train_loader = train_loader,
                              valid_loader = valid_loader,
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



