import torch
import torch.nn as nn




class CNN_MLP(nn.Module):
    
    def __init__(self, 
                 data,
                 output_dim: int,
                 ratios_CNN: list = [4,4,4,4],
                 channel_size: list = [16,32,64,128],
                 size_MLP: list = [128,64,32]
                 ):
        super(CNN_MLP, self).__init__()

        self.output_dim = output_dim
        self.ratios_CNN = ratios_CNN
        self.channel_size = channel_size
        self.size_MLP = size_MLP
        self.in_channels =  data[0].size()[1] #On prend 1 car 0 c'est de batch
        self.ratio_CNN_global = 1
        for ratio in self.ratios_CNN:
            self.ratio_CNN_global *= ratio
        self.MLPin_size = int(self.channel_size[-1]*data[0].size()[2]/(self.ratio_CNN_global))
        
        ################### Build CNN layers ###################
        modules = nn.Sequential()
        for i,ratio in enumerate(self.ratios_CNN):
            modules.append(
                    nn.Conv1d(self.in_channels,
                              out_channels=self.channel_size[i],
                              kernel_size= (2*ratio)+1, 
                              stride = ratio, 
                              padding  = ratio))
            modules.append(
                    nn.BatchNorm1d(self.channel_size[i]))
            modules.append(
                    nn.LeakyReLU())
            
            self.in_channels = self.channel_size[i]

        self.CNN = modules



        ################### Build MLP layers ###################
        modules = nn.Sequential(nn.Flatten())
        
        for i,ratio in enumerate(self.size_MLP):
            modules.append(
                    nn.Linear(in_features = self.MLPin_size,
                              out_features = self.size_MLP[i]))
            modules.append(
                    nn.LeakyReLU())
            
            self.MLPin_size = self.size_MLP[i]
        modules.append(
                nn.Linear(in_features = self.MLPin_size,
                              out_features = self.output_dim))
        modules.append(
                nn.LeakyReLU())
        self.MLP = modules

    def forward(self, x):
        CNN_output = self.CNN(x)
        theta = self.MLP(CNN_output)
        return theta
    
    
