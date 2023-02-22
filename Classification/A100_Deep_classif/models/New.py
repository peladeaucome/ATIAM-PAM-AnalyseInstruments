import torch.nn as nn



def nb_echantillons(time,total_echantillons):
    delta_t = 3/total_echantillons
    echantillons = time * 3 /delta_t
    return int(echantillons)


class New_model(nn.Module):
    
    def __init__(self,
                 nb_bins_oct = 12*5,
                 nb_bins = 80*5
                 ):
        super(New_model, self).__init__()

        self.nb_bins_oct = nb_bins_oct
        self.nb_bins = nb_bins
        ################### Build CNN layers ###################
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels = 1, 
                      out_channels = 16, 
                      kernel_size = (self.nb_bins_oct,nb_echantillons(0.05,385)),
                      stride = 4,
                      padding = (self.nb_bins_oct//2-1,nb_echantillons(0.05,385)//2-1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels = 16, 
                      out_channels = 32, 
                      kernel_size = (self.nb_bins_oct//2,nb_echantillons(0.05,192)),
                      stride = 2,
                      padding = (self.nb_bins_oct//4-1,nb_echantillons(0.05,192)//2-1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels = 32, 
                      out_channels = 64, 
                      kernel_size = (self.nb_bins_oct//4,nb_echantillons(0.05,95)),
                      stride = 2,
                      padding = (self.nb_bins_oct//8-1,nb_echantillons(0.05,95)//2-1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels = 64, 
                      out_channels = 128, 
                      kernel_size = (self.nb_bins_oct//8,nb_echantillons(0.05,47)),
                      stride = 2,
                      padding = (self.nb_bins_oct//16-1,nb_echantillons(0.05,47)//2-1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

        )

        self.MLP = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )

        self.last_layer = nn.Sequential(
            nn.Linear(32, 4),
            nn.Softmax(dim=1)
        )



    def forward(self, x):
        CNN_output = self.CNN(x)
        #print(CNN_output.shape)
        mean = CNN_output.mean(dim=3).mean(dim=2)
        #print(mean.shape)
        Linear_output = self.MLP(mean)
        #print(Linear_output.shape)
        output = self.last_layer(Linear_output)

        return output, Linear_output,CNN_output
    
    
