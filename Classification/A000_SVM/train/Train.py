from torch import nn
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa 
import librosa.display

class train(nn.Module):
    def __init__(self, 
                 model, 
                 train_loader, 
                 valid_loader,
                 num_epochs,
                 lr,
                 loss,
                 writer, 
                 save_ckpt,
                 add_fig,
                 model_name, 
                 path_main,
                 device,
                 ):
        super(train, self).__init__()

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.lr = lr
        self.path_main = path_main
        self.trained_model_path = "{}/runs/{}/{}.pt".format(self.path_main, model_name,model_name)
        self.best_trained_model_path = "{}/runs/{}/{}_best.pt".format(self.path_main, model_name,model_name)
        self.writer = writer
        self.num_epochs = num_epochs
        self.add_fig = add_fig
        self.save_ckpt = save_ckpt
        self.device = device
        self.loss = loss


    def load_checkpoint(self):
        optimizer = self.configure_optimizer()
        if os.path.isfile(self.trained_model_path):
            ckpt = torch.load(self.trained_model_path)

            self.model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"]
            print("model parameters loaded from " + self.trained_model_path)
        else:
            start_epoch = 0
            print("new model")
            
        return optimizer, start_epoch


    def configure_optimizer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr
        )
 
        return optimizer

    def compute_loss(self,
                     x,
                     theta): 
        theta_pred = self.model(x)
        loss = nn.MSELoss(reduction="none")
        full_loss = loss(theta_pred,theta).sum(dim=1).mean()
        return full_loss


    def train_step(self):
        optimizer, start_epoch = self.load_checkpoint()
        print("Optimizer, ok")

        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            
            ################## Training loop ####################
            loss = torch.Tensor([0]).to(self.device)

            for n, batch in enumerate(self.train_loader):
                theta = batch[1].to(self.device)
                inputs = batch[0][:,None,:].to(self.device)
                # Compute the loss.
                loss_add = self.compute_loss(inputs,theta)

                # Before the backward pass, zero all of the network gradients
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to parameters
                loss_add.backward()

                # Calling the step function to update the parameters
                optimizer.step()

                # Somme des loss sur tous les batches
                loss += loss_add

            # Normalisation par le nombre de batch
            loss = loss/len(self.train_loader)

            # Add loss in tensorboard 
            print("Epoch : {}, Loss : {}".format(epoch+1, loss))
            self.writer.add_scalar("Loss/Loss", loss, epoch)
            self.writer.flush()

            # Save checkpoint
            if epoch%self.save_ckpt == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "VAE_model" : self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, self.trained_model_path)

""" 
            ##################### Valid ################################
            counter = torch.Tensor([0]).to(self.device)
            valid_loss = torch.Tensor([0]).to(self.device)
            
            for n, x in enumerate(self.valid_loader):
                x = x.to(self.device)
                with torch.no_grad():
                    _,_, valid_loss_add = self.compute_loss(x)
                valid_loss += valid_loss_add
            valid_loss = valid_loss/len(self.valid_loader)
            self.writer.add_scalar("Loss/Valid_Loss", valid_loss, epoch)

            # Stopping criterion
            if epoch==start_epoch:
                old_valid = valid_loss
                min_valid = valid_loss
            if valid_loss < min_valid:
                min_valid = valid_loss
                counter = 0
                # Save best checkpoint
                checkpoint = {
                    "epoch": epoch + 1,
                    "VAE_model" : self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, self.best_trained_model_path)

            if old_valid < valid_loss:
                counter += 1

            # if counter >= 10 :
            #     print("Overfitting, train stopped")
            #     break

            old_valid = valid_loss


            ##################### Visu #############################
            if epoch%self.add_figure_sound == 0:
                nb_images = 3
                batch_test = next(iter(self.valid_loader))
                
                batch_test = batch_test.to(self.device)
                predictions,_ = self.model(batch_test)
                samples_rec = predictions[0:nb_images,:].cpu().detach().numpy()
                samples = batch_test[0:nb_images,:].cpu().detach().numpy()
                Fe = len(samples_rec[0][0])//2

                figure, ax = plt.subplots(nrows=nb_images, sharex=True)
                for i in range(nb_images):
                    librosa.display.waveshow(samples[i][0,:], sr=Fe, alpha=0.9, ax=ax[i], label='Original')
                    librosa.display.waveshow(samples_rec[i][0,:], sr=Fe, color='r', alpha=0.5, ax=ax[i], label='Reconstructed')
                    ax[i].set(title=f'Sound {i+1}')
                    #plt.grid()
                    if i==0:
                        ax[i].legend()

                plt.tight_layout()

                self.writer.add_figure("Waveform", figure, epoch)

                self.writer.flush()

                 """

