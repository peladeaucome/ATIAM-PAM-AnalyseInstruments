from torch import nn
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa 
import librosa.display
import torchmetrics as tm
import seaborn as sns

class train(nn.Module):
    def __init__(self, 
                 model, 
                 train_loader, 
                 valid_loader,
                 train_loader_mes, 
                 valid_loader_mes,
                 num_epochs,
                 lr,
                 loss,
                 writer, 
                 save_ckpt,
                 add_fig,
                 model_name, 
                 path_main,
                 device,
                 mes_in_train = False
                 ):
        super(train, self).__init__()

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_loader_mes = train_loader_mes
        self.valid_loader_mes = valid_loader_mes
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
        self.mes_in_train = mes_in_train


    def load_checkpoint(self):
        """
        Load checkpoint if it exists, otherwise initialize the model and optimizer.
        """
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
        """
        Configure the optimizer with the loaded model.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr
        )
 
        return optimizer

    def compute_loss(self,
                     x,
                     labels): 
        """
        Compute loss for a batch of inputs.
        """
        outputs,_,_ = self.model(x)
        criterion = nn.CrossEntropyLoss()
        full_loss = criterion(outputs, labels)
        return full_loss
    
    # Compute accuracy score
    def compute_acc(self,
                     x,
                     classes): 
        """
        Compute classification accuracy.
        """
        pred_classes = self.compute_pred(x)
        accuracy = tm.Accuracy(task="multiclass", num_classes=4).to(self.device)
        acc = accuracy(pred_classes, classes)
        return acc
    
    def compute_pred(self,
                        x):
        """
        Compute classes prediction for a batch of inputs.
        """
        outputs,_,_ = self.model(x)
        pred_classes = torch.argmax(outputs, dim=1)
        return pred_classes


    def train_step(self):
        optimizer, start_epoch = self.load_checkpoint()
        print("Optimizer, ok")

        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            
            ################## Training loop ####################
            loss = torch.Tensor([0]).to(self.device)

            for n, batch in enumerate(self.train_loader):
                labels = batch[1].to(self.device)
                inputs = batch[0].to(self.device)
                # Compute the loss.
                loss_add = self.compute_loss(inputs,labels)

                # Before the backward pass, zero all of the network gradients
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to parameters
                loss_add.backward()

                # Calling the step function to update the parameters
                optimizer.step()

                # Somme des loss sur tous les batches
                loss += loss_add



            if self.mes_in_train:

                for n, batch in enumerate(self.train_loader_mes):
                    labels = batch[1].to(self.device)
                    inputs = batch[0].to(self.device)
                    # Compute the loss.
                    loss_add = self.compute_loss(inputs,labels)

                    # Before the backward pass, zero all of the network gradients
                    optimizer.zero_grad()

                    # Backward pass: compute gradient of the loss with respect to parameters
                    loss_add.backward()

                    # Calling the step function to update the parameters
                    optimizer.step()

                    # Somme des loss sur tous les batches
                    loss += loss_add

                # Normalisation par le nombre de batch
                loss = loss/(len(self.train_loader)+len(self.train_loader_mes)) #
            else:
                # Normalisation par le nombre de batch
                loss = loss/len(self.train_loader)


            # Add loss in tensorboard 
            print("\nEpoch : {}\nLoss : {}".format(epoch+1, loss))
            self.writer.add_scalar("Training/Loss", loss, epoch)
            self.writer.flush()

            # Save checkpoint
            if epoch%self.save_ckpt == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model" : self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, self.trained_model_path)


            ##################### Valid ################################
            counter = torch.Tensor([0]).to(self.device)
            valid_loss = torch.Tensor([0]).to(self.device)
            accuracy = torch.Tensor([0]).to(self.device)
            
            for n, batch in enumerate(self.valid_loader):
                labels = batch[1].to(self.device)
                inputs = batch[0].to(self.device)
                classes = batch[3].to(self.device)
                with torch.no_grad():
                    valid_loss_add = self.compute_loss(inputs,labels)
                    accuracy_add = self.compute_acc(inputs,classes)
                valid_loss += valid_loss_add
                accuracy += accuracy_add
            valid_loss = valid_loss/len(self.valid_loader)
            accuracy = accuracy/len(self.valid_loader)
            print("Valid Loss : {}\nAccuracy : {}".format(valid_loss, accuracy))
            self.writer.add_scalar("Synthesis/Valid_Loss", valid_loss, epoch)
            self.writer.add_scalar("Synthesis/Accuracy", accuracy, epoch)
            self.writer.flush()

            

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
                    "model" : self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, self.best_trained_model_path)

            if old_valid < valid_loss:
                counter += 1

            if counter >= 10 :
                print("Overfitting, train stopped")
                break

            old_valid = valid_loss

            ####### Validation sur les mesures #######
            valid_loss = torch.Tensor([0]).to(self.device)
            accuracy = torch.Tensor([0]).to(self.device)

            for n, batch in enumerate(self.valid_loader_mes):
                labels = batch[1].to(self.device)
                inputs = batch[0].to(self.device)
                classes = batch[3].to(self.device)
                with torch.no_grad():
                    valid_loss_add = self.compute_loss(inputs,labels)
                    accuracy_add = self.compute_acc(inputs,classes)
                valid_loss += valid_loss_add
                accuracy += accuracy_add

            valid_loss = valid_loss/len(self.valid_loader_mes)
            accuracy = accuracy/len(self.valid_loader_mes)
            print("Mesures -> Loss : {} & Accuracy : {}".format(valid_loss, accuracy))
            self.writer.add_scalar("Mesures/Valid Loss", valid_loss, epoch)
            self.writer.add_scalar("Mesures/Accuracy", accuracy, epoch)
            self.writer.flush()
