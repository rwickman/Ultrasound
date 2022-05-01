import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io as io
import os
import json

# TODO: REMOVE THIS
import matplotlib.pyplot as plt

from model import DensityModel
from config import *
from dataset import create_datasets

class Trainer:
    def __init__(self):
        # self.train_inp = io.loadmat("/media/data/datasets/Ultrasound/experiment_01/OutputRepacked/FSA_Layer03.mat")["FSA"]
        # self.train_inp = (self.train_inp - self.train_inp.min()) / (self.train_inp.max() - self.train_inp.min())
        # self.train_inp = torch.tensor(self.train_inp, device=device).unsqueeze(0).float()

        # self.train_out = io.loadmat("/media/data/datasets/Ultrasound/experiment_01/SOSMapAbOnly.mat")["SOSMap"]
        # self.train_out = (self.train_out - self.train_out.min()) / (self.train_out.max() - self.train_out.min())
        # self.train_out = torch.tensor(self.train_out, device=device).permute((2, 0, 1)).float()
        # print("self.train_out.shape", self.train_out.shape)
        
        self.train_dataset, self.val_dataset, self.test_dataest = create_datasets()
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size,
            num_workers=6,
            prefetch_factor=8,
            drop_last = True,
            shuffle=True,
            multiprocessing_context="fork")
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size,
            num_workers=4,
            prefetch_factor=4,
            drop_last = False,
            shuffle=True,
            multiprocessing_context="fork")

        self.train_dict = {
            "train_loss": [],
            "val_loss": []
        }

        self.model = DensityModel().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        if load:
            self.load()

        print("SIG REDUCE PARAMETERS: ", sum(p.numel() for p in self.model.sig_reduce.parameters() if p.requires_grad))
        print("SIG ATT PARAMETERS: ", sum(p.numel() for p in self.model.sig_att.parameters() if p.requires_grad))
        print("SIG DEC PARAMETERS: ", sum(p.numel() for p in self.model.sig_dec.parameters() if p.requires_grad))
        print("TOTAL NUM PARAMETERS: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))


    def save(self):
        model_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

        torch.save(model_dict, os.path.join(save_dir, f"model_{len(self.train_dict['train_loss'])}.pt"))
        torch.save(model_dict, os.path.join(save_dir, "model.pt"))

        with open(os.path.join(save_dir, "train_dict.json"), "w") as f:
            json.dump(self.train_dict, f)


    def load(self):
        print("LOADING MODELS")
        model_dict = torch.load(os.path.join(save_dir, "model.pt"))

        self.model.load_state_dict(model_dict["model"])
        self.optimizer.load_state_dict(model_dict["optimizer"])
        self.optimizer.param_groups[0]["lr"] = lr

        with open(os.path.join(save_dir, "train_dict.json")) as f:
            self.train_dict = json.load(f)
        
        print("self.train_dict", self.train_dict)


    def validate(self):
        val_loss = 0
        for batch in self.val_loader:
            SOS_true, FSA = batch
            SOS_true = SOS_true.to(device)
            FSA = FSA.to(device)
            SOS_pred = self.model(FSA)
            val_loss += self.loss_fn(SOS_pred, SOS_true).item() * FSA.shape[0]
        
        val_loss = val_loss / len(self.val_dataset)
        
        print("VALIDATION LOSS:", val_loss)
        return val_loss

    
    def train(self):
        
        for i in range(epochs):
            train_loss = 0
            train_exs = 0
            for batch in self.train_loader:
                SOS_true, FSA = batch
                SOS_true = SOS_true.to(device)
                FSA = FSA.to(device)


                # Get model prediction
                SOS_pred = self.model(FSA)

                
                # Compute loss
                loss = self.loss_fn(SOS_pred, SOS_true)
                total_loss = loss*1000
                
                
                # Train the model
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                print("LOSS:", loss.item())
                train_loss += loss.item() * FSA.shape[0]
                train_exs += FSA.shape[0]

            print("SOS_pred.min()", SOS_pred.min(), SOS_pred.max())
            print("SOS_true.min()", SOS_true.min(), SOS_true.max())
            # if (i + 1) % 2 == 0:
            #     fig, axs = plt.subplots(2)

            #     axs[0].imshow(SOS_true[0].detach().cpu())
                
            #     axs[1].imshow(SOS_pred[0].detach().cpu())
            #     plt.show()

            val_loss = self.validate()
             
            self.train_dict["val_loss"].append(val_loss)            
                    

            self.train_dict["train_loss"].append(train_loss/train_exs)
            # if i %  save_iter == 0: 
            self.save()
            print("Saved model.")


# torch.multiprocessing.set_start_method("spawn")

trainer = Trainer()
trainer.train()
# print("READING DATA")


# x = torch.randn(4, 64, 64, SIGNAL_LENGTH).cuda()
# # x = torch.tensor(x).cuda()
# m = 
# x = m(x)


# print("train_data", train_data.min(), train_data.max(), train_data.mean(), train_data.mean())


# print("DATA SIZE: ", x.shape)
