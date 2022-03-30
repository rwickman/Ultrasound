import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as io

# TODO: REMOVE THIS
import matplotlib.pyplot as plt

from model import DensityModel
from config import *

class Trainer:
    def __init__(self):
        self.train_inp = io.loadmat("/media/data/datasets/Ultrasound/experiment_01/OutputRepacked/FSA_Layer03.mat")["FSA"]
        self.train_inp = (self.train_inp - self.train_inp.min()) / (self.train_inp.max() - self.train_inp.min())
        self.train_inp = torch.tensor(self.train_inp, device=device).unsqueeze(0).float()
        


        self.train_out = io.loadmat("/media/data/datasets/Ultrasound/experiment_01/SOSMapAbOnly.mat")["SOSMap"]
        self.train_out = (self.train_out - self.train_out.min()) / (self.train_out.max() - self.train_out.min())
        self.train_out = torch.tensor(self.train_out, device=device).permute((2, 0, 1)).float()
        print("self.train_out.shape", self.train_out.shape)
        

        self.model = DensityModel().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.SmoothL1Loss()
        print("SIG REDUCE PARAMETERS: ", sum(p.numel() for p in self.model.sig_reduce.parameters() if p.requires_grad))
        print("SIG ATT PARAMETERS: ", sum(p.numel() for p in self.model.sig_att.parameters() if p.requires_grad))
        print("SIG DEC PARAMETERS: ", sum(p.numel() for p in self.model.sig_dec.parameters() if p.requires_grad))
        print("TOTAL NUM PARAMETERS: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))




    
    def train(self):
        for i in range(10000):

            # Get model prediction
            out = self.model(self.train_inp)
            
            # Compute loss
            loss = self.loss_fn(out[:, 0], self.train_out[3:4])
            out_of_range_loss = self.loss_fn(out, torch.clip(out.detach(), 0.0, 1.0))
            loss = loss + (out_of_range_loss * 10)
            # Train the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print("LOSS:", loss.item())
            print("out_of_range_loss: ", out_of_range_loss.item() * 10)
            if (i + 1) % 200 == 0:

                
                print("REDUCE FN GRAD: ", self.model.sig_reduce.fn_out.weight.grad.min(), self.model.sig_reduce.fn_out.weight.grad.max(), self.model.sig_reduce.fn_out.weight.grad.mean())
                print("REDUCE FN GRAD: ", self.model.sig_dec.conv_out.weight.grad.min(), self.model.sig_dec.conv_out.weight.grad.max(), self.model.sig_dec.conv_out.weight.grad.mean())
                print("out.min()", out.min(), out.max(), out.mean(), out.std())
                print("self.train_out[3]", self.train_out[3].min(), self.train_out[3].max(), self.train_out[3].mean(), self.train_out[3].std())
                fig, axs = plt.subplots(2)
                axs[0].imshow(self.train_out[3].detach().cpu())
                axs[1].imshow(out[0, 0].detach().cpu())
                plt.show()

        fig, axs = plt.subplots(2)
        axs[0].imshow(self.train_out[0].detach().cpu())
        axs[1].imshow(out[0, 0].detach().cpu())
        plt.show()


            
        
        

trainer = Trainer()
trainer.train()
# print("READING DATA")


# x = torch.randn(4, 64, 64, SIGNAL_LENGTH).cuda()
# # x = torch.tensor(x).cuda()
# m = 
# x = m(x)


# print("train_data", train_data.min(), train_data.max(), train_data.mean(), train_data.mean())


# print("DATA SIZE: ", x.shape)
