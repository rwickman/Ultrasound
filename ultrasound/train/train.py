import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader
import scipy.io as io
import json, os, time

from collections import deque
import numpy as np

from model import DensityModel
from config import *
from dataset import create_datasets
from discriminator import Discriminator, discriminator_loss

class Trainer:
    def __init__(self):
        self.train_dataset, self.val_dataset, self.test_dataest = create_datasets()
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            drop_last = True,
            shuffle=True,
            multiprocessing_context="fork")
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size,
            num_workers=2,
            prefetch_factor=6,
            drop_last = False,
            shuffle=True,
            multiprocessing_context="fork")

        self.train_dict = {
            "train_loss": [],
            "val_loss": [],
            "adv_loss": [],
            "disc_loss": []
        }

        self.model = DensityModel().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        if use_adversarial_loss:
            self.disc = Discriminator().to(device)
            self.disc_optimizer = optim.AdamW(self.disc.parameters(), lr=disc_lr)
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if load:
            self.load()

        print("SIG REDUCE PARAMETERS: ", sum(p.numel() for p in self.model.sig_reduce.parameters() if p.requires_grad))
        print("SIG ATT PARAMETERS: ", sum(p.numel() for p in self.model.sig_att.parameters() if p.requires_grad))
        print("SIG DEC PARAMETERS: ", sum(p.numel() for p in self.model.sig_dec.parameters() if p.requires_grad))
        print("TOTAL NUM PARAMETERS: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))


    def save(self):
        model_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if use_adversarial_loss:
            model_dict["disc"] = self.disc.state_dict()
            model_dict["disc_optimizer"] = self.disc_optimizer.state_dict()

        if len(self.train_dict['train_loss']) % save_iter == 0:
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

        if use_adversarial_loss and "disc" in model_dict:
            self.disc.load_state_dict(model_dict["disc"])
            self.disc_optimizer.load_state_dict(model_dict["disc_optimizer"])
            self.disc_optimizer.param_groups[0]["lr"] = disc_lr


        with open(os.path.join(save_dir, "train_dict.json")) as f:
            train_dict_saved = json.load(f)
            for k, v in train_dict_saved.items():
                self.train_dict[k] = v


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

    
    def train_step(self, SOS_pred, SOS_true, cur_step):
        recon_loss = self.loss_fn(SOS_pred, SOS_true)
        disc_loss = torch.zeros(1)
        adv_loss = torch.zeros(1)
        if use_adversarial_loss:
            SOS_pred = SOS_pred.unsqueeze(1)
            SOS_true = SOS_true.unsqueeze(1)

            disc_true_pred = self.disc(SOS_true)
            disc_fake_pred = self.disc(SOS_pred.detach())

            true_loss = discriminator_loss(disc_true_pred, torch.ones(SOS_pred.shape[0], dtype=torch.float32, device=device) - 0.2)
            fake_loss = discriminator_loss(disc_fake_pred, torch.zeros(SOS_pred.shape[0], dtype=torch.float32, device=device))
            
            
            disc_loss = true_loss + fake_loss
            if cur_step % 4 == 0:
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                self.disc_optimizer.step()
        
            disc_fake_pred = self.disc(SOS_pred)
            adv_loss = discriminator_loss(disc_fake_pred, torch.ones(SOS_pred.shape[0], dtype=torch.float32, device=device) - 0.2) 
            loss = recon_lam * recon_loss + disc_lam * adv_loss
            # Train the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        else:
            loss = recon_lam * recon_loss
            # Train the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("recon_loss", recon_loss.item())

        
        
        return recon_loss, adv_loss, disc_loss

        

    def train(self):
        
        for i in range(epochs):
            train_loss = 0
            adv_loss_sum = 0
            disc_loss_sum = 0
            train_exs = 0
            start_time = time.time()
            prev_batches = deque([])
            for batch in self.train_loader:
                cur_step = 0
                rand_idxs = np.random.permutation(len(prev_batches) + 1)
                for i in rand_idxs:
                    if len(prev_batches) < batch_buffer_size//2:
                        break
                    if i >= 1 and len(prev_batches) > i-1:
                        SOS_true, FSA = prev_batches[i-1]
                    else:
                        SOS_true, FSA = batch

                    SOS_true = SOS_true.to(device)
                    FSA = FSA.to(device)

                    # Get model prediction
                    SOS_pred = self.model(FSA)

                    # Update the model
                    loss, adv_loss, disc_loss = self.train_step(SOS_pred, SOS_true, cur_step)
                    

                    if train_exs % 128 == 0:
                        print("\nsig_at.enc.layers[0].linear2", self.model.sig_att.enc.layers[0].linear2.weight.grad.min(), self.model.sig_att.enc.layers[0].linear2.weight.grad.max(), self.model.sig_att.enc.layers[0].linear2.weight.grad.mean(), self.model.sig_att.enc.layers[0].linear2.weight.grad.std())
                        print("sig_at.enc.layers[2].linear2", self.model.sig_att.enc.layers[2].linear2.weight.grad.min(), self.model.sig_att.enc.layers[2].linear2.weight.grad.max(), self.model.sig_att.enc.layers[2].linear2.weight.grad.mean(), self.model.sig_att.enc.layers[2].linear2.weight.grad.std())                
                        print("sig_reduce.conv_out", self.model.sig_reduce.time_convs[0].weight.grad.min(), self.model.sig_reduce.time_convs[0].weight.grad.max())
                        print("sig_reduce.fn_out", self.model.sig_reduce.fn_out.weight.grad.min(), self.model.sig_reduce.fn_out.weight.grad.max())
                        print("sig_dec.conv_out", self.model.sig_dec.conv_out.weight.grad.min(), self.model.sig_dec.conv_out.weight.grad.max())
                        print("SOS_pred.min()", SOS_pred.min(), SOS_pred.max())

                    train_loss += loss.item() * FSA.shape[0]
                    adv_loss_sum += adv_loss.item() * FSA.shape[0]
                    disc_loss_sum += disc_loss.item() * FSA.shape[0]
                    train_exs += FSA.shape[0]
                    cur_step += 1

                if len(prev_batches) < batch_buffer_size:
                    prev_batches.append(batch)
                else:
                    prev_batches.popleft()
                    prev_batches.append(batch)
                    

                if train_exs > 1024:
                    break 


            print("SOS_pred.min()", SOS_pred.min(), SOS_pred.max())
            print("SOS_true.min()", SOS_true.min(), SOS_true.max())
            # if (i + 1) % 2 == 0:
            #     fig, axs = plt.subplots(2)

            #     axs[0].imshow(SOS_true[0].detach().cpu())
                
            #     axs[1].imshow(SOS_pred[0].detach().cpu())
            #     plt.show()
            print("TRAIN TIME", time.time() - start_time)
            
            if len(self.train_dict["train_loss"]) % save_iter == 0:
                start_time = time.time()
                val_loss = self.validate()
                print("VALIDATION TIME", time.time() - start_time)
                
                self.train_dict["val_loss"].append(val_loss)            
                        
            print("TRAIN LOSS", train_loss/train_exs)
            self.train_dict["train_loss"].append(train_loss/train_exs)
            self.train_dict["adv_loss"].append(adv_loss_sum/train_exs)
            self.train_dict["disc_loss"].append(disc_loss_sum/train_exs)

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
