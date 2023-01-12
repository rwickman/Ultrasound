import torch
import torch.nn as nn
import torch.optim as optim
import random
import gc
from torch.utils.data import DataLoader
import scipy.io as io
import json, os, time

from collections import deque
import numpy as np

from model import DensityModel
from config import *
from dataset import create_datasets, sample_aug
from ultrasound.train.contrastive.contrastive import ContrastiveModel
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

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

        self.train_dict = {
            "train_loss": [],
            "val_loss": [],
            "val_psnr": [],
            "val_ssim": []
        }

        self.model = ContrastiveModel().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.L1Loss()

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if load:
            self.load()


    def save(self):
        model_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }


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

    
    def adjust_lr(self):
        if len(self.train_dict['train_loss']) < warm_up_epochs:
            inter_step = len(self.train_dict['train_loss']) / warm_up_epochs
            print("LR", (1-inter_step) * warm_up_lr + inter_step * lr)
            self.optimizer.param_groups[0]["lr"] = (1-inter_step) * warm_up_lr + inter_step * lr
        else:
            self.optimizer.param_groups[0]["lr"] = lr

    
    def train_step(self, SOS_pred, SOS_true, cur_step):
        recon_loss = self.loss_fn(SOS_pred, SOS_true)

            
        loss = recon_lam * recon_loss

        # Train the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print("recon_loss", recon_loss.item())

        
        return recon_loss

    
  

    def train(self):
        historical_batches = []
        aug = lambda x: torch.flip(x, dims=[2])
        aug_FSA = lambda x: torch.flip(x, dims=[1, 2])

        for i in range(epochs):
            cur_step = 0
            self.adjust_lr()
            train_loss = 0
            train_exs = 0
            start_time = time.time()

            for batch in self.train_loader:
                SOS_true, FSA = batch 
                SOS_true = SOS_true.to(device)
                FSA = FSA.to(device)

                if use_history:
                    if (random.random() <= 0.25 or len(historical_batches) < batch_buffer_size):
                        historical_batches.append([SOS_true.clone(), FSA.clone()])
                        if len(historical_batches) > batch_buffer_size:
                            historical_batches = historical_batches[1:]
                    elif len(historical_batches) == batch_buffer_size:
                        rand_batch_idx = random.randint(0, batch_buffer_size - 1)
                        rand_ex_idx = random.randint(0, batch_size - 1)
                        historical_batches[rand_batch_idx][0][rand_ex_idx] = SOS_true[0].clone()
                        historical_batches[rand_batch_idx][1][rand_ex_idx] = FSA[0].clone()

                if random.random() <= 0.5:
                    SOS_true = aug(SOS_true)
                    FSA = aug_FSA(FSA)
                
                SOS_pred = self.model(FSA)

                # Update the model
                loss = self.train_step(SOS_pred, SOS_true, cur_step)

                if cur_step % 32 == 0:
                    print("\nsig_at.enc.layers[0].linear2", self.model.sig_att.enc.layers[0].linear2.weight.grad.min(), self.model.sig_att.enc.layers[0].linear2.weight.grad.max(), self.model.sig_att.enc.layers[0].linear2.weight.grad.mean(), self.model.sig_att.enc.layers[0].linear2.weight.grad.std())
                    #print("sig_at.enc.layers[1].linear2", self.model.sig_att.enc.layers[1].linear2.weight.grad.min(), self.model.sig_att.enc.layers[1].linear2.weight.grad.max(), self.model.sig_att.enc.layers[1].linear2.weight.grad.mean(), self.model.sig_att.enc.layers[1].linear2.weight.grad.std())                
                    print("sig_reduce.time_convs[0]", self.model.sig_reduce.time_convs[0].weight.grad.min(), self.model.sig_reduce.time_convs[0].weight.grad.max())
                    #print("sig_reduce.fn_out", self.model.sig_reduce.fn_out.weight.grad.min(), self.model.sig_reduce.fn_out.weight.grad.max())
                    print("sig_dec.conv_out", self.model.sig_dec.unet.outc.conv.weight.grad.min(), self.model.sig_dec.unet.outc.conv.weight.grad.max())
                    #print("sig_dec.conv_out", self.model.sig_dec.conv_out[-1].weight.grad.min(), self.model.sig_dec.conv_out[-1].weight.grad.max())
                    #print("sig_dec.conv_out", self.model.sig_dec.att_unet.Conv_1x1.weight.grad.min(), self.model.sig_dec.att_unet.Conv_1x1.weight.grad.max())
                    print("SOS_pred.min()", SOS_pred.min(), SOS_pred.max())
                # if train_exs > 128:
                #     break
                train_loss += loss.item() * FSA.shape[0]
                train_exs += FSA.shape[0]
                cur_step += 1
                
                if random.random() <= history_sample_prob and len(historical_batches) >= batch_buffer_size:
                    self.rand_train(historical_batches)
                
                # if train_exs > 512:
                #     break

            del SOS_true
            del SOS_pred
            del FSA
            del batch
            
            


            print(torch.cuda.empty_cache())


            self.optimizer.zero_grad()
            print("TRAIN TIME", time.time() - start_time)
            
            # if len(self.train_dict["train_loss"]) % save_iter == 0:
            #     start_time = time.time()
            #     val_loss, val_ssim, val_psnr = self.validate()
            #     print("VALIDATION TIME", time.time() - start_time)
                
            #     self.train_dict["val_loss"].append(val_loss)            
            #     self.train_dict["val_ssim"].append(val_ssim)
            #     self.train_dict["val_psnr"].append(val_psnr)
                        
            print("TRAIN LOSS", train_loss/train_exs)
            self.train_dict["train_loss"].append(train_loss/train_exs)

            
            self.save()
            print("Saved model.")


trainer = Trainer()
trainer.train()