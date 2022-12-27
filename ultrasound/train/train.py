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
from discriminator import Discriminator, discriminator_loss
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
        
        self.val_loader = DataLoader(
            self.val_dataset,
            6,
            num_workers=2,
            prefetch_factor=6,
            drop_last = False,
            shuffle=True,
            multiprocessing_context="fork")

        self.train_dict = {
            "train_loss": [],
            "val_loss": [],
            "val_psnr": [],
            "val_ssim": []
        }


        self.model = DensityModel().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.L1Loss()

        if use_adversarial_loss:
            self.train_dict["adv_loss"] = []
            self.train_dict["disc_loss"] = []
            self.disc = Discriminator().to(device)
            self.disc_optimizer = optim.AdamW(self.disc.parameters(), lr=disc_lr)
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if load:
            self.load()

        self.avg_adv_loss = 0

        print("SIG REDUCE PARAMETERS: ", sum(p.numel() for p in self.model.sig_reduce.parameters() if p.requires_grad))
        # print("SIG ATT PARAMETERS: ", sum(p.numel() for p in self.model.sig_att.parameters() if p.requires_grad))
        print("SIG DEC PARAMETERS: ", sum(p.numel() for p in self.model.sig_dec.parameters() if p.requires_grad))
        print("TOTAL NUM PARAMETERS: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.binary_mask = torch.tensor(io.loadmat("/media/data/datasets/Ultrasound/binaryMask.mat")["binaryMask"]).to(device)

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
        self.optimizer.param_groups[0]["weight_decay"] = weight_decay

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
        val_psnr = 0
        val_ssim = 0
        self.model.eval()
        
        for batch in self.val_loader:
            SOS_true, FSA = batch
            SOS_true = SOS_true.to(device) * self.binary_mask
            FSA = FSA.to(device)
            SOS_pred = self.model(FSA) * self.binary_mask
            val_loss += self.loss_fn(SOS_pred, SOS_true).item() * FSA.shape[0]
            for i in range(len(SOS_true)):
                val_ssim += structural_similarity(SOS_pred[i].detach().cpu().numpy(), SOS_true[i].detach().cpu().numpy())
                val_psnr += peak_signal_noise_ratio(SOS_pred[i].detach().cpu().numpy(), SOS_true[i].detach().cpu().numpy())
        
        self.model.train()
        val_loss = val_loss / len(self.val_dataset)
        val_ssim = val_ssim / len(self.val_dataset)
        val_psnr = val_psnr / len(self.val_dataset)

        print("VALIDATION LOSS:", val_loss)

        del FSA
        del SOS_true
        del SOS_pred
        del batch

        return val_loss, val_ssim, val_psnr

    
    def adjust_lr(self):
        if len(self.train_dict['train_loss']) < warm_up_epochs:
            inter_step = len(self.train_dict['train_loss']) / warm_up_epochs
            print("LR", (1-inter_step) * warm_up_lr + inter_step * lr)
            self.optimizer.param_groups[0]["lr"] = (1-inter_step) * warm_up_lr + inter_step * lr
        else:
            self.optimizer.param_groups[0]["lr"] = lr

    
    def train_step(self, SOS_pred, SOS_true, cur_step):
        recon_loss = self.loss_fn(SOS_pred, SOS_true)
        disc_loss = torch.zeros(1)
        adv_loss = torch.zeros(1)
        if use_adversarial_loss:
            SOS_pred = SOS_pred.unsqueeze(1)
            SOS_true = SOS_true.unsqueeze(1)

            gt_output = self.disc(SOS_true.detach())
            sr_output = self.disc(SOS_pred)
            d_loss_gt = discriminator_loss(
                gt_output - torch.mean(sr_output),
                torch.zeros(SOS_pred.shape[0], dtype=torch.float32, device=device)) * 0.5
            d_loss_sr = discriminator_loss(
                sr_output - torch.mean(gt_output),
                torch.ones(SOS_pred.shape[0], dtype=torch.float32, device=device)) * 0.5
            
            adv_loss = d_loss_gt + d_loss_sr
            print("adv_loss", disc_lam * adv_loss)

            loss = recon_lam * recon_loss + disc_lam * adv_loss
            # Train the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.avg_adv_loss == 0.0:
                self.avg_adv_loss = adv_loss.detach().cpu().item()
            else:
                self.avg_adv_loss = self.avg_adv_loss * 0.9 + adv_loss.detach().cpu().item() * 0.1
            
            print("self.avg_adv_loss",  self.avg_adv_loss * disc_lam)
            if disc_lam * self.avg_adv_loss < 0.01:
                gt_output = self.disc(SOS_true)
                sr_output = self.disc(SOS_pred.detach())
                true_loss = discriminator_loss(gt_output - torch.mean(sr_output), torch.ones(SOS_pred.shape[0], dtype=torch.float32, device=device))
                fake_loss = discriminator_loss(sr_output - torch.mean(gt_output), torch.zeros(SOS_pred.shape[0], dtype=torch.float32, device=device))
                disc_loss = (true_loss + fake_loss) / 2
                #print("UPDATING DISC")
                print("gt_output", gt_output, "sr_output", sr_output)
                print("gt_output - torch.mean(sr_output)", gt_output - torch.mean(sr_output))
                print("sr_output - torch.mean(gt_output)", sr_output - torch.mean(gt_output))
                

                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                print("self.disc.features[-3] grad", self.disc.features[-3].weight.grad.min(), self.disc.features[-3].weight.grad.max())
                print("self.disc.features[-3] grad", self.disc.features[0].weight.grad.min(), self.disc.features[0].weight.grad.max())
                self.disc_optimizer.step()
            
        else:
            loss = recon_lam * recon_loss

            # Train the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("recon_loss", recon_loss.item())

        
        return recon_loss, adv_loss, disc_loss

    def train(self):
        historical_batches = []
        aug = lambda x: torch.flip(x, dims=[2])
        aug_FSA = lambda x: torch.flip(x, dims=[1, 2])

        for i in range(epochs):
            cur_step = 0
            self.adjust_lr()
            train_loss = 0
            adv_loss_sum = 0
            disc_loss_sum = 0
            train_exs = 0
            start_time = time.time()

            for batch in self.train_loader:
                SOS_true, FSA = batch 
                SOS_true = SOS_true.to(device)
                FSA = FSA.to(device)

                if random.random() <= 0.5:
                    SOS_true = aug(SOS_true)
                    FSA = aug_FSA(FSA)
                
                SOS_pred = self.model(FSA)

                # Update the model
                loss, adv_loss, disc_loss = self.train_step(SOS_pred, SOS_true, cur_step)

                if cur_step % 32 == 0:
                    print("\nsig_at.enc.layers[0].linear2", self.model.sig_att.enc.layers[0].linear2.weight.grad.min(), self.model.sig_att.enc.layers[0].linear2.weight.grad.max(), self.model.sig_att.enc.layers[0].linear2.weight.grad.mean(), self.model.sig_att.enc.layers[0].linear2.weight.grad.std())
                    #print("sig_at.enc.layers[1].linear2", self.model.sig_att.enc.layers[1].linear2.weight.grad.min(), self.model.sig_att.enc.layers[1].linear2.weight.grad.max(), self.model.sig_att.enc.layers[1].linear2.weight.grad.mean(), self.model.sig_att.enc.layers[1].linear2.weight.grad.std())                
                    print("sig_reduce.time_convs[0]", self.model.sig_reduce.time_convs[0].weight.grad.min(), self.model.sig_reduce.time_convs[0].weight.grad.max())
                    #print("sig_reduce.fn_out", self.model.sig_reduce.fn_out.weight.grad.min(), self.model.sig_reduce.fn_out.weight.grad.max())
                    print("sig_dec.conv_out", self.model.sig_dec.unet.outc.conv.weight.grad.min(), self.model.sig_dec.unet.outc.conv.weight.grad.max())
                    #print("sig_dec.conv_out", self.model.sig_dec.conv_out[-1].weight.grad.min(), self.model.sig_dec.conv_out[-1].weight.grad.max())
                    #print("sig_dec.conv_out", self.model.sig_dec.att_unet.Conv_1x1.weight.grad.min(), self.model.sig_dec.att_unet.Conv_1x1.weight.grad.max())
                    print("SOS_pred.min()", SOS_pred.min(), SOS_pred.max())

                train_loss += loss.item() * FSA.shape[0]
                adv_loss_sum += adv_loss.item() * FSA.shape[0]
                disc_loss_sum += disc_loss.item() * FSA.shape[0]
                train_exs += FSA.shape[0]
                cur_step += 1

                del SOS_true
                del SOS_pred
                del FSA
                del batch


            # print(gc.collect())
            print(torch.cuda.empty_cache())


            self.optimizer.zero_grad()
            print("TRAIN TIME", time.time() - start_time)
            
            if len(self.train_dict["train_loss"]) % save_iter == 0:
                start_time = time.time()
                val_loss, val_ssim, val_psnr = self.validate()
                print("VALIDATION TIME", time.time() - start_time)
                
                self.train_dict["val_loss"].append(val_loss)            
                self.train_dict["val_ssim"].append(val_ssim)
                self.train_dict["val_psnr"].append(val_psnr)
                        
            print("TRAIN LOSS", train_loss/train_exs)
            self.train_dict["train_loss"].append(train_loss/train_exs)
            if use_adversarial_loss:
                self.train_dict["adv_loss"].append(adv_loss_sum/train_exs)
                self.train_dict["disc_loss"].append(disc_loss_sum/train_exs)

            self.save()
            print("Saved model.")


trainer = Trainer()
trainer.train()