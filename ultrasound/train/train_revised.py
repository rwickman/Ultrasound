import torch
import torch.nn as nn
import torch.optim as optim
import random
import gc
from torch.utils.data import DataLoader
import scipy.io as io
import json, os, time
from tqdm import tqdm
from hdf5storage import loadmat, savemat
from metrics import Metrics

from collections import deque
import numpy as np

from model import DensityModel
from config import *
from dataset import create_datasets, sample_aug
from discriminator import Discriminator, discriminator_loss
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, run_name=None):
        self.train_dataset, self.val_dataset, self.test_dataset = create_datasets()
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False,
            shuffle=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            6,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False,
            shuffle=False,
            pin_memory=True
        )

        # If data size is small enough, load it all into memory at once
        # if not use_synth_data and not use_only_synth_data:
        if len(self.train_loader) * batch_size <= 200:
            self.train_loader = [batch for batch in self.train_loader]
        if len(self.val_loader) * batch_size <= 200:
            self.val_loader = [batch for batch in self.val_loader]

        self.train_dict = {}

        self.model = DensityModel().to(device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # self.loss_fn = nn.L1Loss()
        self.loss_fn = nn.SmoothL1Loss()

        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        if run_name is None:
            run_name = time.strftime("%Y%m%d_%H%M")
        self.save_dir = "%s/%s" % (models_dir, run_name)
        print("Processing run %s..." % self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if load:
            self.load()

        self.metrics = Metrics(self.loss_fn)
        self.binary_mask = torch.tensor(
            io.loadmat("%s/binaryMask.mat" % DATADIR)["binaryMask"]
        ).to(device)

    def save(self, current_epoch):
        model_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        i = current_epoch + 1
        if i % save_iter == 0:
            torch.save(
                model_dict,
                os.path.join(self.save_dir, "model%03d.pt" % i),
            )
        torch.save(model_dict, os.path.join(self.save_dir, "model.pt"))

        self.validate()
        savemat("%s/summary.mat" % self.save_dir, self.train_dict)

        # with open(os.path.join(save_dir, "train_dict.json"), "w") as f:
        #     json.dump(self.train_dict, f)

    def load(self):
        print("LOADING MODELS")
        model_dict = torch.load(os.path.join(self.save_dir, "model.pt"))

        self.model.load_state_dict(model_dict["model"])
        self.optimizer.load_state_dict(model_dict["optimizer"])

        self.optimizer.param_groups[0]["lr"] = lr
        self.optimizer.param_groups[0]["weight_decay"] = weight_decay

        if use_adversarial_loss and "disc" in model_dict:
            self.disc.load_state_dict(model_dict["disc"])
            self.disc_optimizer.load_state_dict(model_dict["disc_optimizer"])
            self.disc_optimizer.param_groups[0]["lr"] = disc_lr

        with open(os.path.join(self.save_dir, "train_dict.json")) as f:
            train_dict_saved = json.load(f)
            for k, v in train_dict_saved.items():
                self.train_dict[k] = v

    def validate(self):
        self.model.eval()

        for batch in self.val_loader:
            ctrue, iqraw = batch
            ctrue = ctrue.to(device) * self.binary_mask
            iqraw = iqraw.to(device)
            cpred = self.model(iqraw) * self.binary_mask
            cpred01 = cpred.detach().cpu().numpy()
            ctrue01 = ctrue.detach().cpu().numpy()
            self.metrics.eval(cpred01, ctrue01)

        del iqraw
        del ctrue
        del cpred
        del batch

    def adjust_lr(self, current_epoch, lr):

        if current_epoch < warm_up_epochs:
            inter_step = current_epoch / warm_up_epochs
            # print("LR", (1 - inter_step) * warm_up_lr + inter_step * lr)
            self.optimizer.param_groups[0]["lr"] = (
                1 - inter_step
            ) * warm_up_lr + inter_step * lr
        else:
            self.optimizer.param_groups[0]["lr"] = lr

    def train_step(self, cpred, ctrue, cur_step):
        recon_loss = self.loss_fn(cpred, ctrue)
        loss = recon_lam * recon_loss

        # Train the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, lr, verbose=True):
        # aug = lambda x: torch.flip(x, dims=[2])
        # aug_FSA = lambda x: torch.flip(x, dims=[1, 2])

        self.validate()
        self.metrics.appendToDict(self.train_dict, "val_")
        if verbose:
            self.metrics.print("Validation")
        self.metrics.reset()

        if verbose:
            print("Learning rate is %f" % lr)

        for i in range(epochs):
            if verbose:
                print("Epoch %d" % i)
            cur_step = 0
            self.adjust_lr(i, lr)

            for batch in tqdm(self.train_loader, disable=not verbose):
                # Ordinary run
                ctrue, iqraw = batch
                ctrue = ctrue.to(device)
                iqraw = iqraw.to(device)
                # Update the model
                cpred = self.model(iqraw)
                ctrue = ctrue * self.binary_mask
                cpred = cpred * self.binary_mask
                self.train_step(cpred, ctrue, cur_step)
                # Update the metrics
                ctrue01 = ctrue.detach().cpu().numpy()
                cpred01 = cpred.detach().cpu().numpy()
                self.metrics.eval(cpred01, ctrue01)

                # # Flipped left-right run
                # ctrue, iqraw = batch
                # ctrue = ctrue.to(device)
                # iqraw = iqraw.to(device)
                # ctrue = aug(ctrue)
                # iqraw = aug_FSA(iqraw)
                # cpred = self.model(iqraw)
                # ctrue = ctrue * self.binary_mask
                # cpred = cpred * self.binary_mask
                # # Update the model
                # self.train_step(cpred, ctrue, cur_step)
                # # Update the metrics
                # ctrue01 = ctrue.detach().cpu().numpy()
                # cpred01 = cpred.detach().cpu().numpy()
                # self.metrics.eval(cpred01, ctrue01)
                del ctrue
                del cpred
                del iqraw
                del batch
            # # print(gc.collect())
            # print(torch.cuda.empty_cache())

            self.optimizer.zero_grad()

            # Output results
            if verbose:
                self.metrics.print("Training")
            self.metrics.appendToDict(self.train_dict, "train_")
            self.metrics.reset()

            # Validate again
            self.validate()
            self.metrics.appendToDict(self.train_dict, "val_")
            if verbose:
                self.metrics.print("Validation")
            self.metrics.reset()

            # Save results
            self.train_dict["lr"] = lr
            self.save(i)

        return self.train_dict["val_loss"][-1]


def main():

    lr = 2e-4
    trainer = Trainer()
    trainer.train(lr)


if __name__ == "__main__":
    main()
