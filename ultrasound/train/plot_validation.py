from torch.utils.data import DataLoader
import scipy.io as io
import os
import json
import matplotlib.pyplot as plt
import torch
import random
from model import DensityModel
from config import *
from dataset import create_datasets, load_aug_data, sample_aug
from dataset import UltrasoundDataset


train_dataset, val_dataset, test_dataset = create_datasets()

val_loader = DataLoader(
    val_dataset,
    2,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    drop_last = False,
    shuffle=False,
    multiprocessing_context="fork")

# Create the model
model = DensityModel().eval()

# Load the model dict
model_dict = torch.load(os.path.join(save_dir, "model.pt"))

# Load the model parameters into the model
model.load_state_dict(model_dict["model"])



# Iterate over dataset
cur_iter = 0
for batch in val_loader:
    SOS_true, FSA = batch

    aug = sample_aug()
    if aug:
        SOS_true = aug(SOS_true)

    SOS_pred = model(FSA, aug=aug)
    print(SOS_true, SOS_true.mean())
    # Create prediction
    #SOS_pred = model(FSA)

    # Plot the results    
    fig, axs = plt.subplots(2, 2)
    print("cur_iter", cur_iter)
    for i in range(2):
        cur_iter += 1
        print(SOS_true[i][-1][-1])
        print(SOS_pred[i][-1][-1])
        print("SOS_pred", float(SOS_pred[i].detach().cpu().min()), float(SOS_pred[i].detach().cpu().max()), float(SOS_pred[i].detach().cpu().mean()))

        axs[0, i].imshow(SOS_true[i].detach().cpu())
        axs[0, i].set(
            title="Ground Truth SOS Map"
        )
        axs[1, i].imshow(SOS_pred[i].detach().cpu())
        axs[1, i].set(
            title="Predicted SOS Map"
        )
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    
    plt.show()