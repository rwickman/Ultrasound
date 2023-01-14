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
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

train_dataset, val_dataset, test_dataset = create_datasets()

val_loader = DataLoader(
    val_dataset,
    2,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    drop_last = False,
    shuffle=True,
    multiprocessing_context="fork")

# Create the model
model = DensityModel().eval()

# Load the model dict
model_dict = torch.load(os.path.join(save_dir, "model.pt"))

# Load the model parameters into the model
model.load_state_dict(model_dict["model"])

#binary_mask = torch.tensor(io.loadmat("/media/data/datasets/Ultrasound/binaryMask.mat")["binaryMask"])
#binary_mask_aug = torch.flip(binary_mask, dims=[1])
#binary_mask_val = (1540 - 1478) / (2425 - 1478)
# binary_mask[binary_mask == 0] = 10
# Iterate over dataset
cur_iter = 0
aug = lambda x: torch.flip(x, dims=[2])
aug_FSA = lambda x: torch.flip(x, dims=[1, 2])
for batch in val_loader:
    SOS_true, FSA = batch

    # aug = sample_aug()
    # if aug:
    
    # SOS_true[:, binary_mask ==0] = binary_mask_val
    SOS_true[0] = aug(SOS_true)[0]
    SOS_pred = torch.zeros_like(SOS_true)
    SOS_pred[0] = model(aug_FSA(FSA)[0].unsqueeze(0))
    SOS_pred[1] = model(FSA[1].unsqueeze(0))

    # SOS_pred[:, binary_mask ==0] = binary_mask_val
    # SOS_pred = model(FSA)
    print(SOS_true, SOS_true.mean())
    print(SOS_pred, SOS_pred.mean())
    # Create prediction
    #SOS_pred = model(FSA)

    # Plot the results    
    fig, axs = plt.subplots(2, 2)
    
    print("cur_iter", cur_iter)
    for i in range(len(SOS_pred)):
        print(f"SSIM for img {i + 1}: {structural_similarity(SOS_pred[i].detach().cpu().numpy(), SOS_true[i].detach().cpu().numpy())}")
        print(f"PSNR for img {i + 1}: {peak_signal_noise_ratio(SOS_pred[i].detach().cpu().numpy(), SOS_true[i].detach().cpu().numpy())}")
        
        cur_iter += 1
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