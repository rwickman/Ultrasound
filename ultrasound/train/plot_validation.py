from torch.utils.data import DataLoader
import scipy.io as io
import os
import json
import matplotlib.pyplot as plt

from model import DensityModel
from config import *
from dataset import create_datasets, load_aug_data
from dataset import UltrasoundDataset


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())



train_dataset, val_dataset, test_dataset = create_datasets()

# SOS_maps, FSA_mats = load_aug_data()      

# train_dataset = UltrasoundDataset(SOS_maps, FSA_mats)

val_loader = DataLoader(
    train_dataset,
    2,
    num_workers=4,
    prefetch_factor=4,
    drop_last = False,
    shuffle=False,
    multiprocessing_context="fork")

model = DensityModel().eval()
model_dict = torch.load(os.path.join(save_dir, "model.pt"))
#model_dict = torch.load(os.path.join("model_init", "model_100.pt"))
model.load_state_dict(model_dict["model"])


# SOS_true, FSA = dataset[0]
# SOS_true = SOS_true.to(device)
# FSA = FSA.to(device).unsqueeze(0)
# print("FSA.shape", FSA.shape)
# SOS_pred = model(FSA)


# fig, axs = plt.subplots(2)

# axs[0].imshow(SOS_true.detach().cpu())
# axs[1].imshow(normalize(SOS_pred[0].detach().cpu()))
# print("SOS_pred.min()", SOS_pred.min(), SOS_pred.max())

# plt.show()




for batch in val_loader:
    SOS_true, FSA = batch
    # SOS_true = SOS_true.to(device)
    # FSA = FSA.to(device)
    SOS_pred = model(FSA)
        
    fig, axs = plt.subplots(2, 2)
    for i in range(2):
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

    print("SOS_pred.min()", SOS_pred.min(), SOS_pred.max())


    
    
    plt.show()
    



