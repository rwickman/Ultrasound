
from torch.utils.data import DataLoader
import scipy.io as io
import os
import json
import matplotlib.pyplot as plt

from model import DensityModel
from config import *
from dataset import create_slices_dataset

dataset = create_slices_dataset()

loader = DataLoader(
    dataset,
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

slices_save_dir = "slices_output/pred_maps.mat"
cur_iter = 0
sos_maps = []
for batch in loader:
    SOS_true, FSA = batch

    # Create prediction
    SOS_pred = model(FSA)
    print(SOS_pred.shape)
    # Convert to original range
    SOS_pred = SOS_pred * (SOS_RANGE[1] - SOS_RANGE[0]) + SOS_RANGE[0]
    for sos_map in SOS_pred:
        sos_maps.append(sos_map)
    print()

sos_maps = torch.cat(sos_maps)
d = {"SOSMap": sos_maps.tolist()}

io.savemat(slices_save_dir, d)