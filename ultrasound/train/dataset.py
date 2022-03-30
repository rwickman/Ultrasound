import torch
from torch.utils.data import Dataset
import scipy.io as io
import os
import numpy as np

from config import *

def load_aug_data():
    
    aug_SOS_mats = ["data/aug_data/aug_sos_0.mat", "data/aug_data/aug_sos_1.mat", "data/aug_data/aug_sos_2.mat"]
    SOS_maps = []
    for aug_SOS_mat in aug_SOS_mats:
        SOS_maps.append(io.loadmat(aug_SOS_mat)["SOS"])
        
    
    SOS_maps = np.array(SOS_maps)
    print("AUG SOS_map.max()", SOS_maps.max(), SOS_maps.min())
    SOS_maps = (SOS_maps - SOS_maps.min()) / (SOS_maps.max() - SOS_maps.min())
    print("AUG SOS_map.max()", SOS_maps.max(), SOS_maps.min())

    FSA_mats = [
        os.path.join(aug_FSA_dir, FSA_mat) 
        for FSA_mat in os.listdir(aug_FSA_dir) 
        if ".mat" in FSA_mat]
    FSA_mats = sorted(FSA_mats, 
        key=lambda FSA_mat: int(FSA_mat.split("/")[-1].split("aug_")[-1].split(".mat")[0]))

    print("FSA_mats", FSA_mats)
    return SOS_maps, FSA_mats



def load_data():
    SOS_map = io.loadmat(SOS_MAP_mat)["SOSMap"].astype(np.float32)

    print("SOS_map.max()", SOS_map.max(), SOS_map.min())

    SOS_map = SOS_map.transpose((2,0,1))
    SOS_map = (SOS_map - SOS_map.min()) / (SOS_map.max() - SOS_map.min())
    print("SOS_map.max()", SOS_map.max(), SOS_map.min())

    SOS_train = SOS_map[:-test_size * 2]
    

    SOS_val = SOS_map[-test_size * 2:-test_size]
    SOS_test = SOS_map[-test_size:]
    
    # Get the FSA mat files
    FSA_mats = [
        os.path.join(data_inp_dir, FSA_mat) 
        for FSA_mat in os.listdir(data_inp_dir) 
        if ".mat" in FSA_mat]
    
    # Sort them
    FSA_mats = sorted(FSA_mats, 
        key=lambda FSA_mat: int(FSA_mat.split("/")[-1].split("Layer")[-1].split(".mat")[0]))
    
    FSA_mats_train = FSA_mats[:-test_size*2]
    FSA_mats_val = FSA_mats[-test_size*2:-test_size]
    FSA_mats_test = FSA_mats[-test_size:]

    return (SOS_train, SOS_val, SOS_test), (FSA_mats_train, FSA_mats_val, FSA_mats_test)


def create_datasets():
    SOS_maps, FSA_mats = load_data()
    SOS_maps_aug, FSA_mats_aug = load_aug_data()
    print("SOS_maps.shape", SOS_maps[0].shape)
    print("SOS_maps_aug.shape", SOS_maps_aug.shape)

    SOS_maps_train = np.concatenate((SOS_maps[0], SOS_maps_aug))
    FSA_mats_train = FSA_mats[0] + FSA_mats_aug

    train_dataset = UltrasoundDataset(SOS_maps_train, FSA_mats_train)
    val_dataset = UltrasoundDataset(SOS_maps[1], FSA_mats[1])
    test_dataset = UltrasoundDataset(SOS_maps[2], FSA_mats[2])

    return train_dataset, val_dataset, test_dataset



class UltrasoundDataset(Dataset):
    def __init__(self, SOS_maps, FSA_mats):
        
        self.SOS_maps = torch.tensor(SOS_maps, dtype=torch.float32)
        
        self.FSA_mats = FSA_mats
        print("self.SOS_maps", len(self.SOS_maps))
        print("self.FSA_mats", len(self.FSA_mats))

    def __len__(self):
        return len(self.SOS_maps)
    
    def __getitem__(self, idx):
        SOS_map = self.SOS_maps[idx]
        FSA = torch.tensor(io.loadmat(self.FSA_mats[idx])["FSA"],  dtype=torch.float32)

        if FSA.shape[-1] < SIGNAL_LENGTH:
            FSA_pad = torch.zeros(NUM_TRANSMIT, NUM_TRANSMIT, SIGNAL_LENGTH, dtype=torch.float32)
            FSA_pad[:, :, :FSA.shape[-1]] = FSA
            FSA = FSA_pad

        return SOS_map, FSA 





