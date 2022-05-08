import torch
from torch.utils.data import Dataset
import scipy.io as io
import os
import numpy as np
import time
from config import *

def load_aug_data():
    
    # Load the SOS maps
    SOS_maps = []
    # density_maps = []
    for aug_SOS_mat in aug_SOS_mats:
        SOS_maps.append(io.loadmat(aug_SOS_mat)["SOSMap"])
    
    SOS_maps = np.concatenate(SOS_maps, axis=-1)
    print("SOS_maps.shape", SOS_maps.shape)
    SOS_maps = SOS_maps.transpose((2, 0, 1))
    print("AUG SOS_maps ", SOS_maps.min(), SOS_maps.max())
    SOS_maps = (SOS_maps - SOS_maps.min()) / (SOS_maps.max() - SOS_maps.min())
    print("AUG SOS_maps.shape", SOS_maps.min(), SOS_maps.max())


    # Sort the FSA files
    FSA_mats_inp = []
    for aug_FSA_dir in aug_FSA_inp_dirs:
        FSA_mats = [
            os.path.join(aug_FSA_dir, FSA_mat) 
            for FSA_mat in os.listdir(aug_FSA_dir) 
            if ".npy" in FSA_mat]

        FSA_mats = sorted(FSA_mats, 
            key=lambda FSA_mat: int(FSA_mat.split("/")[-1].split("FSA_Layer")[-1].split(".npy")[0]))
        FSA_mats_inp.extend(FSA_mats)
    print(len(SOS_maps), len(FSA_mats_inp))
    return SOS_maps, FSA_mats_inp

def load_data():
    SOS_map = io.loadmat(SOS_MAP_mat)["SOSMap"].astype(np.float32)

    SOS_map = SOS_map.transpose((2,0,1))
    print("Ab SOS_maps ", SOS_map.min(), SOS_map.max())
    SOS_map = (SOS_map - SOS_map.min()) / (SOS_map.max() - SOS_map.min())
    print("Ab SOS_maps ", SOS_map.min(), SOS_map.max())

    SOS_val = SOS_map[:test_size]
    SOS_test = SOS_map[test_size:]

    # Get the FSA mat files
    # FSA_mats = [
    #     os.path.join(data_inp_dir, FSA_mat) 
    #     for FSA_mat in os.listdir(data_inp_dir) 
    #     if ".mat" in FSA_mat]
    FSA_mats = [
        os.path.join(data_inp_dir, FSA_mat) 
        for FSA_mat in os.listdir(data_inp_dir) 
        if ".npy" in FSA_mat]    
    # Sort them
    # FSA_mats = sorted(FSA_mats, 
    #     key=lambda FSA_mat: int(FSA_mat.split("/")[-1].split("Layer")[-1].split(".mat")[0]))
    FSA_mats = sorted(FSA_mats, 
        key=lambda FSA_mat: int(FSA_mat.split("/")[-1].split("FSA_Layer")[-1].split(".npy")[0]))
    
    # FSA_mats_train = FSA_mats[:-test_size*2]
    FSA_mats_val = FSA_mats[:test_size]
    FSA_mats_test = FSA_mats[test_size:]

    return (SOS_val, SOS_test), (FSA_mats_val, FSA_mats_test)


def create_datasets():
    SOS_maps, FSA_mats = load_data()
    SOS_maps_train, FSA_mats_train = load_aug_data()

    SOS_maps_train = np.concatenate((SOS_maps_train, SOS_maps[0][:3], SOS_maps[1][-3:]))
    FSA_mats_train = FSA_mats_train + FSA_mats[0][:3] + FSA_mats[1][-3:]

    train_dataset = UltrasoundDataset(SOS_maps_train, FSA_mats_train)
    val_dataset = UltrasoundDataset(SOS_maps[0][3:], FSA_mats[0][3:])
    test_dataset = UltrasoundDataset(SOS_maps[1][:-3], FSA_mats[1][:-3])

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
        if ".mat" in self.FSA_mats[idx]:

            FSA = torch.tensor(io.loadmat(self.FSA_mats[idx])["FSA"],  dtype=torch.float32)
        else:
            FSA = torch.tensor(np.load(self.FSA_mats[idx]),  dtype=torch.float32)
        # FSA = torch.tensor(np.load(self.FSA_mats[idx]),  dtype=torch.float32)
        if FSA.shape[-1] < SIGNAL_LENGTH:
            print("TOO SMALL")
            FSA_pad = torch.zeros(NUM_TRANSMIT, NUM_TRANSMIT, SIGNAL_LENGTH, dtype=torch.float32)
            FSA_pad[:, :, :FSA.shape[-1]] = FSA
            FSA = FSA_pad

        return SOS_map, FSA 





