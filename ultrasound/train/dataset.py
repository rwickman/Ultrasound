import torch
from torch.utils.data import Dataset
import scipy.io as io
import os
import numpy as np
import time
from config import *

# def find_nearest(SOS_val):
#     idx = np.abs(SOS_val - unique_SOS).argmin()
#     print(SOS_val, unique_SOS[idx])
#     return unique_SOS[idx]

# def convert_to_class(SOS_map):
#     SOS_map_converted = np.zeros_like(SOS_map)
#     for i in range(SOS_map.shape[0]):
#         for j in range(SOS_map.shape[1]):
#             for k in range(SOS_map.shape[2]):
#                 SOS_map_converted[i, j, k] = find_nearest(SOS_map[i,j,k])
#     return SOS_map_converted
    

def load_aug_data():
    
    # Load the SOS maps
    SOS_maps = []
    # density_maps = []
    for aug_SOS_mat in aug_SOS_mats:
        SOS_maps.append(io.loadmat(aug_SOS_mat)["SOSMap"])
    
    SOS_maps = np.concatenate(SOS_maps, axis=-1)

    SOS_maps = SOS_maps.transpose((2, 0, 1))

    SOS_maps = (SOS_maps - SOS_maps.min()) / (SOS_maps.max() - SOS_maps.min())
    
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

    return SOS_maps, FSA_mats_inp

def load_data():
    SOS_map = io.loadmat(SOS_MAP_mat)["SOSMap"].astype(np.float32)

    SOS_map = SOS_map.transpose((2,0,1))

    # Normalize
    SOS_map = (SOS_map - SOS_map.min()) / (SOS_map.max() - SOS_map.min())

    SOS_val = SOS_map[:test_size]
    SOS_test = SOS_map[test_size:]

    # Get the FSA mat files
    # FSA_mats = [
    #     os.path.join(data_inp_dir, FSA_mat) 
    #     for FSA_mat in os.listdir(data_inp_dir) 
    #     if ".mat" in FSA_mat]

    # Get files on the system
    FSA_mats = [
        os.path.join(data_inp_dir, FSA_mat) 
        for FSA_mat in os.listdir(data_inp_dir) 
        if ".npy" in FSA_mat]    
    # Sort them
    # FSA_mats = sorted(FSA_mats, 
    #     key=lambda FSA_mat: int(FSA_mat.split("/")[-1].split("Layer")[-1].split(".mat")[0]))
    
    # Sort the files
    FSA_mats = sorted(FSA_mats, 
        key=lambda FSA_mat: int(FSA_mat.split("/")[-1].split("FSA_Layer")[-1].split(".npy")[0]))
    
    # FSA_mats_train = FSA_mats[:-test_size*2]
    
    # Split the FSA into validation and test 
    FSA_mats_val = FSA_mats[:test_size]
    FSA_mats_test = FSA_mats[test_size:]

    return (SOS_val, SOS_test), (FSA_mats_val, FSA_mats_test)


def create_datasets(only_test=False):
    SOS_maps, FSA_mats = load_data()
    if not only_test:
        SOS_maps_train, FSA_mats_train = load_aug_data()

        SOS_maps_train = np.concatenate((SOS_maps_train, SOS_maps[0][:6], SOS_maps[1][-6:]))
        FSA_mats_train = FSA_mats_train + FSA_mats[0][:6] + FSA_mats[1][-6:]

        train_dataset = UltrasoundDataset(SOS_maps_train, FSA_mats_train)
        val_dataset = UltrasoundDataset(SOS_maps[0][6:], FSA_mats[0][6:])
        test_dataset = UltrasoundDataset(SOS_maps[1][:-6], FSA_mats[1][:-6])

        return train_dataset, val_dataset, test_dataset
    else:
        test_dataset = UltrasoundDataset(SOS_maps[1][:-6], FSA_mats[1][:-6])
        return test_dataset



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
        #print(self.FSA_mats[idx])
        FSA = FSA / FSA_scale_fac

        if FSA.shape[-1] < SIGNAL_LENGTH:
            FSA_pad = torch.zeros(NUM_TRANSMIT, NUM_TRANSMIT, SIGNAL_LENGTH, dtype=torch.float32)
            FSA_pad[:, :, :FSA.shape[-1]] = FSA
            FSA = FSA_pad

        return SOS_map, FSA 





