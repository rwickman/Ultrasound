import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

SOS_RANGE = [1478, 2425]
SIGNAL_LENGTH = 1244

kernel_sizes = [3, 3, 3, 3, 3]
stride = [1, 1, 1, 1, 1]
padding =[1, 1, 1, 1, 1]
time_channels = [128, 128, 128, 128, 128, 1]
time_out_sizes = [1244, 1244, 1244, 1244, 1244]


up_layer_norm_dict = {
    256: (256, 16, 16),
    128: (128, 32, 32),
    64: (64, 64, 64),
    32: (32, 128, 128),
    16: (16, 256, 256),
    8: (8, 300, 365),    
}

"""Data settings."""
# Location of npy slices
data_inp_dir = "/media/data/datasets/Ultrasound/new/Output_AbSlices/"
# Location of SOS mat 
SOS_MAP_mat = "/media/data/datasets/Ultrasound/new/GT_SOS_AbSlices.mat"

# Location of synthetic SOS mats
aug_SOS_mats = [
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_0.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_1.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_2.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_3.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_4.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_5.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_6.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_7.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_8.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_9.mat",
]

# Location of synthetic input .npy files
aug_FSA_inp_dirs = [
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch1_npy",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch2_npy",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch3_npy",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch4_npy",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch5_npy",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch6_npy",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch7_npy",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch8_npy",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch9_npy",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch10_npy",
]

binary_mask_file = "/media/data/datasets/Ultrasound/binaryMask.mat"
binary_mask_val = (1540 - 1478) / (2425 - 1478)

# Number of slices
num_exs = 62
test_split_pct = 0.5

# Number of examples in testing and validation set
test_size = int(test_split_pct * num_exs)

NUM_TRANSMIT = 64

# Scaling factor for input signals
FSA_scale_fac = 61
FSA_RANGE = [-60, 61]
SOS_RANGE = [1478, 2425]

"""Training hyperparameters."""
save_dir="models"

# Use the synthetic data set for training
use_synth_data = False

# Use only the synthetic data for training and not the original dataset
use_only_synth_data = False

# Loading the model or not
load = True

# Print out debuging info
debug=False

# How often to run over validation set and save a copy of the model
save_iter = 10

# Learning rates other potential values [1e-5, 2e-5, ..., 1e-4, 2e-4]
lr = 2e-4

weight_decay = 1e-3

# Number of images to make prediction on at a time
batch_size = 8

# Number of epochs to train
epochs = 500

# Number of workers loading data: approx.range of [1, 4]
num_workers = 2
# Number of data points: approx. range of [1, 8]
prefetch_factor=4

dropout = 0.05

warm_up_epochs = 10
warm_up_lr = 1e-7


# use_classes = False
# unique_SOS = [1478, 1501, 1534, 1540, 1547, 1572, 1613, 1665, 2425]



"""GAN hyperparameters."""
use_adversarial_loss = False
disc_lr = 1e-5
disc_lam = 0.005

"""Transformer encoder hyperparameters."""
SIG_OUT_SIZE = 1244
EMB_SIZE = 1024
NUM_HEADS =  8
DFF = 1024
NUM_ENC_LAYERS = 1

CHANNELS = 64
WIDTH = 32

"""Decoder hyperparameters."""
IMG_OUT_SIZE = (300, 365)
