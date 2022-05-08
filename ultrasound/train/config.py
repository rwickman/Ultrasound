import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_LENGTH = 18650
KERNEL_SIZE = 16
STRIDE = 8
time_channels = [128, 64, 32, 16, 8, 1]

PADDING = (KERNEL_SIZE - SIGNAL_LENGTH % KERNEL_SIZE) // 2

"""Data settings."""
data_inp_dir = "/media/data/datasets/Ultrasound/AbSlices_npy/"
SOS_MAP_mat = "/media/data/datasets/Ultrasound/experiment_01/SOSMapAbOnly.mat"

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

num_exs = 62
test_split_pct = 0.5

# Number of examples in testing and validation set
test_size = int(test_split_pct * num_exs)


NUM_TRANSMIT = 64

"""Training hyperparameters."""
batch_size = 4
epochs = 400
save_dir = "model_temp/"
lr = 3e-4
load=True
save_iter = 10
dropout = 0.05
disc_lam = 0.025
recon_lam = 10
# Number of previous batches to store. Since loading takes awhile, this enables retraining on previous loaded batches
batch_buffer_size = 14

use_adversarial_loss = False
disc_lr = 5e-5
"""Transformer encoder hyperparameters."""
SIG_OUT_SIZE =  1166#SIGNAL_LENGTH // KERNEL_SIZE + 1 if PADDING > 0 else 0
EMB_SIZE = 512
NUM_HEADS =  8
DFF = 1024
NUM_ENC_LAYERS = 3

"""Decoder hyperparameters."""
IMG_OUT_SIZE = (300, 365)


