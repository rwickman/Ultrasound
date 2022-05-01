import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_LENGTH = 18650
KERNEL_SIZE = 16
STRIDE = 8
time_channels = [128, 64, 32, 16, 8, 1]

PADDING = (KERNEL_SIZE - SIGNAL_LENGTH % KERNEL_SIZE) // 2

"""Data settings."""
data_inp_dir = "/media/data/datasets/Ultrasound/OutputRepacked/"
SOS_MAP_mat = "/media/data/datasets/Ultrasound/experiment_01/SOSMapAbOnly.mat"

aug_SOS_mats = [
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_0.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_1.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_2.mat",
    "/media/data/datasets/Ultrasound/aug_inp/aug_sos_3.mat"
]


aug_FSA_inp_dirs = [
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch1",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch2",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch3",
    "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch4",
]

num_exs = 62
test_split_pct = 0.5

# Number of examples in testing and validation set
test_size = int(test_split_pct * num_exs)


NUM_TRANSMIT = 64

"""Training hyperparameters."""
batch_size = 8
epochs = 400
save_dir = "model/"
lr = 3e-4
load=True
save_iter = 5

"""Transformer encoder hyperparameters."""
SIG_OUT_SIZE =  1166#SIGNAL_LENGTH // KERNEL_SIZE + 1 if PADDING > 0 else 0
EMB_SIZE = 512
NUM_HEADS =  8
DFF = 1024
NUM_ENC_LAYERS = 3

"""Decoder hyperparameters."""
IMG_OUT_SIZE = (300, 365)
