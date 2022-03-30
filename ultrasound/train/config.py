import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_LENGTH = 18650
KERNEL_SIZE = 16
STRIDE = 8
time_channels = [128, 64, 32, 16, 8, 1]

PADDING = (KERNEL_SIZE - SIGNAL_LENGTH % KERNEL_SIZE) // 2

"""Data settings."""
data_inp_dir = "/media/data/datasets/Ultrasound/experiment_01/OutputRepacked/FSA_Layer03"
SOS_MAP = "/media/data/datasets/Ultrasound/experiment_01/SOSMapAbOnly.mat"
num_exs = 62
test_split_pct = 0.1

# Number of examples in testing and validation set
test_size = int(test_split_pct * num_exs)


NUM_TRANSMIT = 64

"""Transformer encoder hyperparameters."""
SIG_OUT_SIZE =  1166#SIGNAL_LENGTH // KERNEL_SIZE + 1 if PADDING > 0 else 0
EMB_SIZE = 512
NUM_HEADS =  8
DFF = 1024
NUM_ENC_LAYERS = 3

"""Decoder hyperparameters."""
IMG_OUT_SIZE = (300, 365)

