import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_LENGTH = 18650

KERNEL_SIZE = 16
STRIDE = 8
time_channels = [128, 64, 32, 16, 8, 1]

PADDING = (KERNEL_SIZE - SIGNAL_LENGTH % KERNEL_SIZE) // 2

"""Data settings."""
# Location of npy slices
data_inp_dir = "/media/data/datasets/Ultrasound/AbSlices_npy/"
# Location of SOS mat 
SOS_MAP_mat = "/media/data/datasets/Ultrasound/experiment_01/SOSMapAbOnly.mat"

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

# Number of slices
num_exs = 62
test_split_pct = 0.5

# Number of examples in testing and validation set
test_size = int(test_split_pct * num_exs)

NUM_TRANSMIT = 64

# Scaling factor for input signals
FSA_scale_fac = 60

"""Training hyperparameters."""
# Where the model should be trained
save_dir="model_3/"

# Loading the model or not
load=True

# How often to run over validation set and save a copy of the model
save_iter = 10

# Learning rates other potential values [1e-5, 2e-5, ..., 1e-4, 2e-4]
lr = 1e-4

# Number of images to make prediction on at a time
batch_size = 4

# Number of previous batches to store. Since loading takes awhile, this enables retraining on previous loaded batches
batch_buffer_size = 8
epochs = 4000

# Number of workers loading data: approx.range of [1, 4]
num_workers = 2
# Number of data points: approx. range of [1, 8]
prefetch_factor=6

dropout = 0.05
recon_lam = 10



# use_classes = False
# unique_SOS = [1478, 1501, 1534, 1540, 1547, 1572, 1613, 1665, 2425]



"""GAN hyperparameters."""
use_adversarial_loss = False
disc_lr = 5e-5
disc_lam = 0.05

"""Transformer encoder hyperparameters."""
SIG_OUT_SIZE =  1166#SIGNAL_LENGTH // KERNEL_SIZE + 1 if PADDING > 0 else 0
EMB_SIZE = 512
NUM_HEADS =  8
DFF = 1024
NUM_ENC_LAYERS = 3


"""Decoder hyperparameters."""
IMG_OUT_SIZE = (300, 365)


