import numpy as np
import scipy.io as io
import os

cur_dir = [f for f in os.listdir(".") if "Output_Aug_Batch" in f or "AbSlices" in f]

for batch_dir in cur_dir:
    save_dir = batch_dir + "_npy"
    os.mkdir(save_dir)
    for batch_file in os.listdir(batch_dir):
        FSA = io.loadmat(os.path.join(batch_dir, batch_file))["FSA"]
        batch_file_npy = batch_file.split(".mat")[0] + ".npy"
        print("SAVING", os.path.join(save_dir, batch_file_npy))
        np.save(os.path.join(save_dir, batch_file_npy), FSA)
