import time
import numpy as np

FSA = [f"/FSA_Layer{i}.npy" for i in range(10, 20)]


FSA_npz = [i + ".npz" for i in FSA]
npy_dir = "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch9_npy"
npz_dir = "/media/data/datasets/Ultrasound/aug_out/Output_Aug_Batch10_npy"

start_time = time.time()

for i in FSA:
    x = np.load(npy_dir + i)

print(".npy load time:", time.time() - start_time)



start_time = time.time()

for i in FSA_npz:
    x = np.load(npz_dir + i)
    for i in range(64):
        for j in range(64):
            y = x[f"{i}_{j}"]


print(".npz load time:", time.time() - start_time)