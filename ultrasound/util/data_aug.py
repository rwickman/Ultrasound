from PIL import Image
import numpy as np
from ultrasound.train.config import IMG_OUT_SIZE
import matplotlib.pyplot as plt
from labels import *
import random, os
import scipy.io as io
from scipy.ndimage import gaussian_filter

aug_imgs_dir = "/media/data/datasets/Ultrasound/aug_inp"
#num_bins = 30
PAD_LENGTH = 20
noise_pct = 0.05
batch_size = 100
sos_vals = list(label2SOS.values())
density_vals = list(label2Density.values())
# bin_vals = list(range(0, 255 + num_bins, num_bins))


def sample_SOS_and_density():
    rand_idx = random.randint(0, len(sos_vals) - 1)
    sos_noise_bound = sos_vals[rand_idx] * noise_pct
    density_noise_bound = density_vals[rand_idx] * noise_pct

    sos_rand_noise = round(random.uniform(-sos_noise_bound, sos_noise_bound))
    density_rand_noise = round(random.uniform(-density_noise_bound, density_noise_bound))

    sos_val = sos_vals[rand_idx] + sos_rand_noise
    density_val = density_vals[rand_idx] + density_rand_noise
    
    sos_val = np.clip(sos_val, SOS_RANGE[0], SOS_RANGE[1])
    density_val = np.clip(density_val, DENSITY_RANGE[0], DENSITY_RANGE[1])

    return sos_val, density_val

def bin_img(img):
    # Select a random number of bins
    num_bins = random.randint(20, 40)
    bin_vals = list(range(0, 255 + num_bins, num_bins))

    img = gaussian_filter(img, sigma=random.uniform(1.5, 3.5))
    for i in range(len(bin_vals)-1):
        img[np.logical_and(img >= bin_vals[i], img < bin_vals[i+1])] = bin_vals[i]
    
    return img

def rand_pad(img_SOS, img_density):
    rand_pads = []
    for i in range(4):
        rand_pads.append(np.random.randint(1, PAD_LENGTH))


    img_SOS = np.pad(img_SOS, ((rand_pads[0], rand_pads[1]), (rand_pads[2], rand_pads[3])), constant_values=label2SOS["airway"])

    img_density = np.pad(img_density, ((rand_pads[0], rand_pads[1]), (rand_pads[2], rand_pads[3])), constant_values=label2Density["airway"])
    return img_SOS, img_density



print("IMG_OUT_SIZE", IMG_OUT_SIZE)

img_files = os.listdir("/media/data/datasets/unlabeled2017/")
img_files = random.sample(img_files, 1000)

img_files = [os.path.join("/media/data/datasets/unlabeled2017/", i) for i in img_files]
binary_mask = io.loadmat("/media/data/datasets/Ultrasound/binaryMask.mat")["binaryMask"] 


def gen_rand(img_files, img_num):
    img_SOSs = np.zeros((IMG_OUT_SIZE[0], IMG_OUT_SIZE[1], len(img_files)))
    img_densities = np.zeros((IMG_OUT_SIZE[0], IMG_OUT_SIZE[1], len(img_files)))

    for img_idx, img_file in enumerate(img_files):
        img = Image.open(
            img_file).convert("L").resize((IMG_OUT_SIZE[1], IMG_OUT_SIZE[0]))


        img = np.array(img)
        img = bin_img(img)

        # print("img.shape", img.shape)
        # plt.imshow(img)
        # plt.show()

        unique_vals = np.unique(img)
        sampled_sos_vals = []
        sampled_density_vals = []
        for val in unique_vals:
            sos_val, density_val = sample_SOS_and_density()
            sampled_sos_vals.append(sos_val)
            sampled_density_vals.append(density_val)
            


        # Force the min and max SOS values to be in the map
        set_min = False
        if SOS_RANGE[0] not in sampled_sos_vals:
            rand_idx_min = random.randint(0, len(sampled_sos_vals) - 1)
            sampled_sos_vals[rand_idx_min] = SOS_RANGE[0]
            sampled_density_vals[rand_idx_min] = DENSITY_RANGE[0]


        if SOS_RANGE[1] not in sampled_sos_vals:    
            # Indices not already sampled
            if set_min:
                rand_idxs = list(range(0, rand_idx_min)) + list(range(rand_idx_min + 1, len(sampled_sos_vals)))
            else:
                rand_idxs = list(range(0, len(sampled_sos_vals)))

            rand_idx_max = random.choice(rand_idxs)
            sampled_sos_vals[rand_idx_max] = SOS_RANGE[1]
            sampled_density_vals[rand_idx_max] = DENSITY_RANGE[1]

        img_SOS = np.zeros_like(img, dtype=float)
        img_density = np.zeros_like(img, dtype=float)
        for i, val in enumerate(unique_vals):
            img_SOS[img == val] = sampled_sos_vals[i]
            img_density[img == val] = sampled_density_vals[i]
        
        img_SOS[binary_mask == 0] = 1540
        img_density[binary_mask == 0] = 993

        img_SOSs[:, :, img_idx] = img_SOS
        img_densities[:, :, img_idx] = img_density
        print("img_idx", img_idx)


    d = {
        "SOSMap" : img_SOSs,
        "DensityMap" : img_densities
    }

    # img_SOS, img_density = rand_pad(img_SOS, img_density)
    #img_SOS_normalized = (img_SOS - img_SOS.min()) / (img_SOS.max() - img_SOS.min())

    # plt.imshow(img_SOS_normalized)
    # plt.show()
    print(os.path.join(aug_imgs_dir, f"aug_sos_{img_num}.mat"))
    io.savemat(os.path.join(aug_imgs_dir, f"aug_sos_{img_num}.mat"), d)

for i in range(len(img_files)//batch_size):
    print("i", i)
    gen_rand(img_files[i*batch_size:(i+1)*batch_size], i)

