import scipy.io as io
import matplotlib.pyplot as plt
import torch
y = io.loadmat("/media/data/datasets/Ultrasound/aug_inp/aug_sos_0.mat")["SOSMap"]
y = (y - y.min()) / (y.max() - y.min())
y = torch.tensor(y)
y_pad = torch.zeros(360, 360)
y_pad[:] = y[0, 0, 0]
y_pad[30:y.shape[0]+30] = y[:, :360, 0]


fig, axs = plt.subplots(2)
print("y.shape", y.shape)
axs[0].imshow(y[:, :360, 0])
plt.show()