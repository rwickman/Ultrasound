import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from ultrasound.train.config import *
sns.set(style="darkgrid", font_scale=1.5)
import os

model_dir = os.path.join(save_dir, "train_dict.json") 

with open(model_dir) as f:
    train_dict = json.load(f)

"""Plot the training results."""
def moving_average(x, w=2):
    return np.convolve(x, np.ones(w), 'valid') / w

fig, axs = plt.subplots(2)
axs[0].plot( moving_average(train_dict["train_loss"]))
axs[0].set(
    title="Training Set",
    xlabel="Epoch",
    ylabel="MSE Loss"
)
print(train_dict["train_loss"], len(train_dict["train_loss"]))
print(train_dict["val_loss"])


axs[1].plot( moving_average(train_dict["val_loss"]))
axs[1].set(
    title="Validation Set",
    xlabel="Epoch",
    ylabel="MSE Loss"
)

plt.show()