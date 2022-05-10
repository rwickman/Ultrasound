import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

sns.set(style="darkgrid", font_scale=1.5)

model_dir = "model_3/train_dict.json"

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
print(train_dict["train_loss"])
print(train_dict["val_loss"])

axs[1].plot( moving_average(train_dict["val_loss"]))
axs[1].set(
    title="Validation Set",
    xlabel="Epoch",
    ylabel="MSE Loss"
)

# axs[2].plot( moving_average(train_dict["adv_loss"]))
# axs[2].set(
#     title="Adversarial Loss ",
#     xlabel="Epoch",
#     ylabel="ADV Loss"
# )


# axs[3].plot( moving_average(train_dict["disc_loss"]))
# axs[3].set(
#     title="Discriminator Loss ",
#     xlabel="Epoch",
#     ylabel="Disc Loss"
# )

plt.show()