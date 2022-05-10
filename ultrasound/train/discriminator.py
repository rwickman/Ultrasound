import torch
import torch.nn as nn
import torch.nn.functional as F

def discriminator_loss(output, label):
    return F.binary_cross_entropy(output, label)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self._disc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(128, 1, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(120, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self._disc(x).view(-1)