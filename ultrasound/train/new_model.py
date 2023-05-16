import torch
import torch.nn as nn
import scipy.io as io
import random
import numpy as np

from ultrasound.train.unet.unet_parts import DoubleConv
from ultrasound.train.unet.unet import UNet

from ultrasound.train.config import *


class FullyConnected(nn.Module):
    """Fully connected layer from IQ to image data."""

    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.isz = input_size
        self.osz1 = [n // 5 for n in output_size]
        self.osz = output_size

        ni = input_size[0] * input_size[1]
        nf = input_size[2]
        n1 = 16
        n2 = 256
        no = np.prod(self.osz1)
        self.conv1 = nn.Conv1d(ni, n1, 1)  # [ni, nf] -> [n1, nf]
        self.conv2 = nn.Conv1d(n1, n1, 1)  # [n1, nf] -> [n1, nf]
        self.conv3 = nn.Conv1d(nf, n2, n1)  # [nf, n1] -> [n2, 1]
        self.conv4 = nn.Conv1d(n2, no, 1)  # [n2, 1] -> [no, 1]
        self.bn1 = torch.nn.BatchNorm1d(n1)
        self.bn2 = torch.nn.BatchNorm1d(n1)
        self.bn3 = torch.nn.BatchNorm1d(n2)
        self.act_out = nn.Sigmoid()

    def forward(self, x):

        x = torch.reshape(x, (-1, self.isz[0] * self.isz[1], self.isz[2]))
        x = self.conv1(x)  # FC in channels
        x = torch.nn.ReLU()(x)
        x = self.bn1(x)

        # x = self.conv2(x)  # FC in freqs
        # x = torch.nn.ReLU()(x)
        # x = self.bn2(x)

        x = torch.transpose(x, 1, 2)
        x = self.conv3(x)  # FC in channels
        x = torch.nn.ReLU()(x)
        x = self.bn3(x)

        x = self.conv4(x)  # FC in channels

        x = torch.reshape(x, (-1, 1, *self.osz1))

        x = nn.functional.interpolate(x, self.osz, mode="bicubic", align_corners=True)

        x = torch.reshape(x, (-1, *self.osz))
        x = self.act_out(x)

        return x

        # # x = torch.transpose(x, 1, 2)
        # # x = self.conv4(x)
        # # x = torch.nn.ReLU()(x)
        # # x = self.bn4(x)
        # # x = self.conv5(x)

        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = self.fc2(x)

        # x = torch.reshape(x, (-1, *self.osz1))
        # x = self.conv4(x)
        # x = torch.nn.ReLU()(x)
        # x = self.bn4(x)
        # x = self.conv5(x)
        # x = torch.nn.ReLU()(x)
        # x = self.bn5(x)
        # x = self.conv6(x)
        # x = torch.reshape(x, (-1, *self.osz))
        # x = self.act_out(x)

        # return x

        # ksz = 11
        # stride = 2
        # n1 = 256
        # n2 = 64
        # n3 = 16
        # self.isz = input_size
        # self.osz = output_size
        # self.conv1 = nn.Conv1d(input_size[0] * input_size[1], n1, ksz, stride=stride)
        # self.conv2 = nn.Conv1d(n1, n2, ksz, stride=stride)
        # self.conv3 = nn.Conv1d(n2, n3, ksz, stride=stride)
        # self.bn1 = torch.nn.BatchNorm1d(n1)
        # self.bn2 = torch.nn.BatchNorm1d(n2)
        # self.bn3 = torch.nn.BatchNorm1d(n3)
        # self.bn4 = torch.nn.BatchNorm2d(8)
        # self.bn5 = torch.nn.BatchNorm2d(8)

        # newdim = lambda x: np.floor((x - (ksz - 1) - 1) / stride + 1).astype("int32")
        # ni = n3 * newdim(newdim(newdim(input_size[2])))
        # self.osz1 = [8, *[n // 5 for n in output_size]]
        # no = np.prod(self.osz1)
        # self.fc1 = torch.nn.Linear(ni, hidden_size)
        # self.fc2 = torch.nn.Linear(hidden_size, no)

        # self.conv4 = nn.ConvTranspose2d(8, 8, 3, 5, 0, 0)
        # self.conv5 = nn.Conv2d(8, 8, 3, 1, 2)
        # self.conv6 = nn.Conv2d(8, 1, 3, 1, 1)

        # # ni = newdim(newdim(newdim(input_size[2])))
        # # no = np.prod(output_size)
        # # self.conv4 = nn.Conv1d(ni, hidden_size, n3)
        # # self.conv5 = nn.Conv1d(hidden_size, no, 1)
        # self.act_out = nn.Sigmoid()

        # ksz = 5
        # stride = 2
        # super().__init__()
        # self.isz = input_size
        # self.osz = output_size
        # self.conv1 = torch.nn.Conv2d(input_size[0], nfilt, ksz, stride=stride)
        # self.bn1 = torch.nn.BatchNorm2d(nfilt)
        # self.conv2 = torch.nn.Conv2d(nfilt, nfilt, ksz, stride=stride)
        # self.bn2 = torch.nn.BatchNorm2d(nfilt)
        # self.conv3 = torch.nn.Conv2d(nfilt, nfilt, ksz, stride=stride)
        # self.bn3 = torch.nn.BatchNorm2d(nfilt)
        # newdim = lambda x: np.floor((x - (ksz - 1) - 1) / stride + 1)

        # hi = newdim(newdim(newdim(input_size[1]))).astype("int32")
        # wi = newdim(newdim(newdim(input_size[2]))).astype("int32")
        # ni = nfilt * hi * wi
        # no = np.prod(output_size)
        # self.fc1 = torch.nn.Linear(ni, hidden_size)
        # self.fc2 = torch.nn.Linear(hidden_size, no)
        # self.act_out = nn.Sigmoid()
