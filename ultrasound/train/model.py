import torch
import torch.nn as nn
import scipy.io as io
import random

from ultrasound.train.unet.unet_parts import DoubleConv
from ultrasound.train.unet.unet import UNet

from ultrasound.train.config import *

class SignalReduce(nn.Module):
    """Reduce the input size."""
    def __init__(self):
        super().__init__()

        # Convolutions for reducing time dimension
        time_convs = []
        for i in range(len(kernel_sizes)):
            time_convs.append(nn.Conv1d(time_channels[i], time_channels[i+1], kernel_size=kernel_sizes[i], stride=stride[i], padding=padding[i], bias=False))
            time_convs.append(nn.BatchNorm1d(time_channels[i+1]))
            time_convs.append(nn.GELU())

        # Convolution for reducing all received for a single transmit into a single vector
        self.time_convs = nn.Sequential(*time_convs)
        
        self.fn_out = nn.Sequential(
            nn.Linear(SIG_OUT_SIZE, EMB_SIZE))

    def forward(self, x):
        batch_size = x.shape[0]

        # Reshape to make all transmit as part of batches
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[-1])

        
        # Run through time convolutions to "collapse" the receive dimension
        x = self.time_convs(x)

        # Reshape into batches
        x = x.view(batch_size, NUM_TRANSMIT, x.shape[-1])    
        x = self.fn_out(x)

        return x

class SignalAtt(nn.Module):
    """Perform MHA attention over the transmit embeddings.""" 
    def __init__(self):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            EMB_SIZE,
            nhead=NUM_HEADS,
            dim_feedforward=DFF,
            batch_first=True)
        self.pos_embs = nn.Parameter(torch.randn(1, NUM_TRANSMIT, EMB_SIZE))
        self.enc = nn.TransformerEncoder(
            enc_layer,
            num_layers=NUM_ENC_LAYERS)

    def forward(self, x):
        x = x + self.pos_embs
        x = self.enc(x)

        return x

class UpsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch, out_size=None, scale_factor=2, residual_layers=True):
        super().__init__()
        #if residual_layers:
        if out_size is None:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DoubleConv(in_ch, out_ch),
                )
        else:
            self.up = nn.Sequential(
                nn.Upsample(out_size, mode='bilinear', align_corners=True),
                DoubleConv(in_ch, out_ch),
                )

        self.dropout = nn.Dropout2d(dropout)


    
    def forward(self, x):
        x = self.up(x)
        #x = self.dropout(x)
        return x
        

class SignalDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up_layers = nn.Sequential(
            nn.Conv2d(EMB_SIZE//16, EMB_SIZE//16, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            UpsampleLayer(EMB_SIZE//16, EMB_SIZE//32),
            UpsampleLayer(EMB_SIZE//32, EMB_SIZE//32),
            UpsampleLayer(EMB_SIZE//32, EMB_SIZE//32, out_size=IMG_OUT_SIZE))

        self.unet = UNet(EMB_SIZE//32, 1)
        
        
    def forward(self, x, aug=None):
        batch_size = x.shape[0]
        
        # Split the signals into the width and height dimension
        img_dim = int(x.shape[2] ** 0.5)
        if aug:
            x = aug(x)

        x = x.view(batch_size, x.shape[1], img_dim, img_dim)
        
        # Upsample it to be batch_size x 16 x 300 x 365
        x = self.up_layers(x)

        # # Run through reconstruction UNet Model to produce batch_size x x 300 x 365        
        x = self.unet(x)
        
        return x
        

class DensityModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.sig_reduce = SignalReduce()
        self.sig_att = SignalAtt()
        self.sig_dec = SignalDecoder()
        self.act_out = nn.Sigmoid()


    def forward(self, x, aug=None):
        x = self.sig_reduce(x)
        x = self.sig_att(x)

        x = self.sig_dec(x, aug)

        x = self.act_out(x)
        print("x.shape", x.shape)

        return x.view(x.shape[0], x.shape[2], x.shape[3])
        


