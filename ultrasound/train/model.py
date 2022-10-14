import torch
import torch.nn as nn
import scipy.io as io

from config import *

class SignalReduce(nn.Module):
    """Reduce the input size."""
    def __init__(self):
        super().__init__()

        # Convolutions for reducing time dimension
        time_convs = []
        for i in range(1, len(time_channels)):
            time_convs.append(nn.Conv1d(time_channels[i-1], time_channels[i], kernel_size=KERNEL_SIZE, stride=STRIDE, padding=0))
            time_convs.append(nn.BatchNorm1d(time_channels[i], momentum=0.01))
            time_convs.append(nn.ReLU())

        self.time_convs = nn.Sequential(*time_convs)

        # Convolution for reducing all received for a single transmit into a single vector
        self.dropout = nn.Dropout(dropout)
        self.fn_out = nn.Linear(SIG_OUT_SIZE, EMB_SIZE)


    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[-1])
        # print("x.shape", x.shape)
        # for l in self.time_convs:
        #     x = l(x)
        #     print("x.shape", x.shape)
        x = self.time_convs(x)
        # Reshape into batches
        x = x.view(batch_size, NUM_TRANSMIT, x.shape[-1])
        x = self.dropout(x)
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
            num_layers=NUM_ENC_LAYERS,)
        # self.layer_norm = nn.LayerNorm(EMB_SIZE)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x + self.pos_embs
        x = self.enc(x)
        x = self.dropout(x)

        return x



class UpsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch, out_size=None):
        super().__init__()
        
        if out_size is None:
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = nn.Upsample(out_size)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.dropout(x)

        return x
        

class SignalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_layer_1 = UpsampleLayer(EMB_SIZE, EMB_SIZE//2)
        self.up_layer_2 = UpsampleLayer(EMB_SIZE//2, EMB_SIZE//4)
        self.up_layer_3 = UpsampleLayer(EMB_SIZE//4, EMB_SIZE//8)
        self.up_layer_4 = UpsampleLayer(EMB_SIZE//8, EMB_SIZE//16)
        self.up_layer_5 = UpsampleLayer(EMB_SIZE//16, EMB_SIZE//32)
        self.up_layer_6 = UpsampleLayer(EMB_SIZE//32, EMB_SIZE//64, out_size=IMG_OUT_SIZE)
        
        self.conv_out = nn.Conv2d(EMB_SIZE//64, 1, kernel_size=1)
        
        
        
    def forward(self, x):
        batch_size = x.shape[0]
        # Make the time dimension the channels
        x = x.permute((0, 2, 1))

        # Split the signals into the width and height dimension
        img_dim = int(x.shape[2] ** 0.5)
        x = x.view(batch_size, x.shape[1], img_dim, img_dim)
        x = self.up_layer_1(x)
        x = self.up_layer_2(x)
        x = self.up_layer_3(x)
        x = self.up_layer_4(x)
        x = self.up_layer_5(x)
        x = self.up_layer_6(x)

        x = self.conv_out(x)
        return x
        

class DensityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig_reduce = SignalReduce()
        self.sig_att = SignalAtt()
        self.sig_dec = SignalDecoder()
        self.act_out = nn.Sigmoid()


    def forward(self, x):
        x = self.sig_reduce(x)
        x = self.sig_att(x)
        x = self.sig_dec(x)
        x = self.act_out(x)
        return x.view(x.shape[0], x.shape[2], x.shape[3])
        


