import torch
import torch.nn as nn
import scipy.io as io
import random

from ultrasound.train.unet.unet_parts import DoubleConv
from ultrasound.train.unet.unet import UNet

from ultrasound.train.config import *


class _ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1), padding_mode="reflect")
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1), padding_mode="reflect")
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1), padding_mode="reflect")
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1), padding_mode="reflect")
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out


class _ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out

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

     
        
        self.dropout = nn.Dropout(dropout)
        self.fn_out = nn.Sequential(
            nn.Linear(SIG_OUT_SIZE, EMB_SIZE))
        
        # self.sig_out = nn.Sequential(
        #     nn.Conv1d(NUM_TRANSMIT, NUM_TRANSMIT, kernel_size=1, stride=1, bias=False),
        #     nn.GELU(),
        #     nn.Conv1d(NUM_TRANSMIT, NUM_TRANSMIT, kernel_size=1, stride=1, bias=False))







    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[-1])
        # print("x.shape", x.shape)
        # for layer in self.time_convs:
        #     x = layer(x)
        #     print("x.shape", x.shape)

        # if self.training and random.random() <= 0.05:
        #     x = x + torch.normal(
        #         mean=torch.tensor(0.0),
        #         std=torch.tensor(0.005),
        #         size=x.shape).to(device)
        
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


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, scale_factor):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * (2 ** 2), 1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)



class DoubleConvResidual(nn.Module):
    def __init__(self, in_ch, out_ch, use_act_out=True):
        super().__init__()
        self.use_act_out = use_act_out
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True))

        self.double_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_ch)
        )
        
        
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.down_conv(x)
        x_1 = self.double_conv(x)
        return self.act(x + x_1 * 0.2)

# class SimpleResidualNet(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.down_conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True))

#         self.conv_1 = DoubleConvResidual(out_ch, out_ch)
#         self.conv_2 = DoubleConvResidual(out_ch, out_ch)
#         self.conv_3 = DoubleConvResidual(out_ch, out_ch, use_act_out=False)
#         self.act = nn.LeakyReLU(0.2, inplace=True)


#     def forward(self, x):
#         x = self.down_conv(x)
#         x_1 = self.conv_1(x)
#         x_1 = self.conv_2(x_1)
#         x_1 = self.conv_3(x_1)

#         return self.act(x + x_1 * 0.2)

        
# class SimpleResidualNet(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.down_conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True))

#         self.conv_1 = DoubleConvResidual(out_ch, out_ch)
#         self.act = nn.LeakyReLU(0.2, inplace=True)


#     def forward(self, x):
#         x = self.down_conv(x)
#         x_1 = self.conv_1(x)

#         return self.act(x + x_1 * 0.2)

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

        # else:
        #     self.up = nn.Sequential(
        #         nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #         nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        #         nn.LeakyReLU(0.2, inplace=True))


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
        

        # Make the time dimension the channels
        # x = x.permute((0, 2, 1))

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

        return x.view(x.shape[0], x.shape[2], x.shape[3])
        


