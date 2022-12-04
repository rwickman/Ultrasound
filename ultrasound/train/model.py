import torch
import torch.nn as nn
import scipy.io as io

from unet.unet_parts import DoubleConv
from unet.att_unet import AttU_Net
from unet.unet import UNet

from config import *


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
            print(i)
            time_convs.append(nn.Conv1d(time_channels[i], time_channels[i+1], kernel_size=kernel_sizes[i], stride=stride[i], padding=padding[i], bias=False))
            time_convs.append(nn.BatchNorm1d(time_channels[i+1]))
            #time_convs.append(nn.LayerNorm((time_channels[i], time_out_sizes[i+1])))
            time_convs.append(nn.GELU())
            # time_convs.append(nn.Conv1d(time_channels[i], time_channels[i], kernel_size=KERNEL_SIZE, stride=1, padding=1))
            # time_convs.append(nn.LayerNorm((time_channels[i], time_out_sizes[i-1]), elementwise_affine=False))
            # time_convs.append(nn.GELU())

        self.time_convs = nn.Sequential(*time_convs)

        # Convolution for reducing all received for a single transmit into a single vector
        
        #self.dropout = nn.Dropout(dropout)
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
        x = self.time_convs(x)
        
        # Reshape into batches
        x = x.view(batch_size, NUM_TRANSMIT, x.shape[-1])

        x = self.fn_out(x)
        # x = self.sig_out(x)

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
        # self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        batch_size = x.shape[0]
        # org_x = x
        x = x + self.pos_embs
        x = self.enc(x)
        # x = x * 0.2 + org_x

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



class UpsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch, out_size=None, scale_factor=2):
        super().__init__()
        
        if out_size is None:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DoubleConv(in_ch, out_ch, in_ch // 2))
        else:
            self.up = nn.Sequential(
                nn.Upsample(out_size, mode='bilinear', align_corners=True),
                DoubleConv(in_ch, out_ch, in_ch // 2))
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect"),
        #     #nn.LayerNorm(up_layer_norm_dict[out_ch], elementwise_affine=False),
        #     nn.BatchNorm2d(out_ch),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        self.dropout = nn.Dropout2d(dropout)


    
    def forward(self, x):
        # print("x.shape", x.shape)
        x = self.up(x)
        #print("x.shape", x.shape)

        
        # x = self.conv(x)
        x = self.dropout(x)


        return x
        

class SignalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_layers = nn.Sequential(
            UpsampleLayer(EMB_SIZE, EMB_SIZE//2),
            UpsampleLayer(EMB_SIZE//2, EMB_SIZE//4), 
            UpsampleLayer(EMB_SIZE//4, EMB_SIZE//8),
            UpsampleLayer(EMB_SIZE//8, EMB_SIZE//16),
            UpsampleLayer(EMB_SIZE//16, EMB_SIZE//32),
            UpsampleLayer(EMB_SIZE//32, EMB_SIZE//32, out_size=IMG_OUT_SIZE))

        # self.conv_out = nn.Sequential(
        #     nn.Conv2d(EMB_SIZE//64, EMB_SIZE//64, 3, padding=1, padding_mode="reflect"),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(EMB_SIZE//64, 1, kernel_size=1))
        self.unet = UNet(EMB_SIZE//32, 1)
        #self.conv_out = nn.Conv2d(EMB_SIZE//64, EMB_SIZE//64, 3, padding=1, padding_mode="reflect", bias=False)

        
        
    def forward(self, x, aug=None):
        batch_size = x.shape[0]

        # Make the time dimension the channels
        x = x.permute((0, 2, 1))

        # Split the signals into the width and height dimension
        img_dim = int(x.shape[2] ** 0.5)
        if aug:
            x = aug(x)

        x = x.view(batch_size, x.shape[1], img_dim, img_dim)
     
        # Upsample it to be 16 x 300 x 365
        x = self.up_layers(x)

        # Run through reconstruction Att UNet Model to produce 1 x 300 x 365 
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
        


