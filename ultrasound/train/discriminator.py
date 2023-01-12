import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def discriminator_loss(output, label):
    return F.binary_cross_entropy_with_logits(output, label)

disc_norm_layers = {
    64: (64, 150, 182),
    128: (128, 75, 91),
    256: (256, 75, 91),

}


def loss_hinge_gen(dis_fake, weight_fake = None):
  loss = -torch.mean(dis_fake)
  return loss

def loss_hinge_dis(dis_fake, dis_real, weight_real = None, weight_fake = None):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # self._disc = nn.Sequential(
        #     nn.Conv2d(1, 32, 3, 2, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.1),

        #     nn.Conv2d(32, 64, 3, 2, 1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.1),

        #     nn.Conv2d(64, 128, 3, 2, 1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.1),

        #     nn.Conv2d(128, 128, 3, 2, 1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.1),

        #     nn.Conv2d(128, 256, 3, 2, 1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.1),

        #     nn.Conv2d(256, 1, 3, 2, 1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Flatten(),
        #     nn.Dropout(0.1),
        #     nn.Linear(150, 1)
        #     # nn.Sigmoid()
        # )
        norm = spectral_norm
        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            # state size. (64) x 64 x 64
            norm(nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False)),
            #nn.LayerNorm((64, 150, 182), elementwise_affine=False),
            nn.LeakyReLU(0.2, True),
            norm(nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False)),
            #nn.LayerNorm((128, 150, 182), elementwise_affine=False),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            # state size. (128) x 32 x 32
            norm(nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False)),
            #nn.LayerNorm((128, 75, 91), elementwise_affine=False),
            nn.LeakyReLU(0.2, True),
            norm(nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False)),
            #nn.LayerNorm((256, 75, 91), elementwise_affine=False),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            # state size. (256) x 16 x 16
            norm(nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False)),
            #nn.LayerNorm((256, 37, 45), elementwise_affine=False),
            nn.LeakyReLU(0.2, True),
            norm(nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False)),
            #nn.LayerNorm((512, 37, 45), elementwise_affine=False),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            # state size. (512) x 8 x 8
            norm(nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False)),
            #nn.LayerNorm((512, 18, 22), elementwise_affine=False),
            nn.LeakyReLU(0.2, True),
            norm(nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False)),
            #nn.LayerNorm((512, 9, 11), elementwise_affine=False),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            #nn.LayerNorm((512, 4, 5), elementwise_affine=False),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 5, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        # for l in self.features:
        #     print(l)
        #     x = l(x)
        #     print("x.shape", x.shape)
        out = torch.flatten(out, 1)
        out = self.classifier(out).view(-1)

        return out


class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)
    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.
    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        x4 = F.interpolate(x4, size=(75, 91), mode='bilinear', align_corners=False)

        if self.skip_connection:
            print("x4.shape", x4.shape)
            print("x2.shape", x2.shape)
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            print("x5.shape", x5.shape)
            print("x1.shape", x1.shape)
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        x6 = F.interpolate(x6, size=(300, 365), mode='bilinear', align_corners=False)

        if self.skip_connection:
            print("x6.shape", x6.shape)
            print("x0.shape", x0.shape)
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out