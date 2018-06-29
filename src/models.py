import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # Convolutional Layer: 100x1x1 -> 512x1x2, F=(1,2), S=1, P=0
        self.conv1 = nn.ConvTranspose2d(100, 512, (1, 2), 1, 0, bias=False)
        self.conv1_bn = nn.BatchNorm2d(512)

        # Convolutional Layer: 512x1x2 -> 256x2x4, F=4, S=2, P=1
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(256)

        # Convolutional Layer: 256x2x4 -> 128x4x8, F=4, S=2, P=1
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(128)

        # Convolutional Layer: 128x4x8 -> 64x8x16, F=4, S=2, P=1
        self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(64)

        # Convolutional Layer: 64x8x16 -> 4x16x32, F=4, S=2, P=1
        self.conv5 = nn.ConvTranspose2d(64, 4, 4, 2, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.sigmoid(self.conv5(x))
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # Convolutional Layer: 4x16x32 -> 64x8x16, F=4, S=2, P=1
        self.conv1 = nn.Conv2d(4, 64, 4, 2, 1, bias=False)

        # Convolutional Layer: 64x8x16 -> 128x4x8, F=4, S=2, P=1
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(128)

        # Convolutional Layer: 128x4x8 -> 256x2x4, F=4, S=2, P=1
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(256)

        # Convolutional Layer: 256x2x4 -> 512x1x2, F=4, S=2, P=1
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(512)

        # Convolutional Layer: 512x1x2 -> 1x1x1, F=(1,2), S=1, P=0
        self.conv5 = nn.Conv2d(512, 1, (1, 2), 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), negative_slope=0.2)
        x = F.sigmoid(self.conv5(x))
        return x
