"""
encoder.py
A network model which extract features from a image
A Encoder class(heritage from nn.model)

@author: OminousBlackCat
"""

import torchvision as tv
from torch import nn
import math
import matplotlib.pyplot as plt
import layer


class Encoder(nn.Module):
    """
    6*conv layer
    input: raw image with shape(1(maybe 3), W, H)
    output: conv out with shape (384, W', H')
    """
    def __init__(self, input_channels=3, medium_channels=192 ,output_channels=384):
        super(Encoder, self).__init__()
        self.norm = layer.GDN
        self.cov1 = nn.Conv2d(input_channels, medium_channels, 5, stride=2, padding=2)
        nn.init.xavier_normal_(self.cov1.weight.data, (math.sqrt(2 * (3 + medium_channels) / 6)))
        nn.init.constant_(self.cov1.bias.data, 0.01)
        self.cov2 = nn.Conv2d(medium_channels, medium_channels, 5, stride=2, padding=2)
        nn.init.xavier_normal_(self.cov2.weight.data, math.sqrt(2))
        nn.init.constant_(self.cov2.bias.data, 0.01)
        self.cov3 = nn.Conv2d(medium_channels, medium_channels, 5, stride=2, padding=2)
        nn.init.xavier_normal_(self.cov3.weight.data, math.sqrt(2))
        self.cov4 = nn.Conv2d(medium_channels, output_channels, 5, stride=2, padding=2)
        nn.init.xavier_normal_(self.cov4.weight.data, math.sqrt(2))
        self.norm1 = layer.GDN(medium_channels, inverse=False)
        self.norm2 = layer.GDN(medium_channels, inverse=False)
        self.norm3 = layer.GDN(medium_channels, inverse=False)


    def forward(self, input_x):
        output_x = self.norm1(self.cov1(input_x))
        output_x = self.norm2(self.cov2(output_x))
        output_x = self.norm3(self.cov3(output_x))
        output_x = self.cov4(output_x)
        return output_x


class Decoder(nn.Module):
    """
    input: feature map of raw image(or decoder bytestream)
    layer1:
    """
    def __init__(self, input_channels=192, output_channels=3):
        super(Decoder, self).__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.norm1 = layer.GDN(input_channels, inverse=True)
        self.norm2 = layer.GDN(input_channels, inverse=True)
        self.norm3 = layer.GDN(input_channels, inverse=True)
        self.de_conv1 = nn.ConvTranspose2d(self.in_channels, self.in_channels, 5, stride=2, padding=2, output_padding=1)
        self.de_conv2 = nn.ConvTranspose2d(self.in_channels, self.in_channels, 5, stride=2, padding=2, output_padding=1)
        self.de_conv3 = nn.ConvTranspose2d(self.in_channels, self.in_channels, 5, stride=2, padding=2, output_padding=1)
        self.de_conv4 = nn.ConvTranspose2d(self.in_channels, 1, 5, stride=2, padding=2, output_padding=1)


    def forward(self, input_z):
        output_z = self.norm1(self.de_conv1(input_z))
        output_z = self.norm2(self.de_conv2(output_z))
        output_z = self.norm3(self.de_conv3(output_z))
        output_z = self.de_conv4(output_z)
        return output_z


if __name__ == '__main__':
    encoder = Encoder(1, 192, 192)
    img = tv.io.read_image("data/train/RSM20221001T000037_0001_HA.png", tv.io.ImageReadMode.GRAY)
    img = img[:, 78: 1078, 76: 1076].unsqueeze(0)
    # plt.imshow(img[0][0])
    # plt.show()
    img = img / 255
    print(img.shape)
    output_img = encoder.forward(img)
    print(output_img.shape)
    # plt.imshow(output_img[0][0].detach().numpy())
    # plt.show()
    decoder = Decoder(192, 192)
    decoder_out = decoder.forward(output_img)
    # plt.imshow(decoder_out[0][0].detach().numpy())
    # plt.show()

    print(decoder_out.shape)

