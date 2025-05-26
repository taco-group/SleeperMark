import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.linear = nn.Linear(in_features, out_features)

        if self.activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'selu':
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.act is not None:
            outputs = self.act(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1, init = None):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
 
        if init == "kaiming_normal":
            nn.init.kaiming_normal_(self.conv.weight)
        if init == "zero":
            nn.init.constant_(self.conv.weight, 0)
            nn.init.constant_(self.conv.bias, 0)

        if self.activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'selu':
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.act is not None:
            outputs = self.act(outputs)

        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.contiguous().view(input.size(0), -1)

    
class Extractor_forLatent(nn.Module):
    def __init__(self, secret_size = 48):
        super(Extractor_forLatent, self).__init__()
        self.decoder = nn.Sequential(
            Conv2D(4, 64, 3, strides=2, activation='selu'),
            Conv2D(64, 64, 3, activation='selu'),
            Conv2D(64, 128, 3, strides=2, activation='selu'),
            Conv2D(128, 128, 3, activation='selu'),
            Conv2D(128, 256, 3, strides=2, activation='selu'),
            Conv2D(256, 256, 3, activation='selu'),
            Conv2D(256, 512, 3, strides=2, activation='selu'),
            Conv2D(512, 512, 3, activation='selu'),
            Flatten())
        
        self.mlps = nn.Sequential(
            Linear(8192, 2048, activation='selu'), 
            Linear(2048, 2048, activation='selu'), 
            Linear(2048, 2048, activation='selu'), 
            torch.nn.Dropout(p=0.1),
            Linear(2048, secret_size, activation=None))

    def forward(self, latent):
        decoded = self.decoder(latent)
        decoded = self.mlps(decoded)

        return decoded
    