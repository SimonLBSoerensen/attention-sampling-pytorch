import torch.nn as nn
from ats.utils.layers import SampleSoftmax

class AttentionModelTrafficSigns(nn.Module):
    """ Base class for calculating the attention map of a low resolution image """

    def __init__(self,
                 squeeze_channels=False,
                 softmax_smoothing=0.0):
        super(AttentionModelTrafficSigns, self).__init__()

        conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding_mode='zeros')
        relu1 = nn.ReLU()

        conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding_mode='zeros')
        relu2 = nn.ReLU()

        conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding_mode='zeros')
        relu3 = nn.ReLU()

        conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding_mode='zeros')

        pool = nn.MaxPool2d(kernel_size=8)
        sample_softmax = SampleSoftmax(squeeze_channels, softmax_smoothing)

        self.part1 = nn.Sequential(conv1, relu1, conv2, relu2, conv3, relu3)
        self.part2 = nn.Sequential(conv4, pool, sample_softmax)

    def forward(self, x_low):
        out = self.part1(x_low)

        out = self.part2(out)

        return out


class AttentionModelJD(nn.Module):
    """ Base class for calculating the attention map of a low resolution image """

    def __init__(self,
                 squeeze_channels=False,
                 softmax_smoothing=1e-4):
        super(AttentionModelJD, self).__init__()

        layers = []

        conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding_mode='zeros')
        bnorm1 = nn.BatchNorm2d(8)
        relu1 = nn.ReLU()
        layers += [conv1, bnorm1, relu1]

        conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding_mode='zeros')
        bnorm2 = nn.BatchNorm2d(16)
        relu2 = nn.ReLU()
        layers += [conv2, bnorm2, relu2]

        conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding_mode='zeros')
        bnorm3 = nn.BatchNorm2d(32)
        relu3 = nn.ReLU()
        layers += [conv3, bnorm3, relu3]

        conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding_mode='zeros')
        bnorm4 = nn.BatchNorm2d(64)
        relu4 = nn.ReLU()
        layers += [conv4, bnorm4, relu4]

        conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding_mode='zeros')
        bnorm5 = nn.BatchNorm2d(128)
        relu5 = nn.ReLU()
        layers += [conv5, bnorm5, relu5]

        conv6 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding_mode='zeros')
        sample_softmax = SampleSoftmax(squeeze_channels, softmax_smoothing)

        self.part1 = nn.Sequential(*layers)
        self.part2 = nn.Sequential(conv6, sample_softmax)

    def forward(self, x_low):
        out = self.part1(x_low)

        out = self.part2(out)

        return out


class AttentionModelMNIST(nn.Module):
    """ Base class for calculating the attention map of a low resolution image """

    def __init__(self,
                 squeeze_channels=False,
                 softmax_smoothing=0.0):
        super(AttentionModelMNIST, self).__init__()

        self.squeeze_channels = squeeze_channels
        self.softmax_smoothing = softmax_smoothing

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, padding_mode='reflect')
        self.tanh1 = nn.Tanh()

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, padding_mode='reflect')
        self.tanh2 = nn.Tanh()

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect')

        self.sample_softmax = SampleSoftmax(squeeze_channels, softmax_smoothing)

    def forward(self, x_low):
        out = self.conv1(x_low)
        out = self.tanh1(out)

        out = self.conv2(out)
        out = self.tanh2(out)

        out = self.conv3(out)
        out = self.sample_softmax(out)

        return out


class AttentionModelColonCancer(nn.Module):
    """ Base class for calculating the attention map of a low resolution image """

    def __init__(self,
                 squeeze_channels=False,
                 softmax_smoothing=0.0):
        super(AttentionModelColonCancer, self).__init__()

        conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding_mode='zeros', padding=1)
        relu1 = nn.ReLU()

        conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding_mode='zeros', padding=1)
        relu2 = nn.ReLU()

        conv3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding_mode='zeros', padding=1)

        sample_softmax = SampleSoftmax(squeeze_channels, softmax_smoothing)

        self.forward_pass = nn.Sequential(conv1, relu1, conv2, relu2, conv3, sample_softmax)

    def forward(self, x_low):
        out = self.forward_pass(x_low)
        return out
