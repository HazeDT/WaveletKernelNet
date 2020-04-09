import torch
import torch.nn as nn
from math import pi
import torch.nn.functional as F

def Mexh(p):
    p = 0.04 * p  # 将时间转化为在[-5,5]这个区间内
    y = (1 - torch.pow(p, 2)) * torch.exp(-torch.pow(p, 2) / 2)

    return y

class Mexh_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):

        super(Mexh_fast, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels

        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1


        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):

        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))

        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = time_disc_right.cuda() - self.b_.cuda() / self.a_.cuda()

        p2 = time_disc_left.cuda() - self.b_.cuda() / self.a_.cuda()

        Mexh_right = Mexh(p1)
        Mexh_left = Mexh(p2)

        Mexh_filter = torch.cat([Mexh_left, Mexh_right], dim=1)  # 40x1x250

        self.filters = (Mexh_filter).view(self.out_channels, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)

# -----------------------input size>=32---------------------------------
class Mexhat_LeNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(Mexhat_LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            Mexh_fast(64, 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(25)  # adaptive change the outputsize to (16,5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 25, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x