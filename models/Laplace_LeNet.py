import torch
import torch.nn as nn
from math import pi
import torch.nn.functional as F

def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 50
    w = 2 * pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))
    return y


class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):

        super(Laplace_fast, self).__init__()

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

        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))

        p1 = time_disc.cuda() - self.b_.cuda() / self.a_.cuda()

        laplace_filter = Laplace(p1)

        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size).cuda()


        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)

# -----------------------input size>=32---------------------------------
class Laplace_LeNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(Laplace_LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            Laplace_fast(64, 16),
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