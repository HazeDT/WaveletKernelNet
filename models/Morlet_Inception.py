#coding=utf-8
import torch
import torch.nn as nn
from math import pi
import torch.nn.functional as F

def Morlet(p):
    C = pow(pi, 0.25)
    # p = 0.03 * p
    y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * pi * p)
    return y

class Morlet_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):

        super(Morlet_fast, self).__init__()

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

        Morlet_right = Morlet(p1)
        Morlet_left = Morlet(p2)

        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250

        self.filters = (Morlet_filter).view(self.out_channels, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)
#----------------------------------------------------------------------------
class BasicConv1d(nn.Module):
 
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,**kwargs, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



class InceptionA(nn.Module):
 
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, 64, kernel_size=1)
 
        self.branch5x5_1 = BasicConv1d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv1d(48, 64, kernel_size=5, padding=2)
 
        self.branch3x3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(96, 96, kernel_size=3, padding=1)
 
        self.branch_pool = BasicConv1d(in_channels, pool_features, kernel_size=1)
 
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
 
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
 
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
 
        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
 
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
 
 
class InceptionB(nn.Module):
 
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv1d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(96, 96, kernel_size=3, stride=2)
 
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
 
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
 
        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)
 
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class Morlet_Inception(nn.Module):
 
    def __init__(self,in_channel,out_channel=10):
        super(Morlet_Inception, self).__init__()
        self.Morlet = Morlet_fast(32,16)
        self.bn = nn.BatchNorm1d(32, eps=0.001)
        self.Conv1d_2a_3x3 = BasicConv1d(32, 32, kernel_size=3)
        self.Conv1d_2b_3x3 = BasicConv1d(32, 64, kernel_size=3, padding=1)
        self.Conv1d_3b_1x1 = BasicConv1d(64, 80, kernel_size=1)
        self.Conv1d_4a_3x3 = BasicConv1d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.fc = nn.Linear(768, out_channel)
    def forward(self,x):
        # 299 x 299 x 3
        #print("x0:",x.size())
        x = self.Morlet(x)
        x = self.bn(x)
        #print("x1:",x.size())
        # 64, 32, 511
        x = self.Conv1d_2a_3x3(x)
        #print("x2:",x.size())
        # 147 x 147 x 32
        x = self.Conv1d_2b_3x3(x)
        #print("x3:",x.size())
        # 147 x 147 x 64
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        #print("x4:",x.size())
        # 73 x 73 x 64
        x = self.Conv1d_3b_1x1(x)
        #print("x5:",x.size())
        # 73 x 73 x 80
        x = self.Conv1d_4a_3x3(x)
        #print("x6:",x.size())
        # 71 x 71 x 192
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        #print("x7:",x.size())
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        #print("x8:",x.size())
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        #print("x9:",x.size())
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        #print("x10:",x.size())
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        #print("x11:",x.size())
        # 17 x 17 x768
        x = nn.AdaptiveMaxPool1d(1)(x)
        #print("x12:",x.size())
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        #print("x13:",x.size())
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        #print("x14:",x.size())
        # 2048

        x = self.fc(x)
        #print("x15:",x.size())
        return x

