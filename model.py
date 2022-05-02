import torch
import torch.nn as nn
from torch import Tensor as tensor
from torch.nn import functional as F
import torchvision.models as models

class SCA_Block(nn.Module):
    def __init__(self, in_channel, downsample_channel):
        super().__init__()
        self.conv_A = nn.Conv2d(in_channel, downsample_channel, (1,1))
        self.conv_B = nn.Conv2d(in_channel, downsample_channel, (1,1))
        self.conv_E = nn.Conv2d(in_channel, downsample_channel, (1,1))
        self.linear = nn.Linear(downsample_channel,in_channel)
    def forward(self, feature_in):
        b_size,c,w,h = feature_in.shape
        A = self.conv_A(feature_in)
        B = self.conv_B(feature_in)
        E = self.conv_E(feature_in)
        c1 = A.shape[1]
        Z = F.softmax(torch.dot(torch.reshape(A,(b_size,c1,-1)),
                                torch.reshape(B,(b_size,-1,c1))), axis = 1 )
        D = torch.reshape( Z * torch.reshape(E,(b_size,c1,-1)) , (b_size,c1,w,h))
        out = feature_in * F.sigmoid(F.adaptive_avg_pool2d(D))
        return out


class AQC_NET(nn.Module):
    def __init__(self, pretrain = True, num_label = 5):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained = pretrain)
        self.resnet18.layer3[0].add_module('sca_1', SCA_Block(256,16))
        self.resnet18.layer3[1].add_module('sca_2', SCA_Block(256,16))
        self.resnet18.fc = nn.Linear(512,num_label)
    def forward(self,x):
        return F.softmax(self.resnet18(x))
