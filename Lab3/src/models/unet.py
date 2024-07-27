import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample(in_channel, out_channel):
    return nn.Sequential(nn.Conv2d（（)
class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3))
        self.conv2 = nn.Conv2d(64, 64, (3, 3))
        self.pool1 = nn.MaxPool2d(64, 128, (2, 2))
        self.conv3 = nn.Conv2d(3, 64, (3, 3))
        self.conv4 = nn.Conv2d(3, 64, (3, 3))
        self.pool2 = nn.MaxPool2d(64, 128, (2, 2))
        self.conv5 = nn.Conv2d(3, 64, (3, 3))