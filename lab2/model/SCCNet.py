# implement SCCNet model
import torch
import torch.nn as nn
import torch.nn.functional as F
# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = (0, 0), stride=1):
        super(SquareLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape()
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        unfolded_x = F.unfold(x, self.kernel_size, stride=self.stride, padding=self.padding)
        unfolded_weight = self.weight.view(self.out_channels, -1)
        out_unfolded = unfolded_weight @ unfolded_x
        out_unfolded += self.bias.view(-1, 1)
        out = F.fold(out_unfolded, output_size=(out_height, out_width), kernel_size=1)
        return out

    def back(self):
        pass

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=0, Nu=0, C=0, Nc=0, Nt=0, dropoutRate=0.5):
        super(SCCNet, self).__init__()
        self.conv1 = SquareLayer(1, Nu, (C, Nt))
        self.conv2 = SquareLayer(1, Nc, (Nu, 12))
        self.dropout1 = nn.Dropout2d(dropoutRate)
        self.dropout2 = nn.Dropout2d(dropoutRate)
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))
        self.softmax = nn.Softmax(dim=0)
              
    def forward(self, x):
        x = self.conv1(x).permute(2,1,3)
        x = x**2
        self.dropout1(x)
        x = self.conv2(x)
        x = x**2
        x = self.dropout2(x)
        x = self.avg_pool(x)
        return self.softmax(x)
    
    def train(self, train_loader, optimizer, epoch, device='cuda'):
        print(self.foward(train_loader))
        

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass