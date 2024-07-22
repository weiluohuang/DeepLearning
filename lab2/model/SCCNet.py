import torch
import torch.nn as nn
import torch.nn.functional as F
# reference paper: https://ieeexplore.ieee.org/document/8716937

class SquareLayer(nn.Module):
    def __init__(self):
        super(SquareLayer, self).__init__()

    def forward(self, x):
        return x ** 2

    def back(self, x):
        return 2*x

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=44, C=22, Nc=20, Nt=8, dropoutRate=0.5):
        super(SCCNet, self).__init__()
        self.conv1 = nn.Conv2d(1, Nu, (C, Nt), padding=0)
        self.bn1 = nn.BatchNorm2d(Nu)
        self.conv2 = nn.Conv2d(1, Nc, (Nu, 12), padding=(0,6))
        self.bn2 = nn.BatchNorm2d(Nc)
        self.square = SquareLayer()
        self.dropout = nn.Dropout(dropoutRate)
        self.pool = nn.AvgPool2d((1, 62), stride=(1, 12)) 
        PooledSize = (timeSample-Nt+1+1-62)//12+1 #Nt reduce, padding increase
        self.fc = nn.Linear(Nc*PooledSize, numClasses)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1, 3)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.square(x)
        x = self.dropout(x) 
        x = x.permute(0, 2, 1, 3)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
    
    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass

