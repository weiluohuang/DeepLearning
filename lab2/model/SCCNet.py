# implement SCCNet model
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
    def __init__(self, numClasses=4, timeSample=438, Nu=22, C=22, Nc=20, Nt=16, dropoutRate=0.5):
        super(SCCNet, self).__init__()
        self.conv1 = nn.Conv2d(1, Nu, (C, Nt), padding=0) #(batch size ,kernel ,height ,width)
        self.bn1 = nn.BatchNorm2d(Nu)
        self.conv2 = nn.Conv2d(1, Nc, (Nu, 12), padding=(0,6))
        self.bn2 = nn.BatchNorm2d(Nc)
        self.square = SquareLayer()
        self.dropout = nn.Dropout(dropoutRate)
        self.pool = nn.AvgPool2d((1, 64), stride=(1, 12)) 
        afterPoolSize = (timeSample-64)//12 
        self.fc = nn.Linear(Nc*afterPoolSize, numClasses) #620
        
     
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.permute(0,2, 1, 3)
        x = self.conv2(x) #[32, 20, 1, 427]
        x = self.bn2(x) #[32, 20, 1, 427]
        x = self.square(x) # [32, 20, 1, 427]
        x = self.dropout(x) # [32, 20, 1, 427]     
        x = x.permute(0,2, 1, 3)
        x = self.pool(x) # [32, 20, 1, 31]
        x = x.flatten(1)   #[32, 620]
        x = self.fc(x) #[32, 4]
        return x
    
    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass
    