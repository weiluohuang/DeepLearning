import torch
import torch.nn as nn

print(torch.version.cuda)
print(torch.__version__)
print(torch.backends.cudnn.version())

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
        print("Input shape:", x.shape)
        x = x.unsqueeze(1)  # (batch , 1 ,22,438)
        print("After unsqueeze:", x.shape)
       
        ##############first layer####################
        x = self.conv1(x) #  (batch,22,1,438)
        print("After conv1:", x.shape)
        x = self.bn1(x)
        print("After bn1:", x.shape)
        print("After square1:", x.shape)
        x = self.dropout(x)
        x = x.permute(0,2, 1, 3) # [32, 1, 22, 438]
        print("After permute:", x.shape)
        
        ##############Second layer####################
        x = self.conv2(x) #[32, 20, 1, 427]
        print("After conv2:", x.shape)
        x = self.bn2(x) #[32, 20, 1, 427]
        print("After bn2:", x.shape)
        x = self.square(x) # [32, 20, 1, 427]
        print("After square:", x.shape)
        x = self.dropout(x) # [32, 20, 1, 427]
        print("After dropout:", x.shape)
        
        x = x.permute(0,2, 1, 3)
        print("After permute:", x.shape)
        ##############third layer####################
        x = self.pool(x) # [32, 20, 1, 31]
        print("After pool:", x.shape)

        ##############forth layer##############
        x = x.flatten(1)   #[32, 620]
        
        print("After flatten:", x.shape)
        x = self.fc(x) #[32, 4]
       # print("After fc:", x.shape)
       
        return x
    
model = SCCNet()
model.forward(torch.rand((87,22,438)))