# implement SCCNet model

import torch
import torch.nn as nn
import torch.nn.functional as F

# # reference paper: https://ieeexplore.ieee.org/document/8716937
# class SquareLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         pass

# class SCCNet(nn.Module):
#     def __init__(self, numClasses=0, timeSample=0, Nu=0, C=0, Nc=0, Nt=0, dropoutRate=0):
#         super(SCCNet, self).__init__()
#         pass

#     def forward(self, x):
#         pass

#     # if needed, implement the get_size method for the in channel of fc layer
#     def get_size(self, C, N):
#         pass

class SCCNet(nn.Module):
    def __init__(self, num_channels, num_classes, num_filters=22, kernel_size=(22, 1), second_kernel_size=(22, 12), dropout_rate=0.5):
        super(SCCNet, self).__init__()
        
        # First Convolution Block (Spatial Filtering)
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=(num_channels, kernel_size[1]), padding=(0, 0))
        self.batch_norm1 = nn.BatchNorm2d(num_filters)
        
        # Second Convolution Block (Spatial-Temporal Filtering)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=second_kernel_size, padding=(0, 0))
        self.batch_norm2 = nn.BatchNorm2d(num_filters)
        
        # Average Pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12), padding=0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # First Convolution Block
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = x ** 2
        
        # Second Convolution Block
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = x ** 2
        
        # Average Pooling
        x = self.avg_pool(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Define Fully Connected Layer dynamically
        if not hasattr(self, 'fc'):
            self.fc = nn.Linear(x.size(1), num_classes)
        
        # Dropout
        x = self.dropout(x)
        
        # Fully Connected Layer
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

# Example usage:
# num_channels = 22 (number of EEG channels)
# num_classes = 4 (left hand, right hand, feet, tongue)
model = SCCNet(num_channels=22, num_classes=4)
print(model)