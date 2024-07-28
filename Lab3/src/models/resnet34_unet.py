import torch
import torch.nn as nn

def crop_and_concat(upsampled, bypass):
    c = (bypass.size()[2] - upsampled.size()[2]) // 2
    bypass = bypass[:, :, c:c+upsampled.size()[2], c:c+upsampled.size()[3]]
    return torch.cat((upsampled, bypass), 1)

class building_block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(building_block, self).__init__()
        if downsample:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3, 3), stride=(2, 2), padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
                nn.BatchNorm2d(out_channels)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=(2, 2)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
                nn.BatchNorm2d(out_channels)
            )
            if in_channels is not out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, (1, 1)),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()
        
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv(x)
        x += identity
        x = self.relu(x)
        return x

class upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, (2, 2), (2, 2))
        self.conv1 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x, bypass):
        x = self.up(x)
        x = crop_and_concat(x, bypass)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class ResNet34_UNet(nn.Module):
    def __init__(self):
        super(ResNet34_UNet, self).__init__()
        
        self.conv1 = building_block(3, 64)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = building_block(64, 64)
        self.conv3 = building_block(64, 128)
        self.conv4 = building_block(128, 256)
        self.conv5 = building_block(256, 512)
        
        self.bridge = building_block(512, 256)
        
        self.up1 = upsample(256 + 512, 32)
        self.up2 = upsample(32 + 256, 32)
        self.up3 = upsample(32 + 128, 32)
        self.up4 = upsample(32 + 64, 32)
        self.up5 = upsample(32, 32)
        
        self.outc = building_block(32, 2)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        print("e1 size :",x.shape)
        x = self.maxpool(x)
        print("e2 size :",x.shape)
        x1 = self.conv2(x)
        print("e3 size :",x1.shape)
        x2 = self.conv3(x1)
        print("e4 size :",x2.shape)
        x3 = self.conv4(x2)
        print("e5 size :",x3.shape)
        x4 = self.conv5(x3)
        print("e6 size :",x4.shape)
        
        x = self.bridge(x4)
        print("m size :",x.shape)
        
        # Decoder
        x = self.up1(x, x3)
        print("d1 size :",x.shape)
        x = self.up2(x, x2)
        print("d2 size :",x.shape)
        x = self.up3(x, x1)
        print("d3 size :",x.shape)
        x = self.up4(x, None)
        print("d4 size :",x.shape)
        x = self.up5(x, None)
        print("d5 size :",x.shape)
        
        x = self.outc(x)
        print("output size :",x.shape)
        return x

model = ResNet34_UNet()
model.forward(torch.rand(64, 3, 256, 256))