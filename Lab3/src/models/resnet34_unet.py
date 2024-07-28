import torch
import torch.nn as nn

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
        if bypass is not None:
            x = torch.cat((x, bypass), 1)
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu2(x)

class ResNet34_UNet(nn.Module):
    def __init__(self):
        super(ResNet34_UNet, self).__init__()
        
        self.conv1 = building_block(3, 64, downsample=True)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = building_block(64, 64)
        self.conv3 = building_block(64, 128, downsample=True)
        self.conv4 = building_block(128, 256, downsample=True)
        self.conv5 = building_block(256, 512, downsample=True)
        self.conv6 = building_block(512, 256)
        
        self.up1 = upsample(768, 32)
        self.up2 = upsample(288, 32)
        self.up3 = upsample(160, 32)
        self.up4 = upsample(96, 32)
        self.up5 = upsample(32, 32)
        
        self.outc = building_block(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        # print("after conv1 :",x.shape)
        x = self.maxpool(x)
        # print("after maxpool :",x.shape)
        x1 = self.conv2(x)
        # print("after conv2 :",x1.shape)
        x2 = self.conv3(x1)
        # print("after conv3 :",x2.shape)
        x3 = self.conv4(x2)
        # print("after conv4 :",x3.shape)
        x4 = self.conv5(x3)
        # print("after conv5 :",x4.shape)
        x = self.conv6(x4)
        # print("after conv6 :",x.shape)
        
        # Decoder
        x = self.up1(x, x4)
        # print("after up1 :",x.shape)
        x = self.up2(x, x3)
        # print("after up2 :",x.shape)
        x = self.up3(x, x2)
        # print("after up3 :",x.shape)
        x = self.up4(x, x1)
        # print("after up4 :",x.shape)
        x = self.up5(x, None)
        # print("after up5 :",x.shape)
        
        x = self.outc(x)
        # print("output size :",x.shape)
        return x

# model = ResNet34_UNet()
# model.forward(torch.rand(64, 3, 256, 256))

# after conv1 : torch.Size([64, 64, 128, 128])
# after maxpool : torch.Size([64, 64, 64, 64])
# after conv2 : torch.Size([64, 64, 64, 64])
# after conv3 : torch.Size([64, 128, 32, 32])
# after conv4 : torch.Size([64, 256, 16, 16])
# after conv5 : torch.Size([64, 512, 8, 8])
# after conv6 : torch.Size([64, 256, 8, 8])
# after up1 : torch.Size([64, 32, 16, 16])
# after up2 : torch.Size([64, 32, 32, 32])
# after up3 : torch.Size([64, 32, 64, 64])
# after up4 : torch.Size([64, 32, 128, 128])
# after up5 : torch.Size([64, 32, 256, 256])
# output size : torch.Size([64, 2, 256, 256])