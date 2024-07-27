import torch
import torch.nn as nn

def crop_and_concat(upsampled, bypass):
    c = (bypass.size()[2] - upsampled.size()[2]) // 2
    bypass = bypass[:, :, c:c+upsampled.size()[2], c:c+upsampled.size()[3]]
    return torch.cat((upsampled, bypass), 1)

def downsample(in_channels):
    return nn.Sequential(nn.MaxPool2d((2, 2),(2, 2)),
                        nn.Conv2d(in_channels, 2*in_channels, (3, 3), padding=1),
                        nn.ReLU(),
                        nn.Conv2d(2*in_channels, 2*in_channels, (3, 3), padding=1),
                        nn.ReLU())

class upsample(nn.Module):
    def __init__(self, in_channels):
        super(upsample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, (2, 2), (2, 2))
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, (3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, (3, 3), padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x, bypass):
        x = self.up(x)
        x = crop_and_concat(x, bypass)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU()
        )
        self.down1 = downsample(64)
        self.down2 = downsample(128)
        self.down3 = downsample(256)
        self.down4 = downsample(512)
        
        self.up1 = upsample(1024)
        self.up2 = upsample(512)
        self.up3 = upsample(256)
        self.up4 = upsample(128)
        
        self.outc = nn.Conv2d(64, out_channels, (1, 1))

    def forward(self, x):
        x1 = self.inc(x)
        # print("d1 size : ",x1.shape)
        x2 = self.down1(x1)
        # print("d2 size : ",x2.shape)
        x3 = self.down2(x2)
        # print("d3 size : ",x3.shape)
        x4 = self.down3(x3)
        # print("d4 size : ",x4.shape)
        x5 = self.down4(x4)
        # print("d5 size : ",x5.shape)

        x = self.up1(x5, x4)
        # print("u1 size : ",x.shape)
        x = self.up2(x, x3)
        # print("u2 size : ",x.shape)
        x = self.up3(x, x2)
        # print("u3 size : ",x.shape)
        x = self.up4(x, x1)
        # print("u4 size : ",x.shape)
        
        x = self.outc(x)
        # print("output size : ",x.shape)
        return x

# d1 size :  torch.Size([64, 64, 256, 256])
# d2 size :  torch.Size([64, 128, 128, 128])
# d3 size :  torch.Size([64, 256, 64, 64])
# d4 size :  torch.Size([64, 512, 32, 32])
# d5 size :  torch.Size([64, 1024, 16, 16])
# u1 size :  torch.Size([64, 512, 32, 32])
# u2 size :  torch.Size([64, 256, 64, 64])
# u3 size :  torch.Size([64, 128, 128, 128])
# u4 size :  torch.Size([64, 64, 256, 256])
# output size :  torch.Size([64, 1, 256, 256])
# model = UNet(3, 1)
# model.forward(torch.rand(64, 3, 256, 256))