import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv2(self.conv1(x))

        return x


class UpSampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.pre_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.upconv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.upconv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm = norm_layer(out_channels)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.pre_conv(x)
        x = torch.cat((x, res), dim=1)
        x = self.upconv2(self.upconv1(x))
        x = self.norm(x)

        return x


class Unet(nn.Module):
    def __init__(self,
                 in_channels=(3, 64, 128, 256, 512, 1024),
                 out_channels=(1024, 512, 256, 128, 64),
                 n_classes=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(in_channels[1])
        self.conv64 = nn.Conv2d(in_channels=out_channels[4], out_channels=out_channels[4], kernel_size=3, padding=1)

        self.b1 = ConvBlock(in_channel=in_channels[0], out_channel=in_channels[1])
        self.b2 = ConvBlock(in_channel=in_channels[1], out_channel=in_channels[2])
        self.b3 = ConvBlock(in_channel=in_channels[2], out_channel=in_channels[3])
        self.b4 = ConvBlock(in_channel=in_channels[3], out_channel=in_channels[4])

        self.post_conv_512to1024 = nn.Conv2d(in_channels=in_channels[4], out_channels=in_channels[5], kernel_size=1)
        self.post_conv = nn.Conv2d(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1)

        self.up1 = UpSampleBlock(in_channels=out_channels[0], out_channels=out_channels[1])
        self.up2 = UpSampleBlock(in_channels=out_channels[1], out_channels=out_channels[2])
        self.up3 = UpSampleBlock(in_channels=out_channels[2], out_channels=out_channels[3])
        self.up4 = UpSampleBlock(in_channels=out_channels[3], out_channels=out_channels[4])

        self.seghead = nn.Conv2d(in_channels=out_channels[4], out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.b1(x)  # (2, 3, 1024, 1024) -> (2, 64, 1024, 1024)
        x1_0 = self.maxpool(x1) # (2, 64, 1024, 1024) -> (2, 64, 512, 512)

        x2 = self.b2(x1_0)  # (2, 64, 512, 512) -> (2, 128, 512, 512)
        x2_0 = self.maxpool(x2)  # (2, 128, 512, 512) -> (2, 128, 256, 256)

        x3 = self.b3(x2_0)  # (2, 128, 256, 256) -> (2, 256, 256, 256)
        x3_0 = self.maxpool(x3) # (2, 256, 256, 256) -> (2, 256, 128, 128)

        x4 = self.b4(x3_0)  # (2, 256, 128, 128) -> (2, 512, 128, 128)
        x4_0 = self.maxpool(x4)  # (2, 512, 128, 128) -> (2, 512, 64, 64)

        x = self.post_conv_512to1024(x4_0)
        x = self.post_conv(x)  # x = (2, 1024, 64, 64)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.conv64(x)
        x = self.seghead(x)

        return x


model = Unet(n_classes=2)

import numpy as np
device = torch.device('cuda')
model.to(device)
dummy_input = torch.randn(1, 3, 1024, 1024,dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
   _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     starter.record()
     _ = model(dummy_input)
     ender.record()
     # WAIT FOR GPU SYNC
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)
     timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000. / mean_syn
print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
print(mean_syn)




