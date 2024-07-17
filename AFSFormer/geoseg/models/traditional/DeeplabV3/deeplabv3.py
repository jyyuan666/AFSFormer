import torch
import torch.nn as nn
import torch.nn.functional as F

from geoseg.models.traditional.DeeplabV3.assp import ASSP
from geoseg.models.traditional.DeeplabV3.resnet_50 import ResNet_50


class DeepLabv3(nn.Module):

    def __init__(self, nc):
        super(DeepLabv3, self).__init__()

        self.nc = nc

        self.resnet = ResNet_50()

        self.assp = ASSP(in_channels=1024)

        self.conv = nn.Conv2d(in_channels=256, out_channels=self.nc,
                              kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.resnet(x)
        x = self.assp(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear')  # scale_factor = 16, mode='bilinear')
        return x


model = DeepLabv3(2)


import numpy as np
device = torch.device('cuda')
model.to(device)
dummy_input = torch.randn(2, 3, 1024, 1024,dtype=torch.float).to(device)
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
