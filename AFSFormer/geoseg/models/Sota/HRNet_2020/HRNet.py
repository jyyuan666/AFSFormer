import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geoseg.models.Sota.HRNet_2020.backbone import BN_MOMENTUM, hrnet_classification


class HRnet_Backbone(nn.Module):
    def __init__(self, backbone='hrnetv2_w32', pretrained=False):
        super(HRnet_Backbone, self).__init__()
        self.model = hrnet_classification(backbone=backbone, pretrained=pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)

        return y_list


class HRnet(nn.Module):
    def __init__(self, num_classes=2, backbone='hrnetv2_w32', pretrained=False):
        super(HRnet, self).__init__()
        self.backbone = HRnet_Backbone(backbone=backbone, pretrained=pretrained)

        # last_inp_channels = np.int(np.sum(self.backbone.model.pre_stage_channels))
        last_inp_channels = int(np.sum(self.backbone.model.pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x




# net = HRnet(num_classes=2)

import torch
from thop import profile, clever_format

net = HRnet(num_classes=2)

# input_size = (3, 224, 224)
# input_data = torch.randn(1, *input_size)
#
# # 计算FLOPs和参数量
# flops, params = profile(net, inputs=(input_data,))
# flops = clever_format(flops, "%.3f")
# params = clever_format(params, "%.3f")
#
# print("FLOPs:", flops)
# print("Params:", params)
#
# # 计算内存占用量
# input_data = input_data.cuda()  # 将输入数据移动到GPU上
# net = net.cuda()  # 将模型移动到GPU上
#
# with torch.cuda.device(0):
#     input_data = input_data.cuda()
#     net = net.cuda()
#
#     # 使用torch.cuda.max_memory_allocated()和torch.cuda.max_memory_cached()获取内存占用量
#     torch.cuda.reset_max_memory_allocated()
#     torch.cuda.reset_max_memory_cached()
#
#     _ = net(input_data)
#
#     memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0  # 将字节转换为兆字节
#
# print("Memory:", memory)


print(net.backbone)
print(222222222222222222222222)


model = net
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
#
# from torchstat import stat
# import torchvision.models as models
# # model = models.resnet152()
# stat(net.backbone, (3, 224, 224))
