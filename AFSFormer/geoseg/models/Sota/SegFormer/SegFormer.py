# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from geoseg.models.Sota.SegFormer.backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


# class MLP(nn.Module):
#     """
#     Linear Embedding
#     """
#
#     def __init__(self, input_dim=2048, embed_dim=768):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, embed_dim)
#
#     def forward(self, x):
#         x = x.flatten(2).transpose(1, 2)
#         x = self.proj(x)
#         return x

class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = embed_dim or input_dim
        hidden_features = hidden_features or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 从原始MLP类中添加
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x




class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

# 第一次加
# class ConvModule(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
#         super(ConvModule, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
#         self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#         self.cloformer = self.CloformerLayer()  # 添加CloFormer层
#
#     def forward(self, x):
#         x = self.conv(x)
#
#         # 以下为CloFormer的模拟操作
#         b, c, h, w = x.shape
#         x = x.view(b, c, -1)  # 展平操作
#         x_mean = torch.mean(x, dim=2, keepdim=True)  # 计算每个通道的均值
#         x = x - x_mean  # 减去均值
#         x_std = torch.std(x, dim=2, keepdim=True)  # 计算每个通道的标准差
#         x = x / (x_std + 1e-6)  # 归一化
#         x = x.view(b, c, h, w)  # 恢复原始形状
#
#         x = self.bn(x)
#         x = self.act(x)
#
#         return x
#
#     class CloformerLayer(nn.Module):
#         def __init__(self):
#             super(ConvModule.CloformerLayer, self).__init__()
#
#         def forward(self, x):
#             b, c, h, w = x.shape
#             x = x.view(b, c, -1)  # 展平操作
#             x_mean = torch.mean(x, dim=2, keepdim=True)  # 计算每个通道的均值
#             x = x - x_mean  # 减去均值
#             x_std = torch.std(x, dim=2, keepdim=True)  # 计算每个通道的标准差
#             x = x / (x_std + 1e-6)  # 归一化
#             x = x.view(b, c, h, w)  # 恢复原始形状
#
#             return x

# from dilateformer import DilateAttention


# 添加了MSDA模块
class MSDA(nn.Module):
    # def __init__(self, in_channels, out_channels, dilations=[1, 2, 3, 4]):
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 2, 1]):
        super(MSDA, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
            for dilation in dilations
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels * len(dilations), out_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        global_features = torch.cat([self.gap(feature).view(x.size(0), -1) for feature in features], dim=1)
        attention = self.fc(global_features).view(x.size(0), -1, 1, 1)
        return sum(feature * coef for feature, coef in zip(features, attention.split(1, dim=1)))

class SegFormerHead(nn.Module):
    # def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
    #     super(SegFormerHead, self).__init__()
    def __init__(self, num_classes=20, in_channels=[16, 32, 80, 128], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim // 2,
            k=1,
        )

        self.msda = MSDA(embedding_dim // 2, embedding_dim // 2)  # 添加MSDA模块

        self.linear_pred = nn.Conv2d(embedding_dim // 2, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.msda(_c)  # 应用MSDA模块

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

# class SegFormerHead(nn.Module):
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#
#     def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
#         super(SegFormerHead, self).__init__()
#         c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
#
#         self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
#         self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
#         self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
#         self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
#
#         self.linear_fuse = ConvModule(
#             c1=embedding_dim * 4,
#             c2=embedding_dim,
#             k=1,
#         )
#
#         self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
#         self.dropout = nn.Dropout2d(dropout_ratio)
#
#     def forward(self, inputs):
#         c1, c2, c3, c4 = inputs
#
#         ############## MLP decoder on C1-C4 ###########
#         n, _, h, w = c4.shape
#
#         _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
#
#         _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
#
#         _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
#         _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
#
#         _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
#
#         _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
#
#         x = self.dropout(_c)
#         x = self.linear_pred(x)
#
#         return x


class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b0', pretrained=False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
            # 'b0': [16, 32, 80, 128], 'b1': [32, 64, 160, 256], 'b2': [32, 64, 160, 256],
            # 'b3': [32, 64, 160, 256], 'b4': [32, 64, 160, 256], 'b5': [32, 64, 160, 256],
        }[phi]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
            # 'b0': 128, 'b1': 128, 'b2': 384,
            # 'b3': 384, 'b4': 384, 'b5': 384,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x




net = SegFormer(num_classes=2, phi='b1', pretrained=False)

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

