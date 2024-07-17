import torch
import torch.nn as nn
import torch.nn.functional as F


import timm
from timm.models.layers import DropPath


class DWConvBNRelu(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 norm_layer=nn.BatchNorm2d
                 ):
        super(DWConvBNRelu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class DWConvBN(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 norm_layer=nn.BatchNorm2d):
        super(DWConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
            norm_layer(out_channels)
        )


class DWConv(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1):
        super(DWConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )


class Mlp(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim=None,
                 out_dim=None,
                 act_layer=nn.ReLU6,
                 drop=0.,):
        super().__init__()

        hidden_dim = in_dim or hidden_dim
        out_dim = in_dim or out_dim

        self.fc1 = nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 fuse_channels,
                 last_block = False
                 ):
        super().__init__()

        self.flag = last_block

        self.in_channels = input_channels
        self.out_channels = output_channels
        self.fuse_channel = fuse_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        self.preconv = nn.Conv2d(self.fuse_channel, self.out_channels, kernel_size=1)
        self.activate = nn.ReLU6()
        self.norm = nn.BatchNorm2d(self.out_channels)

    def forward(self, x, res):

        if self.flag == False:
            res = self.preconv(res)

            x = self.conv2(self.conv1(x))
            x = self.norm(self.activate(x))
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            out = res + x

        return out


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64,128,256,512),
                 decoder_channels=64,
                 dropout=0.1,
                 num_classes=2):
        super(Decoder, self).__init__()

        self.b1 = DecoderBlock(input_channels=encoder_channels[-1], output_channels=encoder_channels[2],
                               fuse_channels=encoder_channels[2])

        self.b2 = DecoderBlock(input_channels=encoder_channels[2], output_channels=encoder_channels[1],
                               fuse_channels=encoder_channels[1])

        self.b3 = DecoderBlock(input_channels=encoder_channels[1], output_channels=encoder_channels[0],
                               fuse_channels=encoder_channels[0])

        self.refine = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0]//2, kernel_size=3, padding=1),
            nn.Conv2d(encoder_channels[0]//2, encoder_channels[0], kernel_size=1)
        )



        self.segmentation_head = nn.Sequential(
            DWConvBNRelu(in_channels=decoder_channels, out_channels=decoder_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            DWConv(decoder_channels, num_classes, kernel_size=1, padding=0))



        self.init_weight()



    def forward(self, res1, res2, res3, res4):

        out = self.b1(res4, res3)

        out = self.b2(out, res2)

        out = self.b3(out, res1)

        out = self.refine(out)

        out = self.segmentation_head(out)

        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Baseline(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, num_classes)

    def forward(self, x):
        res1, res2, res3, res4 = self.backbone(x)

        x = self.decoder(res1, res2, res3, res4)

        return x



import torch

x = torch.randn(1, 3, 1024, 1024)

model = Baseline()

y = model(x)

print(y.shape)

