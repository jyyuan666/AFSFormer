import torch
import torch.nn as nn
import torch.nn.functional as F

import torchsummary
import timm
from timm.models.layers import DropPath

'''Newv2 -> change the attention, HLA -> HLAv2'''
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


class HLA(nn.Module):
    '''
    Hilo attention.

    input dim (int): the channels of the input.
    num_heads(int): number of heads. default = 8.
    qkv_bias(float,optional): in not set, we use head_dim**-0.5
    window_size(int): the size of local window
    alpha(float): the ratio of num_heads used in Lo-Fi
    '''
    def __init__(self,
                 input_dim,
                 num_heads= 8,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_drop=0.,
                 projection_drop=0.,
                 window_size=2,
                 alpha=0.5):
        super().__init__()

        assert input_dim % num_heads == 0, print('input dim must divided by num_heads')
        head_dim = int(input_dim/num_heads)

        self.input_dim = input_dim
        # the number of heads of Lo-Fi
        self.l_heads = int(num_heads * alpha)
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim

        # the number of heads of Ho-Fi
        self.h_heads = num_heads - self.l_heads
        # token dimension in Ho-Fi
        self.h_dim = self.h_heads * head_dim

        self.window_size = window_size

        if self.window_size == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.l_heads = num_heads
            self.l_dim = input_dim
            self.h_heads = 0
            self.h_dim = 0

        self.scale = qk_scale or head_dim ** -0.5

        # low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.window_size != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
                self.l_q = nn.Linear(self.input_dim, self.l_dim, bias=qkv_bias)
                self.l_kv = nn.Linear(self.input_dim, self.l_dim * 2, bias=qkv_bias)
                self.l_projection = nn.Linear(self.l_dim, self.l_dim)

        # high frequence attention (Hi-Fi)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.input_dim, self.h_dim * 3, bias=qkv_bias)
            self.h_projection = nn.Linear(self.h_dim, self.h_dim)

    def HiFi(self, x):
        B, H, W, C = x.shape
        h_groups, w_groups = H // self.window_size, W // self.window_size
        total_groups = h_groups * w_groups

        x = x.reshape(B, h_groups, self.window_size, w_groups, self.window_size, C).transpose(2,3)
        # x-> (B, h_groups, w_groups, self.window_size, self.window_size, C)

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim//self.h_heads).permute(3,0,1,4,2,5)
        # self.h_qkv(x) -> (B, h_groups, w_groups, self.window_size, self.window_size, self.h_dim * 3)
        # self.reshape -> (B, h_groups * w_groups, window_size * window_size, 3, self.h_heads, self.h_dim//self.h_heads)
        # permute(3,0,1,4,2,5) -> (3, B, h_groups * w_groups, self.h_heads, window_size * window_size, self.h_dim//self.h_heads)

        q, k, v = qkv[0], qkv[1], qkv[2]
        # (B, total_groups, self.h_heads, window_size * window_size, self.h_dim//self.h_heads)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        # (B, total_groups, self.h_heads, window_size * window_size, window_size * window_size)
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_groups, w_groups, self.window_size, self.window_size, self.h_dim)
        # attn @ v -> (B, total_groups, self.h_heads, window_size * window_size, self.h_dim//self.h_heads)
        # transpose -> (B, total_groups, window_size * window_size, self.h_heads, self.h_dim//self.h_heads)

        x = attn.transpose(2, 3).reshape(B, h_groups * self.window_size, w_groups * self.window_size, self.h_dim)
        x = self.h_projection(x)
        # x -> (B, H, W, h_dim)
        return x

    def LoFi(self, x):
        B, H, W, C = x.shape

        q = self.l_q(x).reshape(B, H*W, self.l_heads, self.l_dim//self.l_heads).permute(0, 2, 1, 3)
        # self.l_q(x) -> (B, H, W, l_dim)
        # q -> (B, self.l_heads, H*W, self.l_dim//self.l_heads)

        if self.window_size > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # sr(x_) -> (B, C, H/2, W/2)
            # reshape -> (B, C, H*W/4)
            # permute -> (B, H*W/4, C)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim//self.l_heads).permute(2, 0, 3, 1, 4)
            # self.l_kv(x_) -> (B, H*W/4, l_dim *2)
            # reshape -> (B, H*W/4, 2, self.l_heads, self.l_dim//self.l_heads)
            # permute -> (2, B, self.l_heads, H*W/4, self.l_dim//self.l_heads)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim//self.l_heads).permute(2, 0,3, 1, 4)
        k, v = kv[0], kv[1]
        # (B, self.l_heads, H*W/4, self.l_dim//self.l_heads)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # (B, self.l_heads, H*W, H*W/4)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        # attn @ v -> (B, self.l_heads, H*W, self.l_dim//self.l_heads)
        x = self.l_projection(x)
        # (B, H, W, l_dim)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        if self.h_heads == 0:
            x = self.LoFi(x)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]
            return x.reshape(B, C, H, W)

        if self.l_heads == 0:
            x = self.HiFi(x)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]
            return x.reshape(B, C, H, W)

        HiFi_out = self.HiFi(x)
        LoFi_out = self.LoFi(x)
        if pad_r > 0 or pad_b > 0:
            x = torch.cat((HiFi_out[:, :H, :W, :], LoFi_out[:, :H, :W, :]), dim=-1)
        else:
            x = torch.cat((HiFi_out, LoFi_out), dim=-1)

        x = x.reshape(B, C, H, W)
        return x


class LocalFeatureEnhance(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.Local1 = DWConv(in_channels=in_channels, out_channels=out_channels)
        self.Local2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.norm1 = norm_layer(out_channels)
        self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        local1 = self.Local1(x)
        local1 = self.norm1(local1)

        local2 = self.Local2(x)

        local2 = self.norm2(local2)

        return local1 + local2


class HLAv2(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 num_heads=8,
                 window_size=2,
                 alpha=0.5,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,):
        super().__init__()

        self.HLA = HLA(input_dim=in_channels, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale, window_size=window_size
                       , alpha=alpha)

        self.LFE = LocalFeatureEnhance(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x_lfe = self.LFE(x)

        x_hla = self.HLA(x)

        return x_hla + x_lfe


class Block(nn.Module):
    def __init__(self,
                 in_dim=64,
                 out_dim=64,
                 num_heads=8,
                 window_size=2,
                 qkv_bias=False,
                 qk_scale=None,
                 alpha=0.5,
                 mlp_ratio=4.,
                 drop_path=0.,
                 act_layer=nn.ReLU6,
                 norm_layer=nn.BatchNorm2d
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.norm1 = norm_layer(out_dim)
        self.norm2 = norm_layer(out_dim)

        hidden_dim = int(mlp_ratio * in_dim)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.hlav2 = HLAv2(in_channels=in_dim, out_channels=out_dim, num_heads=num_heads, window_size=window_size,
                           alpha=alpha, drop=drop_path)
        self.mlp = Mlp(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, act_layer=act_layer, drop=drop_path)

    def forward(self, x):

        x = self.drop_path(self.hlav2(self.norm1(x)))
        x = self.drop_path(self.mlp(self.norm2(x)))

        return x


class FeatureFuse(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 eps=1e-8):
        super().__init__()

        self.pre_conv = DWConv(in_channels=in_channels, out_channels=out_channels)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = DWConvBNRelu(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinement(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 ):
        super().__init__()
        self.pre_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.post_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.DWconv = DWConv(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8

        self.Average_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_reduce_1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.channel_reduce_2 = nn.Conv2d(out_channels // 4, out_channels // 8, kernel_size=1)
        self.channel_expand_1 = nn.Conv2d(out_channels // 8, out_channels // 4, kernel_size=1)
        self.channel_expand_2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)
        self.act_layer = nn.ReLU6()
        self.norm = nn.BatchNorm2d(out_channels)

    def channels_refinement(self, x):
        x_ = self.Average_pool(x)
        x_ = self.channel_reduce_2(self.channel_reduce_1(x_))
        x_ = self.channel_expand_2(self.channel_expand_1(x_))

        x_ = self.softmax(x_)
        x = x_ * x

        return x

    def spatial_refinement(self, x):
        x_ = self.DWconv(x)
        x_ = self.conv(x_)

        x_ = self.softmax(x_)
        x = x_ * x

        return x

    def shortcut(self, x):
        x = self.pre_conv(x)
        x = self.norm(x)

        return x

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.pre_conv(x)
        shortcut = self.shortcut(x)

        cr = self.channels_refinement(x)
        sr = self.spatial_refinement(x)

        x = cr + sr
        x = self.conv(self.norm(self.DWconv(x)))
        x = x + shortcut

        x = self.conv(self.act_layer(x))

        return x


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64,128,256,512),
                 decoder_channels=64,
                 dropout=0.1,
                 num_classes=2):
        super(Decoder, self).__init__()

        self.pre_conv = nn.Conv2d(encoder_channels[-1], decoder_channels, kernel_size=1)
        self.b1 = Block(in_dim=decoder_channels, num_heads=16, window_size=8)

        self.b2 = Block(in_dim=decoder_channels, num_heads=16, window_size=8)
        self.p2 = FeatureFuse(encoder_channels[-2], decoder_channels)

        self.b3 = Block(in_dim=decoder_channels, num_heads=16, window_size=8)
        self.p3 = FeatureFuse(encoder_channels[-3], decoder_channels)

        self.pre_conv_refine = nn.Conv2d(decoder_channels, decoder_channels,kernel_size=1,padding=0)

        self.p4 = FeatureRefinement(in_channels=decoder_channels, out_channels=decoder_channels)

        self.segmentation_head = nn.Sequential(DWConvBNRelu(in_channels=decoder_channels, out_channels=decoder_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               DWConv(decoder_channels, num_classes, kernel_size=1, padding=0))

        self.init_weight()

    def forward(self, res1, res2, res3, res4):
        x = self.b1(self.pre_conv(res4))  # HLAv2

        x = self.p2(x, res3)   # WF

        x = self.b2(x)  # HLAv2

        x = self.p3(x, res2)  # WF

        x = self.b3(x)  # HLAv2

        x = self.p4(x, res1)  # FRH

        x = self.segmentation_head(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class NewV2(nn.Module):
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NewV2().to(device)
summary = torchsummary.summary(model, (3,1024,1024))
print(summary)

