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


class Bi_path_attention(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 qkv_bias=False,
                 qk_scale=None,
                 window_size=2,
                 num_heads=8):
        super().__init__()

        assert in_dim % num_heads == 0, print('input dim must divided by num_heads')
        head_dim = int(in_dim / num_heads)

        self.head_dim = head_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.linear_qkv = nn.Linear(self.in_dim, self.in_dim * 3)
        self.projection = nn.Linear(self.in_dim, self.in_dim)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

    def Global_path(self, x):
        B, H, W, C = x.shape
        h_groups = H // self.window_size
        w_groups = W // self.window_size
        total_groups = h_groups * w_groups

        x = x.reshape(B, h_groups, self.window_size, w_groups, self.window_size, C).transpose(2, 3)
        # x->(B, h_groups, w_groups, self.wz, self.wz, C)
        qkv = self.linear_qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, self.in_dim // self.num_heads).permute(3,0,1,4,2,5)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_groups, w_groups, self.window_size, self.window_size, C)

        x = attn.transpose(2, 3).reshape(B, h_groups * self.window_size, w_groups * self.window_size, C)
        x = self.projection(x)

        return x

    def Local_path(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, C, H, W)
        x = self.avg_pool(x)
        b, c, h, w = x.shape
        x = x.reshape(b, h, w, c)

        h_groups = h // self.window_size
        w_groups = w // self.window_size
        total_groups = h_groups * w_groups

        x = x.reshape(B, h_groups, self.window_size, w_groups, self.window_size, C)
        qkv = self.linear_qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, self.in_dim // self.num_heads).permute(3,0,1,4,2,5)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_groups, w_groups, self.window_size, self.window_size, C)

        x = attn.transpose(2, 3).reshape(B, h_groups * self.window_size, w_groups * self.window_size, C)

        b_0, h_0, w_0, c_0 = x.shape
        x = x.reshape(b_0, c_0, h_0, w_0)
        x_x = self.attn_x(F.pad(x, pad=(0, 0, 1, 0), mode='reflect'))
        x_y = self.attn_y(F.pad(x, pad=(0, 1, 0, 0), mode='reflect'))

        x = x_x + x_y

        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = x.reshape(B, H, W, C)


        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, H, W, C)
        Global = self.Global_path(x)
        Local = self.Local_path(x)

        return (Global + Local).permute(0, 3, 1, 2)



class LocalFeatureEnhance(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.Local1 = DWConvBNRelu(in_channels=in_channels, out_channels=out_channels)
        self.Local2 = DWConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)


    def forward(self, x):
        local1 = self.Local1(x)


        local2 = self.Local2(x)



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

        self.bi_attn = Bi_path_attention(in_dim=in_channels,out_dim=in_channels, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale, window_size=window_size
                       )

        self.LFE = LocalFeatureEnhance(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x_lfe = self.LFE(x)

        x_hla = self.bi_attn(x)

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


class SpatialRefine(nn.Module):
    def __init__(self,
                 in_dim=64,
                 out_dim=64):
        super().__init__()

        self.dwconv7 = nn.Conv2d(in_channels=out_dim // 2, out_channels=out_dim // 2, kernel_size=7, padding=3,
                                 groups=out_dim // 2)
        self.dwconv5 = nn.Conv2d(in_channels=out_dim // 2, out_channels=out_dim // 2, kernel_size=5, padding=2,
                                 groups=out_dim // 2)

        self.dwconv3 = nn.Conv2d(in_channels=out_dim // 2, out_channels=out_dim // 2, kernel_size=3, padding=1,
                                 groups=out_dim // 2)
        self.conv1 = nn.Conv2d(in_channels=out_dim // 2, out_channels=out_dim // 2, kernel_size=1)

    def forward(self, x):

        x_1 = self.dwconv7(x)
        x_1 = self.dwconv5(x_1)

        x_2 = self.dwconv3(x)
        x_2 = self.conv1(x_2)

        return torch.cat((x_1, x_2), dim=1)


class ChannelRefine(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.reduce1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1)
        self.reduce2 = nn.Conv2d(out_channels//2, out_channels//4, kernel_size=5, padding=2)

        self.expand1 = nn.Conv2d(out_channels//4, out_channels//2, kernel_size=5, padding=2)
        self.expand2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.reduce2(self.reduce1(x))

        x = self.expand2(self.expand1(x))

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
                 in_dim=64,
                 out_dim=64,
                 eps=1e-8
                 ):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8

        self.pre_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1, groups=in_dim)

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.cr1 = ChannelRefine(in_channels=32, out_channels=32)
        self.cr2 = ChannelRefine(in_channels=16, out_channels=16)
        self.cr3 = ChannelRefine(in_channels=16, out_channels=16)

        self.sr = SpatialRefine(in_dim=in_dim, out_dim=out_dim)

        self.conv_cr1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim//2, kernel_size=1)
        self.conv_cr2 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim//4, kernel_size=1)
        self.conv_cr3 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim//4, kernel_size=1)
        self.proj = DWConvBNRelu(in_channels=in_dim, out_channels=out_dim)


        self.softmax = nn.Softmax(dim=1)
        self.act_layer = nn.ReLU6()
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x, res):

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        x = self.pre_conv(x)
        x = fuse_weights[0] * x + fuse_weights[1] * res
        x_1 = self.avg(x)

        x_cr_1 = self.conv_cr1(x_1)
        x_cr_1 = self.cr1(x_cr_1)

        x_cr_2 = self.conv_cr2(x_1)
        x_cr_2 = self.cr2(x_cr_2)

        x_cr_3 = self.conv_cr3(x_1)
        x_cr_3 = self.cr3(x_cr_3)

        x_cr = torch.cat((x_cr_1, x_cr_2, x_cr_3), dim=1)
        x_cr_out = self.softmax(x_cr) * x

        x_sr = self.conv_cr1(x)

        x_sr = self.sr(x_sr)

        x_out = x_cr_out + x_sr

        x = self.norm(x) + self.proj(x_out)
        x = self.act_layer(x)

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

        self.p4 = FeatureRefinement(in_dim=decoder_channels, out_dim=decoder_channels)

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
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.segmentation_head(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class NewV5(nn.Module):
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
'''
x = torch.randn(1, 3, 1024, 1024)
FireFormer = NewV5()
y = FireFormer(x)
print(y.shape)'''

'''device = torch.device('cuda')
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
print(mean_syn)'''

from torchstat import stat

print('==> Building model..')
model = NewV5()

stat(model, (3, 1024, 1024))

