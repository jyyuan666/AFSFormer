import torch.nn as nn
import torch.nn.functional as F
import torch

from timm.models.layers import DropPath
import timm


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


class Block(nn.Module):
    def __init__(self,
                 dim=64,
                 num_heads=16,
                 qkv_bias=False,
                 qk_scale=None,
                 window_size=8,
                 alpha=0.5,
                 mlp_ratio=4.,
                 drop=0.,
                 atnn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.ReLU6,
                 norm_layer=nn.BatchNorm2d,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = HLA(input_dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, window_size=window_size, alpha=alpha)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(in_dim=dim, hidden_dim=mlp_hidden_dim, out_dim=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 fuse_channels,
                 output_channels=64,
                 last_block=False
                 ):
        super().__init__()

        self.flag = last_block

        self.in_channels = input_channels
        self.out_channels = output_channels
        self.fuse_channel = fuse_channels


        self.preconv = nn.Conv2d(self.fuse_channel, self.out_channels, kernel_size=1)
        self.activate = nn.ReLU6()
        self.norm = nn.BatchNorm2d(self.out_channels)

    def forward(self, x, res):

        if self.flag == False:
            res = self.preconv(res)

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

        self.preconv = nn.Conv2d(encoder_channels[-1], decoder_channels, kernel_size=1)

        self.attnb1 = Block(dim=decoder_channels, num_heads=16, window_size=8)
        self.b1 = DecoderBlock(input_channels=encoder_channels[-1],
                               fuse_channels=encoder_channels[2])

        self.attnb2 = Block(dim=decoder_channels, num_heads=16, window_size=8)
        self.b2 = DecoderBlock(input_channels=encoder_channels[2],
                               fuse_channels=encoder_channels[1])

        self.attnb3 = Block(dim=decoder_channels, num_heads=16, window_size=8)
        self.b3 = DecoderBlock(input_channels=encoder_channels[1],
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
        res4 = self.attnb1(self.preconv(res4))
        out = self.b1(res4, res3)

        out = self.attnb2(out)
        out = self.b2(out, res2)

        out = self.attnb3(out)
        out = self.b3(out, res1)

        out = self.refine(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

        out = self.segmentation_head(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Baseline_with_HLA(nn.Module):
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



