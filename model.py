import math

from matplotlib.colors import to_rgb
import config
import numpy as np
from typing import Optional, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
from blurpool import BlurPool2D

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def Equalized(*shape:int, pre_bias:float=0.0, gain:float=1, lrm:float=1):
    if len(shape) == 4:
        order = 1, 0, 2, 3
    elif len(shape) == 2:
        order = 1, 0
    else:
        ValueError('Invalid equalized shape!')
    shape = [shape[i] for i in order]
    mean = torch.zeros(shape)
    std = torch.ones(shape) / lrm
    w_scaler = gain * lrm / math.sqrt(np.prod(shape[1:]))
    weight = nn.Parameter(w_scaler * torch.normal(mean, std)).to(config.DEVICE)
    bias = nn.Parameter(lrm * torch.ones(shape[0]) * pre_bias).to(config.DEVICE)
    return weight, bias

class EqualizedLinear(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias:float=0.0):
        super().__init__()
        w, b = Equalized(
            in_features, out_features, 
            pre_bias=bias
        )
        self.weight = w
        self.bias = b
    def forward(self, input:torch.Tensor):
        return F.linear(input, self.weight, self.bias)

class EqualizedConv2d(nn.Module):
    def __init__(
        self, 
        in_channel:int, 
        out_channel:int, 
        kernel_size:int, 
        padding:int
    ):
        super().__init__()
        w, b = Equalized(
            in_channel, out_channel, kernel_size, kernel_size
        )
        self.weight = w
        self.bias = b
        self.stride = (kernel_size - 1) // 2
        self.padding = padding
    def forward(self, input:torch.Tensor):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)

class Mapping(nn.Module):
    def __init__(self, features:int, n_layers:int=8):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                EqualizedLinear(features, features)
            )
            layers.append(
                nn.LeakyReLU(0.02, inplace=True)
            )
        self.block = nn.Sequential(*layers)
    def forward(self, input:torch.Tensor):
        res = F.normalize(input, dim=1)
        return self.block(res)

class Conv2dDemodulate(nn.Module):
    def __init__(
        self, 
        in_channel:int, 
        out_channel:int, 
        kernel_size:int, 
        stride:int, 
        demodulate:bool=True, 
        eps:float=1e-8
    ):
        super().__init__()
        self.weight, _ = Equalized(
            in_channel, out_channel, kernel_size, kernel_size
        )
        self.stride = stride
        self.padding = (kernel_size  - 1) // 2
        self.demodulate = demodulate
        self.eps = eps
    
    def forward(self, input:torch.Tensor, style:torch.Tensor):
        # input: batch_size, in_features, H, W
        # style: batch_size, in_features
        # weight: out_features, in_features, kernel, kernel
        B, I_C, H, W = input.shape
        # weights: 1, out_features, in_features, kernel, kernel
        weights = self.weight.unsqueeze(0)
        style = style.view(B, 1, I_C, 1, 1)
        weights = weights * style
        if self.demodulate:
            sigma = torch.sqrt(self.eps + torch.sum((weights ** 2), dim=(2, 3, 4), keepdim=True))
            weights = weights / sigma
        # weights: batch_size, out_features, in_features, kernel, kernel
        _, _, *w_shape = weights.shape
        # weight: batch_size  * out_features, in_features, kernel, kernel
        weights = weights.view(-1, *w_shape)
        # res: 1, batch_size * in_features, H, W
        res = input.view(1, -1, H, W)
        res = F.conv2d(res, weights, stride=self.stride, padding=self.padding, groups=B)
        res = res.view(B,-1, H, W)
        return res

class StyleBlock(nn.Module):
    def __init__(
        self, 
        w_dim:int, 
        in_channel:int, 
        out_channel:int, 
        kernel_size:int, 
        stride:int):
        super().__init__()
        self.to_style = EqualizedLinear(w_dim, in_channel)
        self.conv = Conv2dDemodulate(in_channel, out_channel, kernel_size, stride)
        self.bias = nn.Parameter(torch.zeros(out_channel))
        self.raise_noise = nn.Parameter(torch.zeros(1))
    def forward(self, input:torch.Tensor, w:torch.Tensor, noise=Optional[torch.Tensor]):
        style = self.to_style(w)
        res = self.conv(input, style) + self.bias[None, :, None, None]
        if noise is not None:
            res = res + self.raise_noise[None, :, None, None] * noise
        return F.leaky_relu(res, negative_slope=0.2)

class toRGB(nn.Module):
    # change style to RGB
    def __init__(self, w_dim:int, features:int):
        super().__init__()
        self.to_style = EqualizedLinear(w_dim, features, bias=1.0)
        self.conv = Conv2dDemodulate(features, 3, 3, 1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3)) 
    def forward(self, input:torch.Tensor, w:torch.Tensor):
        style = self.to_style(w)
        res = self.conv(input, style) + self.bias[None, :, None, None]
        return F.leaky_relu(res, negative_slope=0.2)

class DownSample(nn.Module):
    def __init__(self, scaler:int=2):
        super().__init__()
        self.blur = BlurPool2D()
        self.scaler = scaler
    def forward(self, input:torch.Tensor):
        res = self.blur(input)
        res = F.interpolate(res, scale_factor=1/self.scaler, align_corners='bilinear', recompute_scale_factor=True)
        return res

class UpSample(nn.Module):
    def __init__(self, scaler:int=2):
        super().__init__()
        self.blur = BlurPool2D()
        self.up_sample = nn.Upsample(scaler_factor=scaler)
    def forward(self, input:torch.Tensor):
        res = self.blur(input)
        res = self.up_sample(res)
        return res

class SkipBlock(nn.Module):
    def __init__(self, w_dim:int, in_features:int, out_features:int):
        self.style_block1 = StyleBlock(w_dim, in_features, out_features)
        self.style_block2 = StyleBlock(w_dim, out_features, out_features)
        self.to_rgb = toRGB(w_dim, out_features)
    
    def forward(self, input:torch.Tensor, w:torch.Tensor, noise:Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        res = self.style_block1(input, w, noise[0])
        res = self.style_block2(res, w, noise[1])
        rgb = self.to_rgb(res, w)
        return res, rgb

class Generator(nn.Module):
    def __init__(self, channels=config.CHANNELS, w_dim:int=config.W_DIM, z_dim:int=config.Z_DIM):
        '''
        channels = {
            4: 512,
            8: 512, 
            16: 512,
            32: 512,
            64: 256 * config.CHANNEL_MULTI,
            128: 128 * config.CHANNEL_MULTI,
            256: 64 * config.CHANNEL_MULTI,
            512: 32 * config.CHANNEL_MULTI,
            1024: 16 * config.CHANNEL_MULTI 
        }
        '''
        
        super().__init__()
        self.initial_const = torch.rand(1, 512, 4, 4) 
        n_layers = len(channels)
        self.style_block = StyleBlock(w_dim, channels[4], channels[4])
        self.toRGB = toRGB(w_dim, channels[4])
        self.blocks = []
        self.up_sample = UpSample()
        for i in range(2, n_layers+1):
            skip_block = SkipBlock(w_dim, channels[2**i], channels[2**(i+1)])
            self.blocks.append(skip_block)
        
    def forward(self, w:torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        # each w will learn a style => w (n_blocks, batch_size, w_dim)
        # so is input_noise(n_blocks, batch_size, )
        batch_size = w.shape[1]
        c = self.initial_const[batch_size, :, :, :]
        res = self.style_block(c, w[0])
        rgb = self.toRGB(res, w[0])
        for i in range(1, len(self.block)):
            res = self.block[i][0](res)
            res, rgb_new = self.block[i][1](res, w[i], input_noise)
            rgb = self.up_sample(rgb) + rgb_new
        return rgb

class ResidualBlock(nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super().__init__()
        self.down_sample = DownSample()
        self.conv = EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1)
        self.scale = 1 / math.sqrt(2)
    def forward(self, input):
        down = self.down_sample(input)
        conv = self.conv(input)
        return (down + conv) * self.scale 

class Discriminator(nn.Module):
    def __init__(self, channels=config.CHANNELS):
        super().__init__()
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, channels[1024], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.blocks = []
        n_layers = len(channels)
        for i in range(n_layers, 1, -1):
            block = ResidualBlock(channels[2**(i+1)], channels[2**(i)])
            self.blocks.append(block)
    def forward(self):
        pass