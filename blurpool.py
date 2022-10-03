import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BlurPool2D(nn.Module):
    def __init__(self, pad_type:str='reflect', filt_size:int=4, stride:int=1, pad_off:int=0):
        super(BlurPool2D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.pad = self.get_pad_layer(pad_type)(self.pad_sizes)

    def init_filter(self, channels):
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((channels,1,1,1)))

    def forward(self, input:torch.Tensor):
        channels = input.shape[1]
        self.init_filter(channels)
        if(self.filt_size==1):
            if(self.pad_off==0):
                return input[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(input)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(input), self.filt, stride=self.stride, groups=channels)

    def get_pad_layer(self, pad_type:str):
        if(pad_type in ['refl','reflect']):
            PadLayer = nn.ReflectionPad2d
        elif(pad_type in ['repl','replicate']):
            PadLayer = nn.ReplicationPad2d
        elif(pad_type=='zero'):
            PadLayer = nn.ZeroPad2d
        else:
            print('Pad type [%s] not recognized'%pad_type)
        return PadLayer

class BlurPool1D(nn.Module):
    def __init__(self, pad_type:str='reflect', filt_size:int=3, stride:int=1, pad_off:int=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.pad = self.get_pad_layer_1d(pad_type)(self.pad_sizes)

    def init_filter(self, channels):
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((channels, 1, 1)))

    def forward(self, input):
        channels = input.shape[1]
        self.init_filter(channels)
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return input[:, :, ::self.stride]
            else:
                return self.pad(input)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(input), self.filt, stride=self.stride, groups=channels)

    def get_pad_layer_1d(self, pad_type:str):
        if(pad_type in ['refl', 'reflect']):
            PadLayer = nn.ReflectionPad1d
        elif(pad_type in ['repl', 'replicate']):
            PadLayer = nn.ReplicationPad1d
        elif(pad_type == 'zero'):
            PadLayer = nn.ZeroPad1d
        else:
            print('Pad type [%s] not recognized' % pad_type)
        return PadLayer