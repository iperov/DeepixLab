import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool(nn.Module):
    def __init__(self, in_ch, kernel_size=3, stride=2):
        super().__init__()
        self._in_ch = in_ch
        self._stride = stride
        self._pad_sizes = (int(1.*(kernel_size-1)/2), int(np.ceil(1.*(kernel_size-1)/2)),) * 2

        if(kernel_size==2):
            kernel = np.array([1., 1.])
        elif(kernel_size==3):
            kernel = np.array([1., 2., 1.])
        elif(kernel_size==4):
            kernel = np.array([1., 3., 3., 1.])
        elif(kernel_size==5):
            kernel = np.array([1., 4., 6., 4., 1.])
        elif(kernel_size==6):
            kernel = np.array([1., 5., 10., 10., 5., 1.])
        elif(kernel_size==7):
            kernel = np.array([1., 6., 15., 20., 15., 6., 1.])

        kernel = kernel[:,None]*kernel[None,:]
        kernel /= kernel.sum()
        kernel = np.tile(kernel[None,None,:,:], (in_ch,1,1,1) ).astype(np.float32)

        self.register_buffer('_kernel', torch.tensor(kernel))

    def forward(self, inp):
        x = F.pad(inp, self._pad_sizes, 'constant', value=0)

        return F.conv2d(x, self._kernel, stride=self._stride, groups=self._in_ch)