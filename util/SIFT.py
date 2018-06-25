import torch
import torch.nn as nn
import math
import numpy as np


def L2norm(x):
    norm = torch.sqrt(torch.abs(torch.sum(x * x, dim=1)) + 1e-10)
    return x / norm.unsqueeze(1).expand_as(x)

def getPoolingKernel(kernlen=10):
    step = 1 / (kernlen // 2)
    x_coef = np.arange(step/2, 1, step)
    xc2 = np.hstack([x_coef, [1], x_coef[::-1]])
    kernel = np.outer(xc2.T, xc2)
    return np.maximum(0, kernel)

def get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins):
    bin_weight_stride = round(2 * math.floor(patch_size / 2) / float(num_spatial_bins + 1))
    bin_weight_kernel_size = 2 * bin_weight_stride - 1
    return int(bin_weight_kernel_size), int(bin_weight_stride)

def CircularGaussKernel(kernlen=10):
    halfSize = kernlen / 2
    r2 = float(halfSize**2)
    sigma2 = 0.9 * r2
    kernel = np.zeros((kernlen,kernlen))
    for y in range(kernlen):
        for x in range(kernlen):
            disq = (y - halfSize)**2 + (x - halfSize)**2
            kernel[y,x] = math.exp(-disq / sigma2)  if disq < r2  else 0.
    return kernel.astype(np.float32)

class SIFTNet(nn.Module):
    def __init__(self, patch_size=30, num_ang_bins=8, num_spatial_bins=4, clipval=0.2):
        super(SIFTNet, self).__init__()
        self.bin_weight_kernel_size, self.bin_weight_stride = get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins)
        self.num_ang_bins, self.num_spatial_bins = num_ang_bins, num_spatial_bins
        self.clipval = clipval
        self.gk = torch.from_numpy(CircularGaussKernel(patch_size)).cuda()

        self.gx = nn.Conv2d(1, 1, kernel_size=(1,3), bias=False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[-1, 0, 1]]]], dtype=np.float32))
        self.gy = nn.Conv2d(1, 1, kernel_size=(3,1), bias=False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[-1], [0], [1]]]], dtype=np.float32))

        self.pk = nn.Conv2d(1, 1, kernel_size=(self.bin_weight_kernel_size, self.bin_weight_kernel_size),
                            stride=(self.bin_weight_stride, self.bin_weight_stride), bias = False)
        nw = getPoolingKernel(kernel_size = self.bin_weight_kernel_size)
        new_weights = np.array(nw.reshape((1, 1, self.bin_weight_kernel_size, self.bin_weight_kernel_size)))
        self.pk.weight.data = torch.from_numpy(new_weights.astype(np.float32))

    def forward(self, x):
        gx = self.gx(nn.functional.pad(x, (1,1,0, 0), 'replicate'))
        gy = self.gy(nn.functional.pad(x, (0,0, 1,1), 'replicate'))
        mag = torch.sqrt(gx * gx + gy * gy + 1e-10)
        mag = mag * self.gk.expand_as(mag)
        ori = torch.atan2(gy, gx + 1e-8)

        o_big = (ori + 2*math.pi) / (2*math.pi) * float(self.num_ang_bins)
        bo0_big = torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big = bo0_big %  self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1 - wo1_big) * mag
        wo1_big = wo1_big * mag

        ang_bins = [self.pk((bo0_big == i) * wo0_big + (bo1_big == i) * wo1_big) \
                    for i in range(self.num_ang_bins)]
        ang_bins = torch.cat(ang_bins, dim=1)
        ang_bins = ang_bins.view(ang_bins.size(0), -1)
        ang_bins = torch.clamp(L2norm(ang_bins), 0., float(self.clipval))
        return L2norm(ang_bins)
