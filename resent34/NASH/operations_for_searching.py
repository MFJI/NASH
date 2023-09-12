import torch
import torch.nn as nn
from bi_real import *

# during all the operation, avg and max pool layer may be not used due to that pool layer can only take care of the stride, but can not take care of 
# the difference between in_channels and out_channels

OPS = {
  'none' :         lambda in_channels, out_channels, stride, first, affine: Zero(in_channels, out_channels, stride, first, affine),
  'avg_pool_3x3' : lambda in_channels, out_channels, stride, first, affine: avgpool2d(in_channels, out_channels, stride, first, affine),
  'max_pool_3x3' : lambda in_channels, out_channels, stride, first, affine: maxpool2d(in_channels, out_channels, stride, first, affine),
  'skip_connect' : lambda in_channels, out_channels, stride, first, affine: Identity() \
                              if stride == 1 else Factorized(in_channels, out_channels, stride, first, affine),
  'conv_1x1' :     lambda in_channels, out_channels, stride, first, affine: conv2d_bn_b(in_channels, out_channels, 1, stride, 0, bias=False, affine=affine),
  'conv_3x3' :     lambda in_channels, out_channels, stride, first, affine: conv2d_bn_b(in_channels, out_channels, 3, stride, 1, bias=False, affine=affine),
  'conv_5x5' :     lambda in_channels, out_channels, stride, first, affine: conv2d_bn_b(in_channels, out_channels, 5, stride, 2, bias=False, affine=affine),
  'dil_conv_1x1' : lambda in_channels, out_channels, stride, first, affine: conv2d_bn_b(in_channels, out_channels, 1,\
                              stride, 0, 2, bias=False, affine=affine),
  'dil_conv_3x3' : lambda in_channels, out_channels, stride, first, affine: conv2d_bn_b(in_channels, out_channels, 3,\
                              stride, 2, 2, bias=False, affine=affine),
  'dil_conv_5x5' : lambda in_channels, out_channels, stride, first, affine: conv2d_bn_b(in_channels, out_channels, 5,\
                              stride, 4, 2, bias=False, affine=affine),
}


class conv2d_bn_b (nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, affine=True):
    super(conv2d_bn_b, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.stride = stride
    self.affine = affine
    self.weight_binarize = weight_binarize_fn()
    self.activation_binarize = activation_binarize_fn()
    self.bn = nn.BatchNorm2d(out_channels, affine=affine)
  def forward(self, input):
    b_w = self.weight
    b_a = torch.tanh(input)
    #b_w = self.weight_binarize(self.weight)
    #b_a = self.activation_binarize(input)
    full_conv2d = F.conv2d(b_a, b_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
    after_bn = self.bn(full_conv2d)
    return after_bn



class maxpool2d(nn.Module):

  def __init__(self, in_channels, out_channels, stride, first, affine=True):
    super(maxpool2d, self).__init__()
    self.stride = stride
    self.first = first
    self.maxpool2d = nn.MaxPool2d(3, stride=self.stride, padding=1)
    self.bn = nn.BatchNorm2d(out_channels, affine=affine)

  def forward(self, x):
    if self.stride == 1:
      out = self.maxpool2d(x)
    elif self.stride == 2:
      out = self.maxpool2d(x)
      out = torch.cat((out, out), dim=1)
    out = self.bn(out)
    return out


class avgpool2d(nn.Module):

  def __init__(self, in_channels, out_channels, stride, first, affine=True):
    super(avgpool2d, self).__init__()
    self.stride = stride
    self.first = first
    self.avgpool2d = nn.AvgPool2d(3, stride=self.stride, padding=1, count_include_pad=False)
    self.bn = nn.BatchNorm2d(out_channels, affine=affine)

  def forward(self, x):
    if self.stride == 1:
      out = self.avgpool2d(x)
    elif self.stride == 2:
      out = self.avgpool2d(x)
      out = torch.cat((out, out), dim=1)
    out = self.bn(out)
    return out


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, in_channels, out_channels, stride, first, affine):
    super(Zero, self).__init__()
    self.stride = stride
    self.first = first

  def forward(self, x):
    if self.stride == 1:
      zero_out = x.mul(0.)
    elif self.stride == 2:
      zero_out = x[:, :, ::self.stride, ::self.stride].mul(0.)
      zero_out = torch.cat((zero_out, zero_out), dim=1)
    return zero_out


class Factorized(nn.Module):

  def __init__(self, in_channels, out_channels, stride, first, affine):
    super(Factorized, self).__init__()
    self.first = first
    self.stride = stride
    self.conv_1 = conv2d_b(in_channels, out_channels//2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = conv2d_b(in_channels, out_channels//2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(out_channels, affine=affine)

  def forward(self, x):
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

