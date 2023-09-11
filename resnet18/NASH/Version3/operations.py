import torch
import torch.nn as nn
import torch.nn.functional as F
#from bi_real import *

import brevitas.nn as qnn

from brevitas.quant_tensor import QuantTensor

#from common import CommonWeightQuant, CommonActQuant


import numpy as np

from brevitas.core.quant import QuantType
from brevitas.quant import IntBias
from brevitas.core.stats import StatsOp
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType


return_quant = True
bit_width_weight = 4
bit_width_act = 2
bit_width_pool = 4
SCALING_MIN_VAL=2e-16


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
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    #self.conv = qnn.QuantConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=self.stride, bias=bias, padding=self.padding, dilation=self.dilation, groups=self.groups, weight_bit_width=bit_width_weight, return_quant_tensor=return_quant)
    self.conv = qnn.QuantConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=self.stride, bias=bias, padding=self.padding, dilation=self.dilation,   groups=self.groups, 
                                            weight_quant_type=QuantType.BINARY,
                                            weight_scaling_stats_op=StatsOp.AVE,
                                            weight_scaling_per_output_channel=True,
                                            weight_bit_width=1,
                                            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                            weight_scaling_impl_type=ScalingImplType.STATS,
                                            weight_scaling_min_val=SCALING_MIN_VAL,
                                                    return_quant_tensor=return_quant)
    #self.relu = nn.ReLU(inplace=False)
    #self.relu = qnn.QuantReLU(bit_width=bit_width_act, return_quant_tensor=return_quant)
    
    #self.weight_binarize = weight_binarize_fn()
    #self.activation_binarize = activation_binarize_fn()
    self.bn = nn.BatchNorm2d(out_channels)
  def forward(self, input):
#    b_w = self.weight
#    b_a = torch.tanh(input)
    #b_w = self.weight_binarize(self.weight)
    #b_a = self.activation_binarize(input)
    #full_conv2d = F.conv2d(b_a, b_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
    full_conv2d = self.conv(input)
    #full_conv2d = self.relu(full_conv2d)
    after_bn = self.bn(full_conv2d)
    #after_bn = self.relu(after_bn)
    return after_bn



class maxpool2d(nn.Module):

  def __init__(self, in_channels, out_channels, stride, first, affine=True):
    super(maxpool2d, self).__init__()
    self.stride = stride
    self.first = first
    self.maxpool2d = nn.MaxPool2d(3, stride=self.stride, padding=1)
    self.bn = nn.BatchNorm2d(out_channels)
    #self.bn2 = nn.BatchNorm2d(out_channels//2)
    #self.relu = qnn.QuantReLU(bit_width=bit_width_act, return_quant_tensor=return_quant)
    #self.relu1 = qnn.QuantReLU(bit_width=1, return_quant_tensor=return_quant)
    #self.relu2 = qnn.QuantReLU(bit_width=1, return_quant_tensor=return_quant)
    #self.id = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=1, return_quant_tensor=return_quant)
    #self.quant_identity = qnn.QuantIdentity(act_quant=CommonActQuant, bit_width=1, return_quant_tensor=return_quant)
  def forward(self, x):
    if self.stride == 1:
      out = self.maxpool2d(x)
    elif self.stride == 2:
      out = self.maxpool2d(x)
      #out2 = self.maxpool2d(x)
      #out = QuantTensor.cat((out, out), dim=1)
      #out = self.bn2(out)
      #out = self.relu(out)
      #out = out.permute(0,2,3,1)
      #out = torch.cat((out, out), dim=3)
      #out = out.permute(0,3,1,2)
      #out = torch.cat((out, out), dim=1)
      
      
      #out = out.permute(0,2,3,1)
      #out = torch.cat((out, out), dim=3)
      #out = out.permute(0,3,1,2)

      #out = self.relu(x)
      #out = torch.transpose(out,0,2,3,1)
      #out = self.relu(out)
      #out = out.permute(0,2,3,1)
      #out2 = out.permute(0,2,3,1)
      #out = out.contiguous()
      #out1 = self.id(out)
      #out2 = self.id(out)
      out = torch.cat((out, out), dim=1)
      #out = QuantTensor.cat([out1, out2], dim=3)
      #out = out.permute(0,3,1,2)
      #out = out.contiguous()
      #out = torch.transpose(out,0,3,1,2)
      #out = self.maxpool2d(out)    
    out = self.bn(out)

    return out


class avgpool2d(nn.Module):

  def __init__(self, in_channels, out_channels, stride, first, affine=True):
    super(avgpool2d, self).__init__()
    self.stride = stride
    self.first = first
    self.avgpool2d = nn.AvgPool2d(3, stride=self.stride, padding=1, count_include_pad=False)
    #self.avgpool2d = qnn.QuantAvgPool2d(kernel_size=3, stride=self.stride, padding=2, count_include_pad=False, bit_width=bit_width_pool)
    #self.avgpool2d = qnn.QuantAvgPool2d(kernel_size=3, stride=self.stride, bit_width=bit_width_pool)
    self.bn = nn.BatchNorm2d(out_channels)
    #self.relu = qnn.QuantReLU(inplace=False, bit_width=bit_width_act, return_quant_tensor=return_quant)
  def forward(self, x):
    if self.stride == 1:
      out = self.avgpool2d(x)
    elif self.stride == 2:
      out = self.avgpool2d(x)
      out = torch.cat((out, out), dim=1)
    out = self.bn(out)
    #out = self.relu(out)
    return out


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()
    #self.relu = qnn.QuantReLU(inplace=False, bit_width=bit_width_act, return_quant_tensor=return_quant)
  def forward(self, x):
    #return self.relu(x)
    return x


class Zero(nn.Module):

  def __init__(self, in_channels, out_channels, stride, first, affine):
    super(Zero, self).__init__()
    self.stride = stride
    self.first = first
    #self.relu = qnn.QuantReLU(inplace=False, bit_width=bit_width_act, return_quant_tensor=return_quant)

  def forward(self, x):
    if self.stride == 1:
      #zero_out = x.mul(0.)
      zero_out = x.__mul__(0.)
    elif self.stride == 2:
      #zero_out = x[:, :, ::self.stride, ::self.stride].mul(0.)
      zero_out = x[:, :, ::self.stride, ::self.stride]
      #print(x)
      #zero_out = x.narrow(dim=2,start=0,length=x.shape[2]//2)
      #print(zero_out)
      #zero_out = zero_out.narrow(dim=3,start=0,length=zero_out.shape[3]//2)
      #zero_out[0] = x[0][:, :, ::self.stride, ::self.stride]
      #print(zero_out)
      #zero_out = x[:, :, ::self.stride, ::self.stride]
      zero_out = zero_out*0
      #zero_out = zero_out.permute(0,2,3,1)
      zero_out = torch.cat((zero_out, zero_out), dim=1)
      #zero_out = zero_out.permute(0,3,1,2)
    #zero_out = self.relu(zero_out)
    return zero_out


class Factorized(nn.Module):

  def __init__(self, in_channels, out_channels, stride, first, affine):
    super(Factorized, self).__init__()
    self.first = first
    self.stride = stride
    #self.relu = nn.ReLU(inplace=False)
    #self.relu = qnn.QuantReLU(inplace=False, bit_width=bit_width_act, return_quant_tensor=return_quant)
    #self.conv_1 = conv2d_b(in_channels, out_channels//2, 1, stride=2, padding=0, bias=False)
    self.conv_1 = qnn.QuantConv2d(in_channels, out_channels//2, 1, stride=2, padding=0, bias=False, weight_bit_width=bit_width_weight, return_quant_tensor=return_quant)
    #self.conv_2 = conv2d_b(in_channels, out_channels//2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = qnn.QuantConv2d(in_channels, out_channels//2, 1, stride=2, padding=0, bias=False, weight_bit_width=bit_width_weight, return_quant_tensor=return_quant)
    self.bn = nn.BatchNorm2d(out_channels)
  def forward(self, x):
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    #out = self.relu(out)
    out = self.bn(out)
    #out = self.relu(out)
    return out

