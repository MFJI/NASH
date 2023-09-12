import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def binarize_weight ():
  class b_weight (torch.autograd.Function):
    @staticmethod
    def forward (ctx, input):
      out = torch.sign (input)
      return out

    @staticmethod
    def backward (ctx, grad_output):
      grad_input = grad_output.clone ()
      return grad_input

  return b_weight().apply

def binarize_activation ():
  class b_activation (torch.autograd.Function):
    @staticmethod
    def forward (ctx, input):
      ctx.save_for_backward(input)
      out = torch.sign (input)
      return out

    @staticmethod
    def backward (ctx, grad_output): 
      input,  = ctx.saved_tensors 
      grad_input = grad_output.clone()
      grad_input = torch.zeros_like(grad_input)
      grad_input = torch.where( ( (input >= 0.0) & (input < 1.0) ),  grad_output * (2.0 - 2.0 * input), grad_input )
      grad_input = torch.where( ( (input > -1.0) & (input < 0.0) ), grad_output * (2.0 + 2.0 * input), grad_input )
      return grad_input

  return b_activation().apply

class weight_binarize_fn (nn.Module):
  def __init__(self):
    super(weight_binarize_fn, self).__init__()
    self.binarize_weight = binarize_weight()

  def forward(self, w):
    E = torch.mean(torch.abs(w)).detach()
    binary_w = self.binarize_weight(w) * E
    return binary_w

class activation_binarize_fn (nn.Module):
  def __init__(self):
    super(activation_binarize_fn, self).__init__()
    self.binarize_activation = binarize_activation()

  def forward(self, x):
    binary_a = self.binarize_activation(x)
    return binary_a

class conv2d_b (nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super(conv2d_b, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.weight_binarize = weight_binarize_fn()
    self.activation_binarize = activation_binarize_fn()
  def forward(self, input):
    b_w = self.weight_binarize(self.weight)
    b_a = self.activation_binarize(input)
    full_conv2d = F.conv2d(b_a, b_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
    return full_conv2d

