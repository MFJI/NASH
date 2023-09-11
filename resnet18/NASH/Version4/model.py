import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype_res18
from bi_real import *
import numpy as np

import brevitas.nn as qnn


from brevitas.core.quant import QuantType
from brevitas.quant import IntBias
from brevitas.core.stats import StatsOp
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType


return_quant = True
bit_width_weight_b = 2
bit_width_weight = 4
bit_width_act = 2
bit_width_act_a = 4
bit_width_act_b = 2
bit_width_pool = 4
act_bit_width = 1
SCALING_MIN_VAL=2e-16


def conv1x1(in_planes, out_planes, stride=1):
    return qnn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                                                    weight_quant_type=QuantType.INT,
                                                    weight_bit_width=8,
                                                    weight_narrow_range=True,
                                                    weight_scaling_per_output_channel=True,
                                                    weight_scaling_impl_type=ScalingImplType.STATS,
                                                    weight_scaling_stats_op=StatsOp.MAX,
                                                    weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                                    weight_scaling_min_val=SCALING_MIN_VAL,
                                                    return_quant_tensor=return_quant)

class group_explore(nn.Module):

  def __init__(self, in_planes, out_planes, op_names_group, indices_group, stride_first, expansion, first):
    super(group_explore, self).__init__()
    self._steps = 4
    self.first = first
    self.stride_first = stride_first
    self.expansion = expansion
    self.in_planes = in_planes
    self.out_planes = out_planes
    norm_layer = nn.BatchNorm2d
    downsample = nn.Sequential(
                conv1x1(self.in_planes, self.out_planes * self.expansion, stride_first),
                norm_layer(self.out_planes * self.expansion),
                
            )
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    #You can change the bitwidth to the one you like for the convs and relus
    self.conv0 = qnn.QuantConv2d(self.in_planes, self.out_planes, 3, stride_first, padding=1, bias=False,
                                            weight_quant_type=QuantType.BINARY,
                                            weight_scaling_stats_op=StatsOp.AVE,
                                            weight_scaling_per_output_channel=True,
                                            weight_bit_width=1,
                                            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                            weight_scaling_impl_type=ScalingImplType.STATS,
                                            weight_scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=return_quant)
    

    self.bn0 = norm_layer(self.out_planes)

    self.relu = qnn.QuantReLU(bit_width=8,
                                max_val=10,
                                quant_type=QuantType.INT,
                                scaling_impl_type=ScalingImplType.PARAMETER,
                                restrict_scaling_type=RestrictValueType.LOG_FP,
                                scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=return_quant)
    
    self.relu3 = qnn.QuantReLU(bit_width=8,
                                max_val=10,
                                quant_type=QuantType.INT,
                                scaling_impl_type=ScalingImplType.PARAMETER,
                                restrict_scaling_type=RestrictValueType.LOG_FP,
                                scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=return_quant)
    
    self.relu4 = qnn.QuantHardTanh(bit_width=8,
                                    min_val=-10.0,
                                    max_val=10.0,
                                    narrow_range=True,
                                    quant_type=QuantType.INT,
                                    scaling_impl_type=ScalingImplType.PARAMETER,
                                    restrict_scaling_type=RestrictValueType.LOG_FP,
                                    scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=return_quant)

    self.conv1 = qnn.QuantConv2d(self.out_planes, self.out_planes, 3, 1, padding=1, bias=False,
                                            weight_quant_type=QuantType.BINARY,
                                            weight_scaling_stats_op=StatsOp.AVE,
                                            weight_scaling_per_output_channel=True,
                                            weight_bit_width=1,
                                            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                            weight_scaling_impl_type=ScalingImplType.STATS,
                                            weight_scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=return_quant)
    self.bn1 = norm_layer(self.out_planes)
    self.downsample = downsample

    self.expansion = expansion
    self.conv2 = qnn.QuantConv2d(self.out_planes, self.out_planes, 3, 1, padding=1, bias=False,
                                            weight_quant_type=QuantType.BINARY,
                                            weight_scaling_stats_op=StatsOp.AVE,
                                            weight_scaling_per_output_channel=True,
                                            weight_bit_width=1,
                                            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                            weight_scaling_impl_type=ScalingImplType.STATS,
                                            weight_scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=return_quant)
    self.bn2 = norm_layer(self.out_planes)
    self.conv3 = qnn.QuantConv2d(self.out_planes, self.out_planes, 3, 1, padding=1, bias=False,
                                            weight_quant_type=QuantType.BINARY,
                                            weight_scaling_stats_op=StatsOp.AVE,
                                            weight_scaling_per_output_channel=True,
                                            weight_bit_width=1,
                                            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                            weight_scaling_impl_type=ScalingImplType.STATS,
                                            weight_scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=return_quant)
    self.bn3 = norm_layer(self.out_planes)

    #nas operations
    self._ops = nn.ModuleList()
    for name, index in zip(op_names_group, indices_group):
      if self.first == True:
        stride = 1
        op = OPS[name](self.out_planes, self.out_planes, stride=stride, first=self.first, affine=True)
      elif self.first == False:
        stride = 2 if index < 1 else 1
        if stride == 2:
          op = OPS[name](self.in_planes, self.out_planes, stride=stride, first=self.first, affine=True)
        elif stride == 1:
          op = OPS[name](self.out_planes, self.out_planes, stride=stride, first=self.first, affine=True)
      self._ops += [op]
    self._indices = indices_group 


  def forward(self, x_block0_layer0):

    states = [x_block0_layer0]
    offset = 0

    # block0-layer0-resnet
    identity_block0 = x_block0_layer0
    out_block0_layer0 = self.conv0(x_block0_layer0)
    out_block0_layer0 = self.bn0(out_block0_layer0)

    # block0-layer0-nas
    index0_state = states[self._indices[0]]
    index0_op = self._ops[0]
    index0_op_out = index0_op(index0_state)
    x_block0_layer1 = self.relu4(out_block0_layer0) + self.relu4(index0_op_out) #
    
    offset += len(states)
    x_block0_layer1_n = self.relu(x_block0_layer1)
    states.append(x_block0_layer1_n)
    

    # block0-layer1-resnet
    out_block0_layer1 = self.relu3(x_block0_layer1)
    out_block0_layer1 = self.conv1(out_block0_layer1)
    out_block0_layer1 = self.bn1(out_block0_layer1)
    identity_block0 = self.downsample(identity_block0)
    out_block0_layer1 = self.relu4(out_block0_layer1) + self.relu4(identity_block0)

    # block0-layer1-nas
    index1_state = states[self._indices[1]]
    index1_op = self._ops[1]
    index1_op_out = index1_op(index1_state)
    x_block1_layer0 = self.relu4(out_block0_layer1) + self.relu4(index1_op_out) #
    
    offset +=len(states)
    x_block1_layer0_n = self.relu(x_block1_layer0)
    states.append(x_block1_layer0_n)
    

    # block1-layer0-resnet
    x_block1_layer0 = self.relu3(x_block1_layer0)
    identity_block1 = x_block1_layer0
    out_block1_layer0 = self.conv2(x_block1_layer0)
    out_block1_layer0 = self.bn2(out_block1_layer0)

    # block1-layer0-nas
    index2_state = states[self._indices[2]]
    index2_op = self._ops[2]
    index2_op_out = index2_op(index2_state)
    x_block1_layer1 = self.relu4(out_block1_layer0) + self.relu4(index2_op_out) #

    
    offset +=len(states)
    x_block1_layer1_n = self.relu(x_block1_layer1)
    states.append(x_block1_layer1_n)
    
    # block1-layer1-resnet
    x_block1_layer1 = self.relu3(x_block1_layer1)
    out_block1_layer1 = self.conv3(x_block1_layer1)
    out_block1_layer1 = self.bn3(out_block1_layer1)
    out_block1_layer1 = self.relu4(out_block1_layer1) + self.relu4(identity_block1)


    # block1-layer1-nas
    index3_state = states[self._indices[3]]
    index3_op = self._ops[3]
    index3_op_out = index3_op(index3_state)
    group_out = self.relu4(index3_op_out) + self.relu4(out_block1_layer1) #
    
    offset +=len(states)
    states.append(group_out)
    
    group_out = self.relu3(group_out)
    
    return group_out


class ResNet18(nn.Module):

  def __init__(self, criterion, genotype, num_classes=1000, zero_init_residual=False, width_per_group=64):
    super(ResNet18, self).__init__()
    self._steps = 4
    self.expansion = 1
    self._criterion = criterion

    # group0-3
    op_names_group0, indices_group0 = zip(*genotype.normal0)
    op_names_group1, indices_group1 = zip(*genotype.normal1)
    op_names_group2, indices_group2 = zip(*genotype.normal2)
    op_names_group3, indices_group3 = zip(*genotype.normal3)
    
    
    norm_layer = nn.BatchNorm2d
    planes = [int(width_per_group * 2 ** i) for i in range(4)]
    self.inplanes = planes[0]
    self.conv1 = qnn.QuantConv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False,
                                                    weight_quant_type=QuantType.INT,
                                                    weight_bit_width=8,
                                                    weight_narrow_range=True,
                                                    weight_scaling_per_output_channel=True,
                                                    weight_scaling_impl_type=ScalingImplType.STATS,
                                                    weight_scaling_stats_op=StatsOp.MAX,
                                                    weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                                    weight_scaling_min_val=SCALING_MIN_VAL,
                                                    return_quant_tensor=return_quant)
    self.bn1 = norm_layer(planes[0])
    self.relu3 = qnn.QuantReLU(bit_width=8,
                                max_val=10,
                                quant_type=QuantType.INT,
                                scaling_impl_type=ScalingImplType.PARAMETER,
                                restrict_scaling_type=RestrictValueType.LOG_FP,
                                scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=return_quant)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.group0 = group_explore(planes[0], planes[0], op_names_group0, indices_group0, stride_first=1, expansion=1, first=True)
    self.group1 = group_explore(planes[0], planes[1], op_names_group1, indices_group1, stride_first=2, expansion=1, first=False)
    self.group2 = group_explore(planes[1], planes[2], op_names_group2, indices_group2, stride_first=2, expansion=1, first=False)
    self.group3 = group_explore(planes[2], planes[3], op_names_group3, indices_group3, stride_first=2, expansion=1, first=False)
    self.avgpool = qnn.QuantAvgPool2d(kernel_size=7, stride=1, bit_width=4,
                                                quant_type=QuantType.INT)
    self.fc = qnn.QuantLinear(planes[3] * self.expansion, num_classes, bias=False,
                                            weight_quant_type=QuantType.INT,
                                            weight_bit_width=8,
                                            weight_scaling_per_output_channel=False,
                                            bias_quant=IntBias,
                                            weight_scaling_impl_type=ScalingImplType.STATS,
                                            weight_scaling_stats_op=StatsOp.MAX,
                                            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                            weight_scaling_min_val=SCALING_MIN_VAL,
                                            compute_output_bit_width=True,
                                            compute_output_scale=True)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)


  def forward(self, input):
    x = self.conv1(input)
    x = self.maxpool(x)
    x = self.bn1(x)
    x = self.relu3(x)

    x = self.group0(x)
    x = self.group1(x)
    x = self.group2(x)
    x = self.group3(x)
    
    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    logits = self.fc(x)

    return logits


  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

