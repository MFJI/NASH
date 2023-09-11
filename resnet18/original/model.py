import torch
import torch.nn as nn

import brevitas.nn as qnn
import brevitas.onnx as bo

from brevitas.core.quant import QuantType
from brevitas.quant import IntBias
from brevitas.core.stats import StatsOp
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType


return_quant = True
bit_width_weight = 4
bit_width_weight_b = 2
bit_width_act = 4
bit_width_pool = 4
bit_width_act_b = 2
SCALING_MIN_VAL=2e-16


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                                            weight_quant_type=QuantType.BINARY,
                                            weight_scaling_stats_op=StatsOp.AVE,
                                            weight_scaling_per_output_channel=True,
                                            weight_bit_width=1,
                                            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                            weight_scaling_impl_type=ScalingImplType.STATS,
                                            weight_scaling_min_val=SCALING_MIN_VAL,
                                            return_quant_tensor=return_quant)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                                            weight_quant_type=QuantType.BINARY,
                                            weight_scaling_stats_op=StatsOp.AVE,
                                            weight_scaling_per_output_channel=True,
                                            weight_bit_width=1,
                                            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                            weight_scaling_impl_type=ScalingImplType.STATS,
                                            weight_scaling_min_val=SCALING_MIN_VAL,
                                            return_quant_tensor=return_quant)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu3 = qnn.QuantReLU(bit_width=2,
                                                max_val=1.5,
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

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu3(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu4(x) + self.relu4(output)
        output = self.relu3(output)
        
        return output


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1, bias=False,
                                            weight_quant_type=QuantType.BINARY,
                                            weight_scaling_stats_op=StatsOp.AVE,
                                            weight_scaling_per_output_channel=True,
                                            weight_bit_width=1,
                                            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                            weight_scaling_impl_type=ScalingImplType.STATS,
                                            weight_scaling_min_val=SCALING_MIN_VAL,
                                            return_quant_tensor=return_quant)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1, bias=False,
                                            weight_quant_type=QuantType.BINARY,
                                            weight_scaling_stats_op=StatsOp.AVE,
                                            weight_scaling_per_output_channel=True,
                                            weight_bit_width=1,
                                            weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                            weight_scaling_impl_type=ScalingImplType.STATS,
                                            weight_scaling_min_val=SCALING_MIN_VAL,
                                            return_quant_tensor=return_quant)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            qnn.QuantConv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0, bias=False,
                                            weight_quant_type=QuantType.INT,
                                                    weight_bit_width=8,
                                                    weight_narrow_range=True,
                                                    weight_scaling_per_output_channel=True,
                                                    weight_scaling_impl_type=ScalingImplType.STATS,
                                                    weight_scaling_stats_op=StatsOp.MAX,
                                                    weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                                    weight_scaling_min_val=SCALING_MIN_VAL,
                                            return_quant_tensor=return_quant),
            nn.BatchNorm2d(out_channels)
        )
        self.relu3 = qnn.QuantReLU(bit_width=2,
                                                max_val=1.5,
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

    def forward(self, x):
        extra_x = self.extra(x)
        extra_x = self.relu4(extra_x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu3(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu4(out)
        out = self.relu3(extra_x + out)
        
        return out


class ResNet18(nn.Module):
    def __init__(self, criterion):
        super(ResNet18, self).__init__()
        self._criterion = criterion
        self.conv1 = qnn.QuantConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False,
                                                    weight_quant_type=QuantType.INT,
                                                    weight_bit_width=8,
                                                    weight_narrow_range=True,
                                                    weight_scaling_per_output_channel=True,
                                                    weight_scaling_impl_type=ScalingImplType.STATS,
                                                    weight_scaling_stats_op=StatsOp.MAX,
                                                    weight_restrict_scaling_type=RestrictValueType.LOG_FP,
                                                    weight_scaling_min_val=SCALING_MIN_VAL,
                                                    return_quant_tensor=return_quant)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(RestNetDownBlock(64, 64, [1, 1]), RestNetBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]), RestNetBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]), RestNetBasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]), RestNetBasicBlock(512, 512, 1))
        self.avgpool = qnn.QuantAvgPool2d(kernel_size=7, stride=1, bit_width=bit_width_pool)
        self.fc = qnn.QuantLinear(512, 1000, bias=False,
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
        self.relu3 = qnn.QuantReLU(bit_width=2,
                                                max_val=1.5,
                                                quant_type=QuantType.INT,
                                                scaling_impl_type=ScalingImplType.PARAMETER,
                                                restrict_scaling_type=RestrictValueType.LOG_FP,
                                                scaling_min_val=SCALING_MIN_VAL, return_quant_tensor=return_quant)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.relu3(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    
    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)
