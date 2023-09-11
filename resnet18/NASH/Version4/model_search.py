import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype_res18
from bi_real import *
import numpy as np


def binarize_path():
  class b_path(torch.autograd.Function):

    @staticmethod
    def forward(ctx, arch_connection_weights):
      probability = F.softmax(arch_connection_weights, dim=-1)
      index = torch.multinomial(probability, 1, replacement=False)
      gates = torch.zeros_like(probability).scatter_(0, index, 1)
      ctx.saved_for_backward = [arch_connection_weights, probability, gates]
      return gates
   
    @staticmethod
    def backward(ctx, grad_output):
      arch_connection_weights, probability, gates = ctx.saved_for_backward
      grad_input = grad_output.clone()
      for i, arch_connection_weight in enumerate(arch_connection_weights):
        grad_arch_connection_weight_i = torch.tensor([0.0]).to('cuda')
        for j, gate in enumerate(gates):
          grad_gate_j = grad_output[j]
          probability_j = probability[j]
          probability_i = probability[i]
          if i == j:
            sigma = torch.tensor([1.0]).to('cuda')
          else:
            sigma = torch.tensor([0.0]).to('cuda')
          grad_arch_connection_weight_i = grad_arch_connection_weight_i + grad_gate_j * probability_j * (sigma - probability_i) 
        grad_input[i] = grad_arch_connection_weight_i
      return  grad_input

  return b_path().apply

def reduce_memory():
  class r_memory(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gates, x, y, operations):
      #output = y.clone()
      output = y.detach()
      ctx.saved_for_backward = [gates, x, y, operations]
      return output

    @staticmethod
    def backward(ctx, grad_output):
      gates, x, y, operations = ctx.saved_for_backward
      grad_gates = gates.clone()
      grad_x = None
      grad_y = grad_output.clone()
      grad_operations = None

      for j, op in enumerate(operations):
        grad_gates[j] = torch.sum(grad_output * op(x))
      return  grad_gates, grad_x, grad_y, grad_operations

  return r_memory().apply

class MixedOp(nn.Module):

  def __init__(self, in_channels, out_channels, stride, first, affine=True):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](in_channels, out_channels, stride, first, affine)
      self._ops.append(op)
    self.binarize_path = binarize_path()
    self.reduce_memory = reduce_memory()

  def forward(self, x, arch_connection_weights):
    gates = self.binarize_path(arch_connection_weights)
    for i, gate in enumerate(gates):
      if gate == torch.tensor([1.0]).to('cuda'):
        y = self._ops[i](x)
    output = self.reduce_memory(gates, x, y, self._ops)
    return output

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class group_explore(nn.Module):

  def __init__(self, in_planes, out_planes, stride_first, expansion, first):
    super(group_explore, self).__init__()
    self._steps = 4
    self.first = first
    self.expansion = expansion
    self.in_planes = in_planes
    self.out_planes = out_planes
    norm_layer = nn.BatchNorm2d
    downsample = nn.Sequential(
                conv1x1(self.in_planes, self.out_planes * self.expansion, stride_first),
                norm_layer(self.out_planes * self.expansion),
            )
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv0 = nn.Conv2d(self.in_planes, self.out_planes, 3, stride_first, padding=1, bias=False)
    self.bn0 = norm_layer(self.out_planes)
    self.conv1 = nn.Conv2d(self.out_planes, self.out_planes, 3, 1, padding=1, bias=False)
    self.bn1 = norm_layer(self.out_planes)
    self.downsample = downsample
    self.relu = nn.Tanh()
    self.relu3 = nn.Tanh()
    
    self.expansion = expansion
    self.conv2 = nn.Conv2d(self.out_planes, self.out_planes, 3, 1, padding=1, bias=False)
    self.bn2 = norm_layer(self.out_planes)
    self.conv3 = nn.Conv2d(self.out_planes, self.out_planes, 3, 1, padding=1, bias=False)
    self.bn3 = norm_layer(self.out_planes)

    #nas operations
    self._ops = nn.ModuleList()
    for i in range(self._steps):
      for j in range(1+i):
        if self.first == True:
          stride = 1
          op = MixedOp(self.out_planes, self.out_planes, stride=stride, first=self.first, affine=True)
          self._ops.append(op)
        elif self.first == False:
          stride = 2 if j < 1 else 1
          if stride == 2:
            op_n = MixedOp(self.in_planes, self.out_planes, stride=stride, first=self.first, affine=True)
            self._ops.append(op_n)
          elif stride == 1:
            op = MixedOp(self.out_planes, self.out_planes, stride=stride, first=self.first, affine=True)
            self._ops.append(op)
        

  def forward(self, x_block0_layer0, group_arch_weights):

    states = [x_block0_layer0]
    offset = 0

    # block0-layer0-resnet
    identity_block0 = x_block0_layer0
    out_block0_layer0 = self.conv0(x_block0_layer0)
    out_block0_layer0 = self.bn0(out_block0_layer0)

    # block0-layer0-nas
    x_block0_layer1 = sum(self._ops[offset+j](h, group_arch_weights[offset+j]) for j, h in enumerate(states)) + out_block0_layer0
    offset +=len(states)
    x_block0_layer1_n = self.relu(x_block0_layer1)
    states.append(x_block0_layer1_n)
    x_block0_layer1 = self.relu3(x_block0_layer1)

    # block0-layer1-resnet
    out_block0_layer1 = self.conv1(x_block0_layer1)
    out_block0_layer1 = self.bn1(out_block0_layer1)
    identity_block0 = self.downsample(identity_block0)
    out_block0_layer1 += identity_block0


    # block0-layer1-nas
    x_block1_layer0 = sum(self._ops[offset+j](h, group_arch_weights[offset+j]) for j, h in enumerate(states)) + out_block0_layer1
    x_block1_layer0_n = self.relu(x_block1_layer0)
    offset +=len(states)
    states.append(x_block1_layer0_n)
    x_block1_layer0 = self.relu3(x_block1_layer0)

    # block1-layer0-resnet
    identity_block1 = x_block1_layer0
    out_block1_layer0 = self.conv2(x_block1_layer0)
    out_block1_layer0 = self.bn2(out_block1_layer0)

    # block1-layer0-nas
    x_block1_layer1 = sum(self._ops[offset+j](h, group_arch_weights[offset+j]) for j, h in enumerate(states)) + out_block1_layer0
    x_block1_layer1_n = self.relu(x_block1_layer1)
    offset +=len(states)
    states.append(x_block1_layer1_n)
    x_block1_layer1 = self.relu3(x_block1_layer1)
    
    # block1-layer1-resnet
    out_block1_layer1 = self.conv3(x_block1_layer1)
    out_block1_layer1 = self.bn3(out_block1_layer1)
    out_block1_layer1 += identity_block1

    # block1-layer1-nas
    group_out = sum(self._ops[offset+j](h, group_arch_weights[offset+j]) for j, h in enumerate(states)) + out_block1_layer1
    group_out_n = self.relu(group_out)
    offset +=len(states)
    states.append(group_out)
    group_out = self.relu3(group_out)
    
    return group_out


class ResNet18(nn.Module):

  def __init__(self, criterion, num_classes=10, zero_init_residual=False, width_per_group=64):
    super(ResNet18, self).__init__()
    self._steps = 4
    self.expansion = 1
    self._criterion = criterion
    self._initialize_alphas()

    # group0-3
    self.group0_arch_weights = self.arch_weights[0]
    self.group1_arch_weights = self.arch_weights[1]
    self.group2_arch_weights = self.arch_weights[2]
    self.group3_arch_weights = self.arch_weights[3]

    
    norm_layer = nn.BatchNorm2d
    planes = [int(width_per_group * 2 ** i) for i in range(4)]
    self.inplanes = planes[0]
    self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(planes[0])
    self.relu3 = nn.ReLU()

    self.group0 = group_explore(planes[0], planes[0], stride_first=1, expansion=1, first=True)
    self.group1 = group_explore(planes[0], planes[1], stride_first=2, expansion=1, first=False)
    self.group2 = group_explore(planes[1], planes[2], stride_first=2, expansion=1, first=False)
    self.group3 = group_explore(planes[2], planes[3], stride_first=2, expansion=1, first=False)
    self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
    self.fc = nn.Linear(planes[3] * self.expansion, num_classes, bias=False)

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
    x = self.bn1(x)
    x = self.relu3(x)

    x = self.group0(x, self.group0_arch_weights )
    x = self.group1(x, self.group1_arch_weights )
    x = self.group2(x, self.group2_arch_weights )
    x = self.group3(x, self.group3_arch_weights )

    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    logits = self.fc(x)

    return logits


  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    # num_groups is the number of groups
    # num_connections is the number of connections
    # num_operations is the number of operations
    num_groups = sum(1 for i in range(4))
    num_connections = sum(1 for i in range(self._steps) for n in range(1 + i))
    num_operations = len(PRIMITIVES)

    self.arch_weights = Variable(1e-3*torch.randn(num_groups, num_connections, num_operations).cuda(), requires_grad=True)
    self._arch_parameters = [self.arch_weights]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    def _parse(group_arch_probabilities):
      gene = []
      n = 1 #//
      start = 0
      for i in range(self._steps):
        end = start + n
        W = group_arch_probabilities[start:end].copy()
        # only one precessor is remained
        edges = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:1] #//
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_g0 = _parse(F.softmax(self.group0_arch_weights, dim=-1).data.cpu().numpy())
    gene_g1 = _parse(F.softmax(self.group1_arch_weights, dim=-1).data.cpu().numpy())
    gene_g2 = _parse(F.softmax(self.group2_arch_weights, dim=-1).data.cpu().numpy())
    gene_g3 = _parse(F.softmax(self.group3_arch_weights, dim=-1).data.cpu().numpy())
    genotype = Genotype_res18(normal0=gene_g0, normal1=gene_g1, normal2=gene_g2, normal3=gene_g3)

    return genotype
