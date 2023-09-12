import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class Architect(object):

  def __init__(self, model, args):
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=args.arch_learning_rate, betas=(0.0, 0.999), weight_decay=args.arch_weight_decay)

  def step(self, input_valid, target_valid):
    self.optimizer.zero_grad()
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()
    self.optimizer.step()
