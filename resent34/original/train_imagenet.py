import onnx

import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import *

import brevitas.onnx as bo

from brevitas.export import FINNManager

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/data/volume_2/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.000, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0000, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=str, default='0,1', help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--pretrained_provide', action='store_true', default=False, help='pretrained_provide')
parser.add_argument('--pretrained_model_path', type=str, default='/PATH/NAME.pth', help='path of pretrained model')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=35, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=True, help='data parallelism')
args = parser.parse_args()



args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  gpus = [int(i) for i in args.gpu.split(',')]
  if len(gpus)==1:
    torch.cuda.set_device(int(args.gpu))
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %s' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
#  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
#  criterion_smooth = criterion_smooth.cuda()
  criterion_smooth = nn.CrossEntropyLoss()
  criterion_smooth = criterion_smooth.cuda()


  model = ResNet34(criterion_smooth)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
#      transforms.ColorJitter(
#        brightness=0.4,
#        contrast=0.4,
#        saturation=0.4,
#        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)

  best_acc_top1 = 0
  best_acc_top5 = 0
  epoch_start = 0

  if args.pretrained_provide :
    checkpoint = torch.load(args.pretrained_model_path)
    best_acc_top1 = checkpoint['best_acc_top1']
    epoch_start = checkpoint['epoch_start']
    state_dict = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict,strict=False)
    model.cuda()
    optimizer.load_state_dict(checkpoint['optimizer'])

  if args.parallel:
    model = nn.DataParallel(model).cuda()
  else:
    model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  for epoch in range(epoch_start, args.epochs):
    lr = adjust_learning_rate(optimizer, epoch, args)
    logging.info('epoch %d lr %e', epoch, lr)

    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc_top1 %f', valid_acc_top1)
    logging.info('valid_acc_top5 %f', valid_acc_top5)

    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True

    if valid_acc_top5 > best_acc_top5:
      best_acc_top5 = valid_acc_top5


  model = model.eval()

  state = {'epoch_start': epoch+1, 'state_dict': model.state_dict(), 'best_acc_top1': best_acc_top1, 'best_acc_top5': best_acc_top5, 'optimizer': optimizer.state_dict()}

  torch.save(state, '/PATH/NAME.pth')

  bo.export_finn_onnx(model.module.cpu(), torch.randn(1,3,224,224), export_path='/PATH/NAME.onnx')


def adjust_learning_rate(optimizer, epoch, args):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.learning_rate * (0.1 ** (epoch // 20))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr


def train(train_queue, model, criterion_smooth, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion_smooth(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()


  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda()

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)


  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main() 
