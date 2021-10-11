#!/usr/bin/env python
###########################################################
# AUTHORS: Cheng-Hao CHEN & Hao LI
# @SJTU
# For free academic & research usage
# NOT for unauthorized commercial usage
###########################################################

from __future__ import print_function
import time
from PIL import Image
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

from misc import *
import models.derain_dense_relu_test as net

from myutils.vgg16 import Vgg16
from myutils import utils
import pdb
import torch.nn.functional as func

# Pre-defined Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix',  help='')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='./datasets/data')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=512, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=512, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--exp', default='sample6', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()

create_exp_dir(opt.exp)

opt.dataset='pix2pix_val'
valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, #opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False
                          )

ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize
# NOTE: size of 2D output maps in the discriminator
sizePatchGAN = 30
real_label = 1
fake_label = 0
# NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
lambdaGAN = opt.lambdaGAN
lambdaIMG = opt.lambdaIMG


netG1=net.Dense_rainall().cuda()
try:
    netG1.load_state_dict(torch.load('./sample/netG1_epoch_4.pth'))
except:
    netG1 = torch.nn.DataParallel(netG1)
    netG1.load_state_dict(torch.load('./sample/netG1_epoch_220.pth'))

netG1.eval()

label_d = torch.FloatTensor(opt.batchSize)

target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input1 = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)

netG1.cuda()

target, input1 = target.cuda(), input1.cuda()
label_d = label_d.cuda()
target = Variable(target, volatile=True)
input1 = Variable(input1, volatile=True)
label_d = Variable(label_d, volatile=True)

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.div_(2).add_(0.5)

    return img

def norm_range(t, range):
    img=norm_ip(t, t.min(), t.max())
    return img

# Begin Testing
for epoch in range(1):
  a1 = time.time()
  for i, data in enumerate(valDataloader, 0):
    with torch.no_grad():
      print('Image:'+str(i))

      from thop import profile
      input = torch.FloatTensor(1, 3, 512, 512).cuda()

      input_cpu, target_cpu, label_cpu,w11,h11 = data
    
      residual_cpu = input_cpu - target_cpu
      
      input_cpu, residual_cpu = input_cpu.float().cuda(),residual_cpu.float().cuda()
   
      torch.cuda.synchronize()
      t0 = time.time()
      residualimg= netG1(input_cpu)
      torch.cuda.synchronize()
      t1 = time.time()
      print(t1 - t0)
     
      a= input_cpu-residualimg
    
      tensor1 =a.data.cpu()
      directory1 = './resulthigh1/'
      if not os.path.exists(directory1):
          os.makedirs(directory1)
      tensor1 = torch.squeeze(tensor1)
      filename1='./resulthigh1/'+str(i)+'.png'
      ndarr1 = tensor1.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
      im1 = Image.fromarray(ndarr1)
      im2 = im1.resize((w11, h11), Image.NEAREST)
      im2.save(filename1)
  a2 = time.time()
  print((a2-a1)/100) 


