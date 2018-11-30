import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader,Dataset
from PIL import Image

import os
import argparse
import shutil
from PIL import Image
import math
import time
import numpy as np

from model import darknet

parser = argparse.ArgumentParser(description='PyTorch CCCV30 prune')

parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', default='darknet', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('--percent', default=0.5, type=float,
                    help='depth of the neural network')
args = parser.parse_args()

def log(log_file,str):
    log = open(log_file,'a+')
    log.writelines(str+'\n') 
    log.close()

def get_layer_fm_index(index, conv_cfg):
    layer_index = 0
    fm_index = 0
    while (index >= conv_cfg[layer_index]):
        index = index - conv_cfg[layer_index]
        layer_index = layer_index + 1
    fm_index = index
    return layer_index, fm_index

def get_index(layer_index, fm_index, conv_cfg):
    index = 0
    for i in range(layer_index):
        index = index + conv_cfg[i]
    index = index + fm_index
    return index

def generate_pruned_cfg(loss_rank, cfg, prune_cnt):
    conv_cfg = []  # store the fm cnt of each conv layer
    pruned_layer_cfg = [[]]  # store the remain fm index of each conv layer
    pruned_model_cfg = cfg  # store pruned cfg to generate new model

    for l in cfg:
        if l != 'M':
            conv_cfg.append(l[0])

    for i in range(len(cfg) - cfg.count('M') - 1):
        pruned_layer_cfg.append([])

    fm_cnt = len(loss_rank) - prune_cnt  # remained fm count

    for i in range(fm_cnt):
        layer_index, fm_index = get_layer_fm_index(loss_rank[i], conv_cfg)
        pruned_layer_cfg[layer_index].append(fm_index)

    layer_index = 0
    for i in range(len(cfg)):
        if cfg[i][0] != 'M':
            pruned_model_cfg[i][0] = len(pruned_layer_cfg[layer_index])
            layer_index += 1
    return pruned_model_cfg, pruned_layer_cfg

def pruned_model_init(model, pruned_model, pruned_layer_cfg):

    conv_layer_index = 0

    for layer_index, pruned_layer in pruned_model.feature._modules.items():

        _, layer = model.feature._modules.items()[int(layer_index)]

        if isinstance(pruned_layer, nn.Conv2d):

            if (conv_layer_index == 0):
                weight = layer.weight.data.cpu().numpy()
                pruned_weight = layer.weight.data.cpu().numpy()[:len(pruned_layer_cfg[conv_layer_index])]
                print pruned_weight.shape

                out_channel = 0
                for i in pruned_layer_cfg[conv_layer_index]:
                    pruned_weight[out_channel] = weight[i]
                    out_channel = out_channel + 1

                pruned_layer.weight.data = torch.from_numpy(pruned_weight).cuda()

            else:
                weight = layer.weight.data.cpu().numpy()
                pruned_weight = layer.weight.data.cpu().numpy()[:len(pruned_layer_cfg[conv_layer_index]), :len(pruned_layer_cfg[conv_layer_index - 1])]
                print pruned_weight.shape

                in_channel = 0
                out_channel = 0
                for i in pruned_layer_cfg[conv_layer_index]:
                    for j in pruned_layer_cfg[conv_layer_index - 1]:
                        pruned_weight[out_channel, in_channel] = weight[i, j]
                        in_channel = in_channel + 1
                    in_channel = 0
                    out_channel = out_channel + 1

                pruned_layer.weight.data = torch.from_numpy(pruned_weight).cuda()

        elif isinstance(pruned_layer, nn.BatchNorm2d):

            if (conv_layer_index < 18):

                channel = 0
                for i in pruned_layer_cfg[conv_layer_index]:
                    pruned_layer.weight.data[channel] = layer.weight.data.cpu()[i].clone()
                    pruned_layer.bias.data[channel] = layer.bias.data.cpu()[i].clone()
                    pruned_layer.running_mean.data[channel] = layer.running_mean.data.cpu()[i].clone()
                    pruned_layer.running_var.data[channel] = layer.running_var.data.cpu()[i].clone()
                    channel = channel + 1

            conv_layer_index += 1

    for layer_index, pruned_layer in pruned_model.classifier._modules.items():

        _, layer = model.classifier._modules.items()[int(layer_index)]
        
        if isinstance(pruned_layer, nn.Conv2d):
            weight = layer.weight.data.cpu().numpy()
            pruned_weight = layer.weight.data.cpu().numpy()[:, :len(pruned_layer_cfg[-1])]
            print pruned_weight.shape

            in_channel = 0
            for i in pruned_layer_cfg[conv_layer_index - 1]:
                pruned_weight[:, in_channel] = weight[:, i]
                in_channel = in_channel + 1

            pruned_layer.weight.data = torch.from_numpy(pruned_weight).cuda()

def get_flops(cfg,img_h,img_w):
    flops = 0
    in_channles=3
    for l in cfg:
        if l != 'M':
            flops += in_channles*l[0]*l[1]*l[1]*img_h*img_w
            in_channles = l[0]
        else:
            img_h=img_h/2
            img_w=img_w/2
    return flops

loss = np.random.random(7200)
loss_rank = np.argsort(loss)
cfg=[[32,3],'M',[64,3],'M',[128,3],[64,1],[128,3],'M',[256,3],[128,1],[256,3],'M',[512,3],[256,1],[512,3],[256,1],[512,3],'M',[1024,3],[512,1],[1024,3],[512,1],[1024,3]]
origin_flops=get_flops(cfg,224,224)
pruned_model_cfg,pruned_layer_cfg = generate_pruned_cfg(loss_rank, cfg, int(7200*args.percent))
pruned_flops=get_flops(pruned_model_cfg,224,224)
print pruned_model_cfg
model = darknet()
model.cuda()
print model
model.load_state_dict(torch.load(args.model))
pruned_model = darknet(cfg=pruned_model_cfg)
pruned_model.cuda()
print pruned_model
pruned_model_init(model, pruned_model, pruned_layer_cfg)
print 'origin model flops: %d, pruned_model flops: %d'%(origin_flops,pruned_flops)
torch.save(pruned_model, './models/prune/prune_%.2f_darknet19.pkl'%args.percent)
print('pruned model saved at models/prune')