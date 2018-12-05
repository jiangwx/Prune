import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader,Dataset

import os
import argparse
import shutil
from PIL import Image
import math
import time
import numpy as np
from model import darknet

parser = argparse.ArgumentParser(description='PyTorch CCCV30 test')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--dataset', type=str, default='CCCV-30',
                    help='training dataset (default: CCCV-30)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()

if not args.model:
    print("=> no model found")

def test(model, data_loader, loss_func):
    model.eval()
    correct = 0
    loss = 0.0

    for inputs, labels in data_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        loss += loss_func(outputs, labels).item()

    accuracy = 100. * float(correct) / float(len(data_loader.dataset))
    loss /= float(len(data_loader))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, loss

test_data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


env_dict = os.environ
dataset_path = env_dict.get('DATASET')

if args.dataset == 'CCCV-30':
    test_dataset = torchvision.datasets.ImageFolder(root=dataset_path+'/CCCV-30/test_set', transform=test_data_transform)
elif args.dataset == 'imagenet':
    test_dataset = torchvision.datasets.ImageFolder(root=dataset_path+'/imagenet/val', transform=test_data_transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=12)

model = torch.load(args.model)

print model
loss_func = nn.CrossEntropyLoss()
test_accuracy, test_loss = test(model, test_loader, loss_func)

