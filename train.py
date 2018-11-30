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


parser = argparse.ArgumentParser(description='PyTorch CCCV30 training')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train (default: 120)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', default='darknet', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('--log', default='./logs/train_darknet19.log', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()

print args

def log(log_file,str):
    log = open(log_file,'a+')
    log.writelines(str+'\n') 
    log.close()

def poly(base_lr, power, total_epoch, now_epoch):
    return base_lr * (1 - math.pow(float(now_epoch) / float(total_epoch), power))

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
    log(args.log,'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, loss

def train(model, data_loader, loss_func, optimizer):
    model.train()
    correct = 0
    train_loss = 0.0

    for inputs, labels in data_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        loss.backward()
        optimizer.step()

    accuracy = 100. * float(correct) / float(len(data_loader.dataset))
    train_loss /= float(len(data_loader))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(train_loss, correct, len(data_loader.dataset), accuracy))
    log(args.log,'Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(train_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, train_loss

train_data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print train_data_transform,test_data_transform

env_dict = os.environ
dataset_path = env_dict.get('DATASET')

train_dataset = torchvision.datasets.ImageFolder(root=dataset_path+'/CCCV-30/train_set', transform=train_data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)
test_dataset = torchvision.datasets.ImageFolder(root=dataset_path+'/CCCV-30/test_set', transform=test_data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=12)

model = darknet()
model.cuda()
print model

start_epoch = args.start_epoch
total_epoch = args.epochs

if start_epoch != 0:
    model.load_state_dict(torch.load(args.resume))

history_score=np.zeros((total_epoch + 1,4))

for epoch in range(start_epoch, total_epoch):
    start = time.time()
    print('epoch%d...'%epoch)
    log(args.log,'epoch%d...'%epoch)
    if epoch<=total_epoch*0.5:
        lr = poly(args.lr, 4, total_epoch*0.5, epoch)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    train_accuracy, train_loss = train(model,train_loader,loss_func,optimizer)
    test_accuracy, test_loss = test(model, test_loader, loss_func)

    torch.save(model.state_dict(), './models/train/check_point.pkl')
    if test_accuracy > max(history_score[:,2]):
        torch.save(model.state_dict(), './models/train/best%.1f.pkl'%test_accuracy)

    history_score[epoch][0] = train_accuracy
    history_score[epoch][1] = train_loss
    history_score[epoch][2] = test_accuracy
    history_score[epoch][3] = test_loss

    print('epoch%d time %.4fs\n' % (epoch,time.time()-start))
    log(args.log,'epoch%d time %.4fs\n' % (epoch,time.time()-start))

