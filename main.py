import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

from model import CNN
from utils import progress_bar, save_checkpoint_and_result

from omegaconf import OmegaConf
import hashlib
import json

hp = OmegaConf.load('./config/default.yaml')
device = torch.device('cuda') if torch.cuda.is_available() else cpu


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
parser.add_argument('--type', default='A')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=hp.train.batch_size, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=hp.test.batch_size, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')


net = CNN(args.type).to(device)


criterion = nn.CrossEntropyLoss().to(device)

# optimizer
if hp.train.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=hp.train.lr, momentum=hp.train.sgd.momentum, weight_decay=hp.train.weight_decay)

if hp.train.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=hp.train.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2), weight_decay=hp.train.weight_decay)

#TODO: Add other optimizers and scheduler option 

train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []

best_acc = 0. #* initial value

for epoch in range(hp.train.epoch):
    print('\nEpoch: %d' % (epoch+1))

    '''train'''
    net.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        

        progress_bar(batch_idx, len(trainloader), 'Train --- Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    train_acc = 100.*correct/total
    train_loss /= len(trainloader)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    '''test'''
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test --- Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        # Save checkpoint.
    test_acc = 100.*correct/total

    test_loss /= len(testloader)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

    if test_acc > best_acc:
        best_acc = test_acc
        result = {
        'epoch_best':epoch,
        'best_test_acc': best_acc
        }
        state = {'net': net.state_dict()}



print('Saving..')
result['train_loss_graph'] = train_loss_list
result['train_acc_graph'] = train_acc_list
result['test_loss_graph'] = test_loss_list
result['test_acc_graph'] = test_acc_list

hp['type'] = args.type

save_checkpoint_and_result(hp, result, state)