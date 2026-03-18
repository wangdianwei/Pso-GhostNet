"""
# !/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import load_dataset.data_loader_FashionMNIST as data_loader
import os
import math
import copy
from datetime import datetime
import multiprocessing
from utils import Utils
from template.drop import drop_path


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class ECALayer(nn.Module):
    def __init__(self, in_channels, out_channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(in_channels, 2)+b)/gamma))
        k = t if t%2 else t+1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1,1,kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,mode=None,args=None):
        super(GhostModuleV2, self).__init__()
        self.mode=mode
        self.gate_fn=nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]
        elif self.mode in ['attn']:
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest')


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, act_func='h_swish',expansion_rate=1, drop_connect_rate=0.0, args=None):
        super(BasicBlock, self).__init__()
        interChannels = expansion_rate*planes

        self.ghost1 = GhostModuleV2(in_planes, interChannels, relu=True,mode='attn',args=args)
        self.conv_dw = nn.Conv2d(interChannels, interChannels, kernel_size, stride=stride, padding=int((kernel_size-1)/2), bias=False, groups=interChannels)
        self.bn_dw = nn.BatchNorm2d(interChannels)
        if act_func == 'relu':
            self.act_func = nn.ReLU(inplace=True)
            self.se = ECALayer(interChannels, interChannels)
        else:
            self.act_func = Hswish(inplace=True)
            self.se = nn.Sequential()
        self.ghost2 = GhostModuleV2(interChannels, planes, relu=False, mode='original',args=args)

        self.point_conv = nn.Conv2d(interChannels, planes, kernel_size=1, stride=stride, padding=0, bias=False, groups=1)
        self.bn_out = nn.BatchNorm2d(planes)
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        out = self.ghost1(x)
        out = self.bn_dw(self.conv_dw(out))
        out = self.se(out)
        out = self.ghost2(out)
        out = self.bn_out(self.point_conv(out))
        if self.drop_connect_rate > 0.:
            out = drop_path(out, drop_prob=self.drop_connect_rate, training=self.training)
        return out





class TrainModel(object):
    def __init__(self, is_test, batch_size, weight_decay):
        if is_test:
            full_trainloader = data_loader.get_train_loader('../datasets/Fashion-MNIST', batch_size=batch_size, augment=True,shuffle=True, random_seed=2312391, show_sample=False,num_workers=4, pin_memory=True)
            testloader = data_loader.get_test_loader('../datasets/Fashion-MNIST', batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)
            self.full_trainloader = full_trainloader
            self.testloader = testloader
        else:
            trainloader, validate_loader = data_loader.get_train_valid_loader('../datasets/Fashion-MNIST', batch_size=256,augment=True, subset_size=1,valid_size=0.1, shuffle=True,random_seed=2312390, show_sample=False,num_workers=4, pin_memory=True)
            self.trainloader = trainloader
            self.validate_loader = validate_loader

        cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss()
        net = net.cuda()
        best_acc = 0.0
        self.net = net
        self.criterion = criterion
        self.best_acc = best_acc
        self.best_epoch = 0
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.weight_decay = weight_decay


    def train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for ii, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f'% (epoch+1, running_loss/total, (correct/total)))


    def validate(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        is_terminate = 0
        for ii, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
        if correct / total > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = correct / total
        self.log_record('Validate-Epoch:%4d,  Validate-Loss:%.4f, Acc:%.4f'%(epoch + 1, test_loss/total, correct/total))
        return is_terminate

    def process(self):
        total_epoch = Utils.get_params('network', 'epoch_test')
        min_epoch_eval = Utils.get_params('network', 'min_epoch_eval')

        lr_rate = 0.08
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr_rate, momentum=0.9, weight_decay=4e-5, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, min_epoch_eval)

        is_terminate = 0
        params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.log_record('#parameters:%d' % (params))
        for p in range(total_epoch):
            if not is_terminate:
                self.train(p, optimizer)
                scheduler.step()
                is_terminate = self.validate(p)
            else:
                return self.best_acc
        return self.best_acc

    def test(self,p):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for ii, data in enumerate(self.testloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct / total > self.best_acc:
            self.best_acc = correct / total
        self.log_record('Test-Loss:%.4f, Acc:%.4f' % (test_loss / total, correct / total))

class RunModel(object):
    def do_work(self, gpu_id, curr_gen, file_id, is_test, batch_size=None, weight_decay=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0

        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-Acc:%.4f'%best_acc)

            f = open('./populations/acc_%02d.txt'%(curr_gen), 'a+')
            f.write('%s=%.5f\n'%(file_id, best_acc))
            f.flush()
            f.close()
"""


