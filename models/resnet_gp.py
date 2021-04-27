from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import  transforms
import pickle
import os.path
import datetime
import numpy as np

class FC(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FC, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes)
        self.bn = nn.BatchNorm1d(outplanes)
        self.act = nn.PReLU()
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return self.act(x)

class GDN(nn.Module):
    def __init__(self, inplanes, outplanes, intermediate_dim = 256):
        super(GDN, self).__init__()
        self.fc1 = FC(inplanes, intermediate_dim)
        self.fc2 = FC(intermediate_dim, outplanes)
        self.softmax = nn.Softmax()
    def forward(self, x):
        intermediate = self.fc1(x)
        out = self.fc2(intermediate)
        return intermediate, self.softmax(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_labeled_classes=5, num_unlabeled_classes=5):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1    = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2   = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4   = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.head1 = nn.Linear(512*block.expansion, num_labeled_classes)
        self.head2 = nn.Linear(512*block.expansion, num_unlabeled_classes)

        self.groups = 4
        self.instance_fc = FC(512*block.expansion, 512*block.expansion)
        self.GDN = GDN(512*block.expansion, self.groups)
        self.group_fc = nn.ModuleList([FC(512*block.expansion,512*block.expansion) for i in range(self.groups)])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        B = x.shape[0]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)


        instacne_representation = self.instance_fc(out)
        # GDN
        group_inter, group_prob = self.GDN(instacne_representation)
        # print(group_prob)
        # group aware repr
        v_G = [Gk(out) for Gk in self.group_fc]  # (B,512)
        '''
        # self distributed labeling
        group_label_p = group_prob.data
        group_label_E = group_label_p.mean(dim=0)
        group_label_u = (group_label_p - group_label_E.unsqueeze(dim=-1).expand(self.groups, B).T) / self.groups + (
                    1 / self.groups)
        group_label = torch.argmax(group_label_u, dim=1).data
        '''
        # group ensemble
        group_mul_p_vk = list()
        for k in range(self.groups):
            Pk = group_prob[:, k].unsqueeze(dim=-1).expand(B, 512)
            group_mul_p_vk.append(torch.mul(v_G[k], Pk))
        group_ensembled = torch.stack(group_mul_p_vk).sum(dim=0)
        # instance , group aggregation
        out = instacne_representation + group_ensembled


        out = F.relu(out) #add ReLU to benifit ranking
        out1 = self.head1(out)
        out2 = self.head2(out)
        return out1, out2, out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.is_padding = 0
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.AvgPool2d(2)
            if in_planes != self.expansion*planes:
                self.is_padding = 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.is_padding:
            shortcut = self.shortcut(x)
            out += torch.cat([shortcut,torch.zeros(shortcut.shape).type(torch.cuda.FloatTensor)],1)
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

if __name__ == '__main__':

    from torch.nn.parameter import Parameter
    device = torch.device('cuda')
    num_labeled_classes = 5
    num_unlabeled_classes = 5
    model = ResNet(BasicBlock, [2,2,2,2],num_labeled_classes, num_unlabeled_classes)
    model = model.to(device)
    print(model)
    y1, y2, y3 = model(Variable(torch.randn(256,3,32,32).to(device)))
    print(y1.size(), y2.size(), y3.size())

