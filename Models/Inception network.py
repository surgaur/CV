import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(BasicConv2d,self).__init__()
        
        self.conv2d = nn.Sequential(nn.Conv2d(in_channels,out_channels,**kwargs)
                                    ,nn.BatchNorm2d(out_channels,eps = .001)
                                    ,nn.ReLU()
                                    ,nn.Dropout(0.15))
    def forward(self,x):
        x = self.conv2d(x)
        
        return x


def conv2d():
    conv2d = nn.Sequential(nn.Conv2d(in_channels,out_channels,**kwargs)
                                    ,nn.BatchNorm2d(out_channels,eps = .001)
                                    ,nn.ReLU()
                                    ,nn.Dropout(0.15))
    return conv2d

class InceptionA(nn.Module):
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        
        ## Branch 1
        self.conv1x1 = BasicConv2d(in_channels,out_channels = 96,kernel_size = 1)
        
        ## Branch 2
        self.conv1x1_1 =   BasicConv2d(in_channels,out_channels = 32,kernel_size = 1)
        self.conv5x5   =   BasicConv2d(in_channels=32,out_channels=64,kernel_size = 5,padding = 2)
        
        ## Branch 3
        self.conv1x1_2 =   BasicConv2d(in_channels,out_channels=64,kernel_size=1)
        self.conv3x3   =   BasicConv2d(64,out_channels=96,kernel_size=3)
        self.conv3x3_1 =   BasicConv2d(in_channels =96,out_channels = 128,kernel_size = 3,padding=2)
        
        ## Branch 4
        self.pooling   =  nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1_3 =  BasicConv2d(in_channels,out_channels= 64 , kernel_size = 1 ,padding = 0)
        
    def forward(self,x):
        inp1 = self.conv1x1(x)
        
        inp2 = self.conv1x1_1(x)
        inp2 = self.conv5x5(inp2)
        
        inp3 = self.conv1x1_2(x)
        inp3   = self.conv3x3(inp3)
        inp3 = self.conv3x3_1(inp3)
        
        inp4 = self.pooling(x)
        inp4  = self.conv1x1_3(inp4)
        
        
        output = [inp1,inp2,inp3,inp4]
        out_1 = torch.cat(output,1)
        return out_1

class InceptionB(nn.Module):
    def __init__(self,in_channels):
        super(InceptionB,self).__init__()
        
        ## Branch 1
        self.conv3x3 = BasicConv2d(in_channels,out_channels = 384,kernel_size = 3,stride =2,padding = 0)
        
        
        ## Branch 2
        self.conv1x1   = BasicConv2d(in_channels,out_channels=64,kernel_size=1,stride=1,padding=0)
        self.conv3x3_1 = BasicConv2d(in_channels=64,out_channels=96,kernel_size=3,stride=1,padding=1)
        self.conv3x3_2 = BasicConv2d(in_channels=96,out_channels=128,kernel_size=3,stride=2,padding=0)
        
        ## Branch 3
        self.pooling   =  nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
    def forward(self,x):
        inp1 = self.conv3x3(x)
        
        inp2 = self.conv1x1(x)
        inp2 = self.conv3x3_1(inp2)
        inp2 = self.conv3x3_2(inp2)
        
        inp3 = self.pooling(x)
        
        output = [inp1,inp2,inp3]
        out2 = torch.cat(output,1)
        
        return out2


class Ensembler(nn.Module):
    def __init__(self,in_channels,conv_basic = None):
        super(Ensembler,self).__init__()
        if conv_basic is None:
            conv_basic = BasicConv2d
                
        self.conv2d_1 = conv_basic(in_channels,32,kernel_size = 3,stride = 1,padding =1)
        self.conv2d_2 = conv_basic(32,64,kernel_size = 3,stride = 1,padding =1)
        self.conv2d_3 = conv_basic(64,96,kernel_size = 3,stride = 1,padding =1)
        
        self.ModelA = InceptionA(96)
        self.ModelB = InceptionB(352)
        self.dropout = nn.Dropout(.3) 
        self.fc1 = nn.Linear(864,96)
        self.fc2 = nn.Linear(96,10)
        
    def forward(self,x):
        
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        
        x = self.ModelA(x)
        x = self.ModelB(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.dropout(x)
        out = x.view(-1,864*1*1)
        x = self.fc1(out)
        x = self.fc2(x)
        return x

model = Ensembler(1) ## 1 is a inputy_channels
print('Archeitiecture of Model',model.cuda())
