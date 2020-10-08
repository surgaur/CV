#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install albumentations


# In[1]:


import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models


class MNIST_VGG16(nn.Module):
    def __init__(self):
        super(MNIST_VGG16,self).__init__()
        
        self.layer1 = nn.Sequential(
                    nn.Conv2d(in_channels =1,out_channels =64,kernel_size =3,stride=1,padding=0 ) ## 26*26*64
                   ,nn.ReLU()
                   ,nn.BatchNorm2d(64)
                   ,nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size=3,stride=1,padding=0) ## 24*24*64
                   ,nn.ReLU()
                   ,nn.BatchNorm2d(64)
                   ,nn.MaxPool2d(2,2)
                   ,nn.Dropout(0.2)) ## 12*12*64
        
        self.layer2 = nn.Sequential(
                    nn.Conv2d(in_channels =64,out_channels =64,kernel_size =3,stride=1,padding=0 ) ## 10*10*64 
                   ,nn.ReLU()
                   ,nn.BatchNorm2d(64)
                   ,nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size=3,stride=1,padding=0) # 8*8*64
                   ,nn.ReLU()
                   ,nn.BatchNorm2d(64)
                   ,nn.Conv2d(in_channels = 64,out_channels = 256,kernel_size=3,stride=1,padding=0) # 6*6*256
                   ,nn.ReLU()
                   ,nn.BatchNorm2d(256)
                   ,nn.MaxPool2d(2,2) ## 3*3*256
                   ,nn.Dropout(0.2)) 
        
        self.AdaptivePooling = nn.AdaptiveAvgPool2d((6, 6))
        
        self.fc1 = nn.Linear(6*6*256,32)
        
        self.fc2 = nn.Linear(32,10)
        
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.AdaptivePooling(x)
        x = x.view(-1,256*6*6)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


model = MNIST_VGG16().to(device)
print('Archeitiecture of Model',model)





