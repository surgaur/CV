## https://github.com/hoya012/pytorch-MobileNet/blob/master/MobileNet-pytorch.ipynb
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
import matplotlib.pyplot as plt
# import EarlyStopping
#from pytorchtools import EarlyStopping
from tqdm.notebook import tqdm
import albumentations
import pandas as pd
import numpy as np
import io,skimage
from torch.utils.data import Dataset, DataLoader
import os,cv2
import gc
import imageio
gc.collect()

## https://debuggercafe.com/image-augmentation-using-pytorch-and-albumentations/
path = 'C:/Kaggle Datasets/dogs-vs-cats-redux-kernels-edition/train/train'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
#device = torch.device("cpu")
#device


import warnings
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")




class depthwise_conv(nn.Module):
    def __init__(self, nin, kernel_size, bias=False, stride=1):
        super(depthwise_conv, self).__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(in_channels=nin,out_channels=nin,kernel_size=kernel_size,groups = nin)
                                  ,nn.BatchNorm2d(nin)
                                  ,nn.ReLU()
                                 )
    def forward(self,x):
        x = self.conv(x)
        return x




class one_by_one(nn.Module):
    def __init__(self,in_channels,out_channels,bias=False,stride=1):
        super(one_by_one, self).__init__()
        self.conv1x1 = nn.Sequential(
               nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
               ,nn.BatchNorm2d(out_channels)
               ,nn.ReLU()
        )
            
        
    def forward(self,x):
        x = self.conv1x1(x)
        return x





class MobileNet(nn.Module):
    def __init__(self, input_channel):
        super(MobileNet, self).__init__()
        self.conv_basic = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
        self.block_1 = nn.Sequential(
                 depthwise_conv(32,3),
                 one_by_one(32,64)
        )
        
    def forward(self,x):
        
        x = self.conv_basic(x)
        x = self.block_1(x)
        
        return x




model = MobileNet(3)
print('Archeitiecture of Model',model)







