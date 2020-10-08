### https://github.com/gsp-27/pytorch_Squeezenet/blob/master/model.py -- pytorch
### https://github.com/DT42/squeezenet_demo/blob/master/model.png
### https://arxiv.org/pdf/1602.07360.pdf - original paper
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import numpy as np
import torch.optim as optim
import math



class Fire(nn.Module):
    def __init__(self,inplanes,outplanes_1,outplanes_2):
        super(Fire,self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, outplanes_1, kernel_size = 1, stride = 1),
        	                       nn.BatchNorm2d(outplanes_1, eps = .001),
        	                       nn.ReLU()) ### Squeeze layer

        self.conv2 = nn.Sequential(nn.Conv2d(outplanes_1, outplanes_2, kernel_size = 1, stride =1),
        	                       nn.BatchNorm2d(outplanes_2,eps = .001),
        	                       nn.ReLU()) ### 1x1 kernel size of Expand layer
        
        self.conv3 = nn.Sequential(nn.Conv2d(outplanes_1,outplanes_2,kernel_size = 3, stride = 1, padding = 1),
        	                      nn.BatchNorm2d(outplanes_2,eps = .001),
        	                      nn.ReLU()) ### 3x3 kernel size of Expand layer

    def forward(seld,x):

    	x = self.conv1(x)
    	out1 = self.conv2(x)
    	out2 = self.conv3(x)

    	out = torch.cat([out1,out2],1)

    	return out




    	