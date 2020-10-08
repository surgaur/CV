import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
import matplotlib.pyplot as plt

def conv1x1_block(in_channels,out_channels,kernel_size = 1,**kwargs):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size =1,**kwargs ),
                                        nn.BatchNorm2d(out_channels, eps=0.001),
                                        nn.ReLU(),
                                        nn.Dropout(0.15))
    
class Residual_Block(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        
        super(Residual_Block,self).__init__()
        r"""
            consist two 3x3 conv2d blocks 
            kernel_size =  Diiferent Types of Kernels
            stride =1
            padding =1
         """
        self.conv3x3_1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,**kwargs ),
                                        nn.BatchNorm2d(out_channels, eps=0.001),
                                        nn.ReLU(),
                                        nn.Dropout(0.15))
        self.conv3x3_2 = nn.Sequential(nn.Conv2d(in_channels = out_channels ,out_channels = out_channels,**kwargs ),
                                        nn.BatchNorm2d(out_channels, eps=0.001))
        
    def forward(self,x):
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        
        return x

class ResnetEnsembler(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(ResnetEnsembler,self).__init__()
        
        self.conv3x3 = nn.Sequential(nn.Conv2d(in_channels,out_channels,**kwargs ),
                                        nn.BatchNorm2d(out_channels, eps=0.001),
                                        nn.ReLU(),
                                        nn.Dropout(0.15))
        self.blk_1 = Residual_Block(in_channels = 32 ,out_channels = 32,kernel_size = 3,padding = 1,stride =1)
        self.conv1 = conv1x1_block(in_channels = 32,out_channels = 64,padding=0,stride=1)
        
        self.blk_2 = Residual_Block(in_channels = 64 ,out_channels = 64,kernel_size = 3,padding = 1,stride =1)
        self.conv2 = conv1x1_block(in_channels = 64,out_channels = 128,padding=0,stride=1)
        
        self.blk_3 = Residual_Block(in_channels = 128 ,out_channels = 128,kernel_size = 3,padding = 1,stride =1)
        self.conv3 = conv1x1_block(in_channels = 128,out_channels = 256,padding=0,stride=1)
        
        self.blk_4 = Residual_Block(in_channels = 256 ,out_channels = 256,kernel_size = 3,padding = 1,stride =1)
        self.conv4 = conv1x1_block(in_channels = 256,out_channels = 512,padding=0,stride=1)
        
        self.blk_5 = Residual_Block(in_channels = 512 ,out_channels = 512,kernel_size = 3,padding = 1,stride =1)
        self.dropout = nn.Dropout(.3)
        self.fc1 = nn.Linear(512,96)
        self.fc2 = nn.Linear(96,10)

        
        
    def forward(self,x):
        
        x = self.conv3x3(x)
        identity = x
        x =  self.blk_1(x)
        x = torch.relu(x + identity)
        x = self.conv1(x)
        identity = x
        
        x = self.blk_2(x)
        x = torch.relu(x + identity)
        x = self.conv2(x)
        identity = x
        
        x = self.blk_3(x)
        x = torch.relu(x + identity)
        x = self.conv3(x)
        identity =x
        
        x = self.blk_4(x)
        x = torch.relu(x + identity)
        x = self.conv4(x)
        identity = x
        
        x = self.blk_5(x)
        x = torch.relu(x + identity)
        
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.dropout(x)
        
        out = x.view(-1,512*1*1)
        x = self.fc1(out)
        x = self.fc2(x)

        return x


model = ResnetEnsembler(in_channels =1 , out_channels=32,kernel_size=3,padding =1,stride =1)
print('Archeitiecture of Model',model.cuda())


