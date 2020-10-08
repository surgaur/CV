import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

class Fire(nn.Module):
    def __init__(self,inplanes,outplanes_1,outplanes_2):
        super(Fire,self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, outplanes_1, kernel_size = 1, stride = 1),
        	                       nn.BatchNorm2d(outplanes_1, eps = .001),
        	                       nn.ReLU()
                                  ,nn.Dropout(0.2)) ### Squeeze layer

        self.conv2 = nn.Sequential(nn.Conv2d(outplanes_1, outplanes_2, kernel_size = 1, stride =1),
        	                       nn.BatchNorm2d(outplanes_2,eps = .001),
        	                       nn.ReLU()
                                  ,nn.Dropout(0.2)) ### 1x1 kernel size of Expand layer
        
        self.conv3 = nn.Sequential(nn.Conv2d(outplanes_1,outplanes_2,kernel_size = 3, stride = 1, padding = 1),
        	                      nn.BatchNorm2d(outplanes_2,eps = .001),
        	                      nn.ReLU()
                                  ,nn.Dropout(0.2)) ### 3x3 kernel size of Expand layer

    def forward(self,x):

    	x = self.conv1(x)
    	out1 = self.conv2(x)
    	out2 = self.conv3(x)

    	out = torch.cat([out1,out2],1)

    	return out


class squeezenet(nn.Module):
    def __init__(self):
        super(squeezenet, self).__init__()
        
        self.conv1 =  nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(96, eps = .001),
                                    nn.ReLU(),
                                    nn.Dropout(0.2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fire1 = Fire(96,16,64)
        self.fire2 = Fire(128,16,96)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire3 = Fire(192, 32, 128)
        self.fire4 = Fire(256, 32, 196)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire5 = Fire(392, 64, 196)
        self.fire6 = Fire(392, 64, 224)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire7 = Fire(448, 96, 256)
        self.fire8 = Fire(512, 96, 296)
        self.conv2 = nn.Sequential(nn.Conv2d(592, 2, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(2, eps = .001),
                                    nn.ReLU()
                                   ,nn.Dropout(0.2))
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.maxpool2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool3(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.maxpool4(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        return torch.flatten(x,1)

model = squeezenet().to(device)
print('Archeitiecture of Model',model)






