import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
import matplotlib.pyplot as plt




def convolution2d(in_channels,out_channels,kernel_size = 3,**kwargs):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(.2)
    )

def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )



class UNet(nn.Module):
    def __init__(self, input_channels, nclasses):
        super(UNet,self).__init__()
        self.input_channels = input_channels
        self.nclasses = nclasses
        ## Encorder Part
        self.conv1 = convolution2d(in_channels =input_channels,out_channels=32,kernel_size = 3,stride =1,padding =1)
        self.conv2 = convolution2d(in_channels =32,out_channels=64,kernel_size = 3,stride =1,padding =1)
        self.conv3 = convolution2d(in_channels =64,out_channels=128,kernel_size = 3,stride =1,padding =1)
        self.conv4 = convolution2d(in_channels =128,out_channels=256,kernel_size = 3,stride =1,padding =1)
        
        ## Pooling Layeer
        self.pool = nn.MaxPool2d(2,2)
        
        ## Central Park
        self.bottleneck = convolution2d(in_channels =256,out_channels=512,kernel_size = 3,stride =1,padding =1)
        
        ## Decorder part
        self.dconv1 = up_pooling(512,256)
        self.conv5  = convolution2d(in_channels =512,out_channels=256,kernel_size = 3,stride =1,padding =1)
        
        self.dconv2 = up_pooling(256,128)
        self.conv6  = convolution2d(in_channels =256,out_channels=128,kernel_size = 3,stride =1,padding =1)
        
        self.dconv3 = up_pooling(128,64)
        self.conv7 =  convolution2d(in_channels =128,out_channels=64,kernel_size = 3,stride =1,padding =1)
        
        self.dconv4 = up_pooling(64,32)
        self.conv8  = convolution2d(in_channels =64,out_channels=32,kernel_size = 3,stride =1,padding =1)
        
        ## Final Layers
        self.conv_last_layer = nn.Conv2d(in_channels=32, out_channels=nclasses, kernel_size=1)
        
    def forward(self,x):
        x1 = self.conv1(x)
        p1 = self.pool(x1)
        x2 = self.conv2(p1)
        p2 = self.pool(x2)
        x3 = self.conv3(p2)
        p3 = self.pool(x3)
        x4 = self.conv4(p3)
        p5 = self.pool(x4)
        x5 = self.bottleneck(p5)
        
        d1 = self.dconv1(x5)
        d1 = torch.cat([d1,x4],dim=1)
        d1 = self.conv5(d1)
        
        d2 = self.dconv2(d1)
        d2 = torch.cat([d2,x3],dim=1)
        d2 = self.conv6(d2)
        
        d3 = self.dconv3(d2)
        d3 = torch.cat([d3,x2],dim=1)
        d3 = self.conv7(d3)
        
        d4 = self.dconv4(d3)
        d4 = torch.cat([d4,x1],dim=1)
        d4 = self.conv8(d4)
        
        out = torch.sigmoid(self.conv_last_layer(d4))
        
        return out




class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()
        self.smooth = 1
    '''
        def dice_loss(y_pred,labels):
    '''
    #smooth does more than that. You can set smooth to zero and add eps to 
    #the denominator to prevent division by zero. However, having a larger 
    #smooth value (also known as Laplace smooth, or Additive smooth) can be 
    #used to avoid overfitting. The larger the smooth value the closer the following term is to 1
    '''
    smooth = 1
    y_pred = y_pred.view(-1)
    labels = labels.view(-1)
    itercection =  (y_pred * labels).sum()
    union = y_pred.sum() + labels.sum() + smooth
    
    score = 2*(itercection/union)
    return (1-score)

    '''
       
    def forward(self,y_pred,labels):
        y_pred = y_pred.contiguous().view(-1)
        labels = labels.contiguous().view(-1)
        
        itercection =  (y_pred * labels).sum()+self.smooth
        union = y_pred.sum() + labels.sum() + self.smooth
        score = 2*(itercection/union)
        
        return 1.- score