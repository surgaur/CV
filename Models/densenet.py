#### https://towardsdatascience.com/simple-implementation-of-densely-connected-convolutional-networks-in-pytorch-3846978f2f36
#### https://arxiv.org/pdf/1608.06993.pdf
import torch
import torch.nn.functional as F


class Dense_Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Dense_Block,self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 3,
                                             stride = 1,padding =1)
                                  ,nn.BatchNorm2d(out_channels)
                                  ,nn.ReLU())
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = out_channels,out_channels = out_channels,kernel_size = 3,
                                            stride = 1,padding =1)
                                  ,nn.BatchNorm2d(out_channels)
                                  ,nn.ReLU())
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = out_channels*2,out_channels = out_channels,kernel_size = 3,
                                            stride = 1,padding =1)
                                  ,nn.BatchNorm2d(out_channels)
                                  ,nn.ReLU())
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = out_channels*3,out_channels = out_channels,kernel_size = 3,
                                             stride = 1,padding =1)
                                  ,nn.BatchNorm2d(out_channels)
                                  ,nn.ReLU())
        
    def forward(self,x):
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out1 = torch.cat([x1,x2],1)
        x3 = self.conv3(out1)
        out2 = torch.cat([x1,x2,x3],1)
        x4 = self.conv4(out2)
        out3 = torch.cat([x1,x2,x3,x4],1)
        return out3


class Transition_Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Transition_Block,self).__init__()
        
        self.conv_trans = nn.Sequential(nn.Conv2d(in_channels = in_channels,out_channels =out_channels,kernel_size =1 )
                                  ,nn.BatchNorm2d(out_channels)
                                  ,nn.ReLU())
        
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
        
    def forward(self,x):
        
        x = self.conv_trans(x)
        x = self.avg_pool(x)
        
        return x



class Densenet(nn.Module):
    def __init__(self):
        super(Densenet,self).__init__()
        
        self.Dense_block_32 = Dense_Block(3,32)
        self.model_trans_64 = Transition_Block(128,64)
        self.model_Dense_64 = Dense_Block(64,64)
        self.model_trans_96 = Transition_Block(256,96)
        
        self.fc = nn.Linear(96*56*56,2)
    def forward(self,x):
        x = self.Dense_block_32(x)
        x = self.model_trans_64(x)
        x = self.model_Dense_64(x)
        x = self.model_trans_96(x)
        x = x.view(-1,96*56*56)
        x = self.fc(x)
        return torch.sigmoid(x)


model = Densenet().to(device)
model