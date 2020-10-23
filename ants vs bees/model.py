# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:44:18 2020

@author: epocxlabs
"""
import torch.nn as nn
import torchvision.models as models


class cnn_resnet18(nn.Module):
    def __init__(self ):
        super(cnn_resnet18,self).__init__()
        
        self.arch = nn.Sequential(*list(nn.Sequential(*list(models.resnet18(pretrained=True)\
                                                            .children())[:-1]).children())[:-1])
        
        self.fc = nn.Linear( 512 , 2 )
        self.dropout = nn.Dropout(.35)
    def forward(self , x):
        x = self.arch(x)
        x = x.mean(axis=-1).mean(axis=-1) ## Global Average Pooling
        x = self.dropout(self.fc(x))
        
        return x
    
    
class cnn_resnet34(nn.Module):
    def __init__(self ):
        super(cnn_resnet34,self).__init__()
        
        self.arch = nn.Sequential(*list(nn.Sequential(*list(models.resnet34(pretrained=True)\
                                                            .children())[:-1]).children())[:-1])
        
        self.fc = nn.Linear( 512 , 2 )
        
        self.dropout = nn.Dropout(.35)
    def forward(self , x):
        x = self.arch(x)
        x = x.mean(axis=-1).mean(axis=-1) ## Global Average Pooling
        x = self.dropout(self.fc(x))
        
        return x