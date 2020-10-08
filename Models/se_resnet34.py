
import torch
from torch import nn, optim
import torchvision

class SE_Block(nn.Module):
    def __init__(self,in_channel,rr):
        super(SE_Block,self).__init__()
        '''
    _input = torch.randn(1, 2, 3, 3)
    m = nn.AvgPool2d((_input.shape[2:]))
    output = m(_input)
    result = torch.reshape(output,output.shape[0:2])
    result.shape is replaced by {x = x.view(*(x.shape[:-2]),-1).mean(-1)}
       '''
        
        self.rr = rr ## reduction-ration
        self.in_channel =in_channel
        
        self.linear_1 = nn.Linear(in_channel,in_channel//rr)
        self.linear_2 = nn.Linear(in_channel//rr,in_channel)
        
    def forward(self,x):
        
        _x = x
        
        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = torch.relu(self.linear_1(x))
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        
        out = _x*x
        return out



class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34,self).__init__()
        
        self.arch = arch = models.resnet34(pretrained=True)
        #fc_layer  = torch.nn.Sequential(*list(arch.children())[9:])
        
        self.arch_layer = torch.nn.Sequential(*list(arch.children())[:8])
        self.se_ = SE_Block(512,16)
        self.adaptive_pooling = torch.nn.Sequential(*list(arch.children())[8:9])
        self.fc1 = nn.Linear(torch.nn.Sequential(*list(arch.children())[9:])[0].in_features,1024)
        self.fc2 = nn.Linear(1024,32)
        self.fc3 = nn.Linear(32,1)
        
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        
        x = self.arch_layer(x)
        x = self.se_(x)
        x = self.adaptive_pooling(x)
        x = x.view(-1,512*1*1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x



model = Resnet34()
print('Archeitiecture of Model',model.to(device))



for param in model.arch_layer.parameters():
    param.requires_grad = False
    
for param in model.fc1.parameters():
    param.requires_grad = False


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters',pytorch_total_params)