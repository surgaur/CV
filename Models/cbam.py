class ChannelAttention(nn.Module):
    def __init__(self,in_channel,rr = 8,Module = 'Conv'):
        super(ChannelAttention,self).__init__()
        
        self.Module = Module
        self.rr = rr
        self.average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        
        
        if self.Module:           
        
            self.convblock = nn.Sequential(nn.Conv2d(in_channel,in_channel//rr,1,bias = False),
                                nn.ReLU(),
                                nn.Conv2d(in_channel//rr,in_channel,1,bias = False),
                                nn.Dropout2d(.25))
        else:
            ### Linear Module
            self.Lineaer_Block = nn.Sequential(nn.Linear(in_channel,in_channel//rr),
                                nn.ReLU(),
                                nn.Linear(in_channel//rr,in_channel),
                                nn.Dropout2d(.25))
            
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        _x_ = x
        
        x = self.average_pool(x)
        if self.Module:
            x = self.convblock(x)
        else:
            x = x.view(x.size(0), -1)
            x = self.Lineaer_Block(x)
        
        _x_ = self.max_pool(_x_)
        
        if self.Module:
            _x_ = self.convblock(_x_)
        else:
            _x_ = _x_.view(x.size(0), -1)
            _x_ = self.Lineaer_Block(_x_)
        
        out = x + _x_
        out = self.sigmoid(out)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        
    
        self.conv1 = nn.Conv2d(2, 1, 7,padding = 3 ,bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        x = torch.cat([avg_out,max_out],dim = 1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        
        return x
    
class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        
        out = x * (self.ca(x))
        out = out * (self.sa(out))
        
        return out