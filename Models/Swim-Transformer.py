## https://python.plainenglish.io/swin-transformer-from-scratch-in-pytorch-31275152bf03
## Above link help me to build all the below code , Please reference above link for better understanding .
class PatchEmbedding(nn.Module):
    def __init__(self,img_size , in_channels = 3 , patch_size=4 , C = 96):
        super().__init__()
        
        '''Split images into patches and embed them.
    img_size : int (size of the image)
    patch_size : int (size of the image)
    in_chans : int (Number of input channel for RGB it is 3 and gray_scals its is 1)
    embed_dim : int (hyperparameter)
        '''
    
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size//patch_size)**2
        self.conv = nn.Conv2d(in_channels
                 , C
                 , kernel_size = patch_size
                 , stride=patch_size,)
        
        self.layer_norm = nn.LayerNorm(C)
        self.relu = nn.ReLU()
    def forward(self,x):
        '''
        input: batch_size,in_channels,img_size,img_size
        output:batch_size,embed_dim,n_patches
        
        '''
        x = self.conv(x)
        x = x.flatten(2) ## flatten will change (batch_size,embed_dim,patch_size,patch_size) into (batch_size,embed_dim,n_patches)
        x = x.transpose(1,2)
        x = self.relu(self.layer_norm(x))
        return x
    
	
class PatchMerging(nn.Module):

    '''
    input shape -> (b, (h*w), C)
    output shape -> (b, (h/2 * w/2), C*2)
    '''

    def __init__(self,C):
        super().__init__()
        self.linear = nn.Linear(4*C, 2*C)
        self.layer_norm = nn.LayerNorm(2*C)

    def forward(self, x):
        height = width = int(math.sqrt(x.shape[1])/2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)', s1=2, s2=2, h=height, w=width)
        return self.layer_norm(self.linear(x))
		
		

class ShiftedWindowMSA(nn.Module):


 #input shape -> (b,(h*w), C)
 #output shape -> (b, (h*w), C)


    def __init__(self, embed_dim, window_size=7):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.proj1 = nn.Linear(embed_dim, 3*embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        
    def forward(self,x):
        
        height = width = int(math.sqrt(x.shape[1]))
        
        x = self.proj1(x)
        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K=3, h=height, w=width)
        x = rearrange(x, 'b (h m1) (w m2) E K -> b h w (m1 m2) E K', m1= self.window_size , m2=self.window_size)
        
        q,k,v = x.chunk(3,dim = -1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        
        energy = torch.matmul(q,k.transpose(3,4))
        #energy = energy/math.sqrt(embed_dim)
        energy = torch.softmax(energy,-1)
        
        att = torch.matmul(energy,v)
        out = rearrange(att , 'b h w (m1 m2) C -> b (h m1) (w m2) C',m1 = self.window_size , m2 = self.window_size)
        out = rearrange(out , 'b h w C -> b (h w) C')
        
        return out
		
		
pe = PatchEmbedding(img_size = 224)
pm = PatchMerging(C = 96)
sw = ShiftedWindowMSA(embed_dim  = 96)


x = pe(img)
print('lvl1 Patch Embedding -->' , x.shape)
x = sw(x)
print('lvl1 Swifted Window -->', x.shape)
x = pm(x)
print('lvl1 Patch Merge -->', x.shape)