#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import albumentations
import pandas as pd
import numpy as np
import io,skimage
from torch.utils.data import Dataset, DataLoader
import os,cv2,time
import gc,collections
import torchvision
import copy
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, models, transforms
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[ ]:


from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report


# In[ ]:


import warnings
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet


# In[ ]:


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1223)


# In[ ]:


csv_file_path = 'C:/Kaggle Datasets/siim-isic-melanoma-classification/train.csv'
train_image_file_path = 'C:/Kaggle Datasets/siim-isic-melanoma-classification/train_256x256/'


# In[ ]:


df = pd.read_csv(csv_file_path)
df.head()


# In[ ]:


_, _, y_train, y_test = train_test_split(df['image_name'], df['target'], test_size=0.2
                                                    , random_state=1223 , stratify = df['target'])


# In[ ]:


image_size = '224'


# In[ ]:


class melanomo_Cancer(Dataset):
    def __init__(self,transform=None):
        self.transform = transform
        self.image_size =image_size
        self.df = pd.read_csv(csv_file_path)
        
        if image_size =='786':
            print('Rnadomly croping image from 786 to 512\n')
            self.aug = albumentations.Compose([
            #albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            #albumentations.RandomCrop(512,512),
            #albumentations.RandomRotate90(p=0.5)
             ])
        else:
            self.aug = albumentations.Compose([
            #albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5)
             ])            
            
        self.Train = True
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        img_id = self.df.loc[index,'image_name']
        label = self.df.loc[index,'target']
        
        img_path = os.path.join(train_image_file_path, img_id+'.jpg')
        image = mpimg.imread(img_path)/255
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        if self.transform is not None:
            image = self.transform(image)
            
        return torch.tensor(image),torch.tensor(label)


# In[ ]:


Batch_size = 64
train_dataset = melanomo_Cancer()
train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
print('Length of dataset is {}'. format(train_dataset.__len__()))
print('Shape of Tensor',train_dataset.__getitem__(0)[0].shape)

############## Train-Test Split
train_sampler = SubsetRandomSampler(y_train.index)
valid_sampler = SubsetRandomSampler(y_test.index)
print('Shape of train and Valid',len(train_sampler),',',len(valid_sampler))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size,sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size,sampler=valid_sampler)
print('Number of Batches in Train',len(train_loader),'and valid',len(validation_loader))


# In[ ]:


fig, axis = plt.subplots(3, 4, figsize=(15, 10))
images, labels = next(iter(train_loader))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        image, label = images[i], labels[i]
        ax.imshow(image.permute(1,2,0)) # add image
        ax.set(title = f"{label}") # add label


# In[ ]:


class AttnBlock(nn.Module):
    def __init__(self,L_in_channels,G_in_channels,factor):
        super(AttnBlock, self).__init__()

        self.factor = factor
        self.W_l = nn.Conv2d(in_channels = L_in_channels,out_channels=256,kernel_size=1)
        self.W_G = nn.Conv2d(in_channels = G_in_channels,out_channels=256,kernel_size=1)
        
        self.phi = nn.Conv2d(in_channels = 256, out_channels = 1, kernel_size = 1)
    def forward(self,l,g):
        N, C, W, H = l.size()
        
        _l = self.W_l(l)
        _g = self.W_G(g)
        
        if True:
            _g  = F.interpolate(_g, scale_factor=self.factor, mode='bilinear', align_corners=False)
        _sum = F.relu(_l + _g)
        c = self.phi(_sum)
        if True:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        f = torch.mul(a.expand_as(l),l)
        #output = f.view(N,C,-1).sum(dim=2)
        return f


# In[ ]:


class Resnet_with_Attn(nn.Module):
    def __init__(self):
        super(Resnet_with_Attn, self).__init__()
        self.arch = models.resnet18(pretrained=True)
    
        self.conv_block_1 = nn.Sequential(*list(self.arch.children())[0:6])
        self.conv_block_2 = nn.Sequential(*list(self.arch.children())[6:7])
        self.conv_block_3 = nn.Sequential(*list(self.arch.children())[7:8])
        
        ### Attention Class # 
        self.attn1 = AttnBlock(128,512,4)
        self.attn2 = AttnBlock(256,512,2)
        
        ### Linear Classifier Block
        
        self.fc1= nn.Linear(896,32)
        self.fc2= nn.Linear(32,1)

        
        
    def forward(self,x):
        x = self.conv_block_1(x)
        _attn_1 = x
        x = self.conv_block_2(x)
        _attn_2 = x       
        x = self.conv_block_3(x)
        
        a1 = self.attn1(_attn_1,x)
        a2 = self.attn2(_attn_2,x)
        
        g1 = a1.view(*(a1.shape[:-2]),-1).mean(-1)
        g2 = a2.view(*(a2.shape[:-2]),-1).mean(-1)
        g = x.view(*(x.shape[:-2]),-1).mean(-1)

        out = torch.cat([g,g1,g2],dim=1)
        
        out = self.fc1(out)
        classifier = self.fc2(out)
        
        return classifier


# In[ ]:


model = Resnet_with_Attn()
print('Archeitiecture of Model',model.to(device))


# In[ ]:


for param in model.arch.parameters():
    param.requires_grad = False


# In[ ]:


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters',pytorch_total_params)


# In[ ]:


optimizer = torch.optim.Adam(params = model.parameters() , lr = .001)
loss_func = nn.BCEWithLogitsLoss()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_losses = []\nvalid_losses = []\nepochs = 3\nmodel_weights = []\n#print("*"*30,f\'Model Name {model_name}\',f\'using size {image_size}\',"*"*30)\n\nfor epoch in range(epochs):\n    print(\'=\'*100)\n    print("Epoch: {}/{}.. ".format(epoch+1, epochs))\n    \n    train_running_loss = 0\n    \n    model.train()\n    for batch_index,(images,labels) in enumerate(train_loader):\n        images,labels = images.to(device),labels.to(device).unsqueeze(1).float()\n        optimizer.zero_grad()        \n        output = model(images)\n        loss = loss_func(output,labels)\n        loss.backward()\n        optimizer.step()\n        train_running_loss +=loss.item()\n    \n    with torch.no_grad():\n        valid_running_loss = 0\n        model.eval()\n        org_labels = list()\n        pred_labels = list()\n        predictive_indices = []\n        roc_auc = 0\n        for index,(images,labels) in enumerate(validation_loader):\n            images,labels = images.to(device),labels.to(device).unsqueeze(1).float()\n            output =  torch.sigmoid(model(images))\n            _,preds = torch.max(output,1)\n            loss = loss_func(output,labels)\n            valid_running_loss+=loss.item()\n            \n            ## Finding ROC- AUC curve\n            org_labels.append(labels)\n            pred_labels.append(output)\n            predictive_indices.append(preds)\n        roc_auc = roc_auc_score(torch.cat(org_labels).detach().cpu(),torch.cat(pred_labels).detach().cpu())\n    \n    train_losses.append(train_running_loss/len(train_loader))\n    valid_losses.append(valid_running_loss/len(validation_loader))\n    \n    print(f"Training Loss: {train_running_loss/len(train_loader):.3f}.. ",\n         f"Validation Loss: {valid_running_loss/len(validation_loader):.3f}.. ",\n         f"ROC-AUC: {roc_auc:.3f}.. ",\'\\n\',\n         f"Classification Report\\n: {classification_report(torch.cat(org_labels).detach().cpu() , torch.cat(predictive_indices).detach().cpu())}"\n         )\n    \n    if epoch>=7:\n        model_weights.append(copy.deepcopy(model.state_dict())) ### Adding weihts in list')


# In[ ]:


new_state_dict = collections.OrderedDict()
for key in model.state_dict():
    if 'num_batches_tracked' in key:
        param = model_weights[0][key]
    else:
        param = torch.mean(torch.stack([sd[key] for sd in model_weights]), dim=0)
    new_state_dict[key] = param

# Load into model
model.load_state_dict(new_state_dict)


# In[ ]:


csv_file_path = 'C:/Kaggle Datasets/siim-isic-melanoma-classification/test.csv'
test_image_file_path = 'C:/Kaggle Datasets/siim-isic-melanoma-classification/680469_1195048_compressed_test/'


# In[ ]:


class test_melanomo_Cancer(Dataset):
    def __init__(self,transform=None):
        self.transform = transform
        self.df = pd.read_csv(csv_file_path)
        self.aug = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            #albumentations.Resize(224,224),
            #albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_id = self.df.loc[index,'image_name']
        img_path = os.path.join(test_image_file_path, img_id+'.png')
        image = mpimg.imread(img_path)/255
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        if self.transform is not None:
            image = self.transform(image)
            
        return torch.tensor(image)


# In[ ]:


Batch_size = 64
test_dataset = test_melanomo_Cancer()
test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)
print('Length of dataset is {}'. format(test_dataset.__len__()))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'labels = list()\nwith torch.no_grad():\n    model.eval()\n    for index,(images) in enumerate(test_loader):\n        images = images.to(device)\n        output =  torch.sigmoid(model(images))\n        labels.append(output)')


# In[ ]:


submission_csv_file = 'C:/Kaggle Datasets/siim-isic-melanoma-classification/sample_submission.csv'
submission = pd.read_csv(submission_csv_file)
submission.head()


# In[ ]:


submission['target'] = np.array(torch.cat(labels).detach().cpu())


# In[ ]:


submission[submission['target']>.2]


# In[ ]:


submission.to_csv('sub.csv', index=False)


# In[ ]:



submission.sample(20)


# In[ ]:




