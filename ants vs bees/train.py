# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:45:35 2020

@author: epocxlabs
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import cv2
import os
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from model import cnn_resnet18,cnn_resnet34
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


model_18= cnn_resnet18().to(device)
model_34 = cnn_resnet34( ).to(device)


ant_dir = 'C:\\datasets\\hymenoptera_data\\train\\ants\\'
bee_dir = 'C:\\datasets\\hymenoptera_data\\train\\bees\\'

ant_dir_test = 'C:\\datasets\\hymenoptera_data\\val\\ants\\'
bee_dir_test = 'C:\\datasets\\hymenoptera_data\\val\\bees\\'  

SIZE = 256
batch_size = 16

############### PANDAS DATAFRAME ############

def build_dataframe( ant_dir ,bee_dir ):
    
    features = []
    labels = []
    for img in os.listdir(ant_dir):
        features.append(os.path.join( ant_dir, img ))
        labels.append(1)
        
    for img in os.listdir(bee_dir):
        features.append(os.path.join( bee_dir, img )) 
        labels.append(0)
        
    data = pd.concat([pd.DataFrame(features , columns= ['images' ]) ,
                  pd.DataFrame(labels , columns= ['labels' ])] , 1)
    
    data = data.sample(frac=1).reset_index(drop=True)
    
    assert len(data) == len(features)
    assert pd.isnull(data).sum().any() == False,'Null Value in DataFrame'
    
    return data

df_train = build_dataframe( ant_dir ,bee_dir )
df_test = build_dataframe( ant_dir_test , bee_dir_test ) 

print('Shape of Train Dataset',df_train.shape )
print('Shape of Test Dataset',df_test.shape )



df_train['folds'] = -1
splits = 5
kf = KFold(n_splits=splits,shuffle = False)
for fold,(train_index, val_idx) in enumerate(kf.split(df_train)):
    #df.iloc[train_index,:]['kfold'] = int(fold+1)
    df_train.loc[val_idx,'folds'] = int(fold)

print('Number of Unique folds in dataset',df_train['folds'].unique())

############### Pytorch Data Loader ############

class Create_dataset(Dataset):
    
    def __init__(self, data ,  mode , transform ):
        
        self.data = data
        self.transforms = transform
        self.mode = mode       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        
        img   = self.data.iloc[idx,0] ## Features
        lab   = self.data.iloc[idx,1]  ## labels        
        img   = cv2.imread(img) ## Read Image                
        img   =  self.transforms(image=img)['image']
        
        if self.mode =='train' or self.mode == 'valid':            
            return img , lab
        
        elif self.mode == 'test':
            return img , lab
            
 
    
## Augmentation 

   
transforms_train = A.Compose([
    A.RandomResizedCrop(height=SIZE, width=SIZE, p=1.0),
    A.Flip(),
    A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])        
        
      
transforms_valid = A.Compose([
    A.Resize(height=SIZE, width=SIZE, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])    

transforms_test = A.Compose([
    A.Resize(height=SIZE, width=SIZE, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

trainset = Create_dataset(df_train, 'train' , transforms_train ) 
train_loader = torch.utils.data.DataLoader( trainset ,shuffle=True , batch_size = batch_size )

## Building Model

 
def train(loader, model , optimizer , loss_func ):
    
    model.train()
    train_running_loss = 0.0  
    for index,(feat,label) in enumerate(loader):
        x = feat.to(device)
        y = label.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        output = model(x)
        
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.data.item()* x.size(0)
        
    return train_running_loss / len(loader.dataset)

def valid( loader, model , loss_func ):    
    
    with torch.no_grad():
        model.eval()
        valid_running_loss = 0.0
        num_correct = 0
        total = 0
        
        for val_index,(feat,label) in enumerate(loader):
            x = feat.to(device)
            y = label.to(device)
            output = model(x)
            loss = loss_func(output,y)
            valid_running_loss += loss.item() * x.size(0)
            
            ## Find Corect Labels
            _,preds = torch.max(F.softmax(output,1),1)
            num_correct += torch.sum(torch.eq(preds,y).float()).item()
            total += y.shape[0]
        
        return num_correct/total , preds ,valid_running_loss/len(loader.dataset)
    
    
def test(loader , model):
    correct = 0
    total = 0
    test_labels = []
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, test_pred = torch.max(outputs.data, 1)
            correct += (test_pred == labels).sum().item()
            total += labels.size(0)
            test_labels.append(test_pred.detach().cpu().numpy())
        print('correct: {:d}  total: {:d}'.format(correct, total))
        print('accuracy = {:f}'.format(correct / total))
    
        test_labels = np.concatenate(test_labels)
        return test_labels
    
def run_model(num_epoch , lr , patience,batch_size,model_name):
    
    df_test_pred = pd.DataFrame(np.zeros((len(df_test),splits)))
        
    for fold_num in range(splits):
        print('='*30,'*****','Fold',fold_num,'*****','='*30)
        trn_idx = df_train[df_train['folds'] != fold_num].index
        val_idx = df_train[df_train['folds'] == fold_num].index
        
        df_trn = df_train.loc[trn_idx].reset_index(drop=True)
        df_val = df_train.loc[val_idx].reset_index(drop=True)
        
        ### Train Dataset
            
        trainset = Create_dataset(df_trn, 'train' , transforms_train ) 
        train_loader = torch.utils.data.DataLoader( trainset ,shuffle=True , batch_size = batch_size )
        
        ### Valid Dataset
        
        validset = Create_dataset( df_val , 'valid' , transforms_valid )
        valid_loader = torch.utils.data.DataLoader( validset ,shuffle=False , batch_size = batch_size )
        
        
        ### Define Model
        if model_name == 'resnet18':
            model = cnn_resnet18().to(device)
            
        elif model_name == 'resnet34':
            model = cnn_resnet34().to(device)
            
        print('Using resnet model' , model_name)
    
        for param in model.arch.parameters():
            param.requires_grad = False
    
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of parameters',pytorch_total_params)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=patience, 
                                      verbose=True, factor=0.2)
        best_acc = 0.0
        counter = 0
        for epoch in range(num_epoch):
            print("Epoch: {}/{}.. ".format(epoch+1, num_epoch))
            
            train_loss = train( train_loader, model , optimizer , criterion ) 
            num_correct , preds ,valid_loss = valid( valid_loader, model , criterion )
            print(f'\tTrain Loss: {train_loss:.5f}')
            print(f'\t Val. Loss: {valid_loss:.5f}')
            print(f'\t Accuracy: {num_correct:.3f}')
            
            if num_correct > best_acc:
                best_acc = num_correct
                torch.save(model.state_dict(), f"fold_{fold_num}.pth")
                
            else:
                print('patience starts .........')
                counter +=1
                print('..counter..',counter)
                if (counter >= patience):
                    break;
                    
            scheduler.step(valid_loss)
        
        
    
        print(f'Inferenceing the test data at epoch {epoch+1} and fold {fold_num}')
        model.load_state_dict(torch.load(f"fold_{fold_num}.pth"))
        
        testset = Create_dataset(df_test, 'test' , transforms_test )
        test_loader = torch.utils.data.DataLoader( testset ,shuffle=False , batch_size = batch_size )
        df_test_pred.loc[:,fold_num] = test(test_loader,model)
    
    return df_test_pred

### Model Run
num_epoch = 20
lr = .0005
patience = 3
df_test_pred_res18   = run_model( num_epoch , lr ,patience,batch_size ,'resnet18' )
df_test_pred_res34   = run_model( num_epoch , lr , patience,batch_size,'resnet34' )



'''
import seaborn as sn
d = pd.DataFrame( zip(np.where(np.sum(df_test_pred_res34,1)>2 , 1,0) , np.array(df_test.labels))
             ,columns=['Predict_Label','Label'])


confusion_matrix = pd.crosstab(d['Label'], d['Predict_Label'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True ,   cmap= 'YlGnBu' )
plt.show()
'''