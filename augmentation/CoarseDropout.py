import torch
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
def coarseDropout(num_holes,length,prob,img):    
    
    if np.random.random() > prob:
        
        h,w = img.size(1) , img.size(2)
        #mask = np.ones((h, w), np.float32)
        
        for i in range(num_holes):
            y = np.random.randint(h)
            x =  np.random.randint(w)
            
            y1 = np.clip(y-length//2,0,h)
            y2 = np.clip(y+length//2,0,h)


            x1 = np.clip(x-length//2,0,w)
            x2 = np.clip(x+length//2,0,w)
            
            mask = np.ones((h, w), np.float32)
            mask[x1:x2,y1:y2] = 0
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
            
        return img
    else:
        return img
    
coarseDropout(50,10,.5,'img_name')

## Fpr Ploting
'''
import matplotlib.pyplot as plt
# Viewing data examples used for training
fig, axis = plt.subplots(3, 3, figsize=(15, 10))
images, labels = next(iter(train_loader))

for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        image, label = coarseDropout(50,10,.5,images[i]), labels[i]
        #image, label = images[i], labels[i]
        image = image.permute(2,1,0).detach().cpu().numpy() # 
        ax.imshow(image.astype('uint8')) # add image
        
        ax.set(title = f"{label}") # add label

'''