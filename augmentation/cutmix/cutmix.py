# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:32:03 2021

@author: epocxlabs
"""

import torch
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

img_1 = plt.imread('C:\\Users\\epocxlabs\\cassava scripts\\casava data\\train_images\\336550.jpg')
img_2 = plt.imread('C:\\Users\\epocxlabs\\cassava scripts\\casava data\\train_images\\42688414.jpg')
img_3 = plt.imread('C:\\Users\\epocxlabs\\cassava scripts\\casava data\\train_images\\57149651.jpg')

plt.imshow(img_1)
plt.imshow(img_2)
plt.imshow(img_3)

image_batch  = np.array([ img_1 , img_1 , img_3 ])
image_labels = np.array( [ [1,0,0] , [0,1,0] , [0,0,1] ] )



def bounding_box(image,lambs):
    
    h = img_1.shape[0]
    w = img_1.shape[1]
    
    cut_rate = np.sqrt(1-lambs)

    cut_w = np.int(cut_rate*w)
    cut_h = np.int(cut_rate*h)
    
    cy = np.random.randint(h)
    cx = np.random.randint(w)
    
    bbx1 = np.clip( cy - cut_h//2,0,h )
    bby1 = np.clip( cx - cut_w//2,0,w )
    bbx2 = np.clip( cy + cut_h//2,0,h )
    bby2 = np.clip( cx + cut_w//2,0,w )
    
    '''
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    '''
    return bbx1,bbx2,bby1,bby2


'''
image = cv2.rectangle( img_1 , (bbx1,bby1) , (bbx2,bby2) , (255, 0, 0),3 )
plt.imshow(image)
'''
def cutmix():
    
    rand_index = np.random.permutation(len(image_batch))
    alpha = 1
    lambdas = np.clip(np.random.beta(alpha,alpha) , .2 , .5 )
    
    bbx1,bbx2,bby1,bby2 = bounding_box( img_1,lambdas ) ## img_1 shape
    
    ## Cutmix on images
    updated_image_batch = image_batch.copy()
    updated_image_batch[:,bby1:bby2 , bbx1:bbx2,:] = image_batch[ rand_index ,bby1:bby2 , bbx1:bbx2,:]
    updated_image_labels = image_labels[rand_index]
    
    ## Cutmix on labels
    
    lamb = 1 - ((bbx2-bbx1)*(bby2-bby1))/(img_1.shape[0] * img_1.shape[1])
    updated_image_labels = image_labels * lamb + updated_image_labels * (1. - lamb)
    
    return updated_image_labels , updated_image_batch


updated_image_labels , updated_image_batch  = cutmix()



plt.imshow(updated_image_batch[0])

plt.imshow(updated_image_batch[2])

plt.imshow(updated_image_batch[1])









