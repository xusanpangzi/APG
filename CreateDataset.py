#!/usr/bin/env python
# coding: utf-8

# In[60]:


import torch
from torch.utils import data
import os
import cv2
import numpy as np
import random


# In[62]:


class Transform():
    def __init__(self,img_data,label_data):
        self.img_data=img_data
        self.label_data=label_data
    def RandomHorizentalFlip(self,prob):   #水平翻转
        if random.random()<prob:
            self.img_data=self.img_data[:,::-1]
            self.label_data=self.label_data[:,::-1]
        return self.img_data,self.label_data
    def RandomVertialFlip(self,prob):
        if random.random()<prob:
            self.img_data=self.img_data[::-1]
            self.label_data=self.label_data[::-1]
        return self.img_data,self.label_data
    def RandomRotate90(self,prob):    #按对角线翻转
        if random.random()<prob:
            self.img_data=self.img_data.swapaxes(1,0)
            self.img_data=self.img_data[:,::-1]
            self.label_data=self.label_data.swapaxes(1,0)
            self.label_data=self.label_data[:,::-1]
        return self.img_data,self.label_data
    def RandomRotate(self,prob):    #按对角线翻转
        if random.random()<prob:
            #img_data=np.asarray([img_data[:,:,x].T for x in range(img_data.shape[-1])])
            self.img_data=self.img_data.swapaxes(1,0)
            self.label_data=self.label_data.swapaxes(1,0)
        return self.img_data,self.label_data
def create_dataset(imgs,labels):
    img_data=[]
    label_data=[]
    for img,label in zip(imgs,labels):
        img=cv2.imread(img)
        label=cv2.imread(label)
#         trans=Transform(img,label)
#         methods=[trans.RandomHorizentalFlip(0.5),trans.RandomVertialFlip(0.5),trans.RandomRotate90(0.5),trans.RandomRotate(0.5)]
#         if random.random()<0.5:
#             prob=random.randint(0,3)
#             img,label=methods[prob]
        img_data.append(img)
        label_data.append(label[:,:,-1])
    img_data=torch.Tensor(np.array(img_data).astype(np.float32)).permute(0,3,1,2)
    label_data=torch.Tensor(np.array(label_data).astype(np.float32))
    label_data[label_data==torch.unique(label_data)[1]]=1.
    return img_data,label_data
class DataSet(data.Dataset):
    def __init__(self,imgs_data,labels_data):
        self.x_data=imgs_data
        self.y_data=labels_data
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return len(self.x_data)

# In[ ]:




