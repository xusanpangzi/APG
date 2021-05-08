#!/usr/bin/env python
# coding: utf-8

import torch 
from torch import nn
import cv2
import glob
from skimage.external import tifffile as tiff
from skimage import measure, color,morphology
from torch.utils import data
import matplotlib.pyplot as plt
import CreateDataset as D
import CreateModel as M
import Evaluation as E
import numpy as np
import random
torch.cuda.set_device(0)
import os
os.chdir("../dp_dataset/")

imgs_list=glob.glob("imgs/*")   #glob是随机的
labels_list=glob.glob("labels/*")
length=len(imgs_list)
print(length,len(labels_list))

inter=int(length*0.7)

train_imgs_list=imgs_list[:inter]
train_labels_list=labels_list[:inter]
test_imgs_list=imgs_list[inter:]
test_labels_list=labels_list[inter:]

train_valid_imgs,train_valid_labels=D.create_dataset(train_imgs_list,train_labels_list)
test_imgs,test_labels=D.create_dataset(test_imgs_list,test_labels_list)
del train_imgs_list,train_labels_list,test_imgs_list,test_labels_list


# # method_one

lossFunc=nn.CrossEntropyLoss()
def train_simple(EPOCHES,X_train,y_train,net):
    loss_value=[]
    train_ls,train_miou=[],[]
    data_set_train=D.DataSet(X_train,y_train)
    data_loader_train=data.DataLoader(data_set_train,batch_size=8,shuffle=True)
    optimizer=torch.optim.Adam(net.parameters(),lr=0.0002,weight_decay=1e-5)
    #optimizer=torch.optim.SGD(net.parameters(),lr=0.0004,momentum=0.9，weight_decay=1e-5)
    for epoch in range(EPOCHES):
        for i,(x,y) in enumerate(data_loader_train):
            x=x.cuda()
            y=y.cuda()
            net=net.train()
            pred_train=net(x)
            loss=lossFunc(pred_train,y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            loss_value.append(loss.item())
#             pred=pred_train.max(1,keepdim=True)[1].squeeze().cpu().detach().numpy()
#             label=y.cpu().detach().numpy()
        print('train epoch:%d/%d,loss:%.4f'%(epoch+1,EPOCHES,loss.item()))
    return loss_value
net=M.UNetModel(3,2)
net.cuda()
loss_value=train_simple(100,train_valid_imgs,train_valid_labels,net)
torch.save(net.state_dict(),'model1/unet_100_addimg3.pth')

with open("../loss.txt","w") as f:
    for l in loss_value:
        f.write(str(l)+'\n')

# # save/use model

# torch.save(net.state_dict(),'model/unet_50_addimg2.pth')
net=M.UNetModel(3,2)
net.cuda()
net.load_state_dict(torch.load("model/unet_100_addimg3.pth"))


# # evaluation

# In[4]:


net.eval()
#train set
data_set_train=D.DataSet(train_valid_imgs,train_valid_labels)
data_loader_train=data.DataLoader(data_set_train,batch_size=8,shuffle=True)
MM,KAPPA,T,U=0,0,0,0
dst_min_size=50
for i,(x,y) in enumerate(data_loader_train):
    x=x.cuda()
    pred=net(x)
    pred=pred.max(1,keepdim=True)[1].squeeze().cpu().detach().numpy()
    
    dst = measure.label(pred, connectivity=2)
    region=measure.regionprops(dst)
    pred=morphology.remove_small_objects(dst,min_size=dst_min_size,connectivity=2)
    
    pred[pred!=0]=1

    label=np.asanyarray(y).squeeze()
    mm=E.get_miou(pred,label)
    kappa=E.get_kappa(pred,label)
    t_acc=E.get_total_acc(pred,label)
    u_acc=E.get_user_acc(pred,label)
    MM+=mm
    KAPPA+=kappa
    T+=t_acc
    U+=u_acc
    print('Train:miou_mean:%.4f,kappa_mean:%.4f,total_acc:%.4f,user_acc:%.4f'%(mm,kappa,t_acc,u_acc))
print('total_miou_mean:%.4f,total_kappa_mean:%.4f,total_total_acc:%.4f,total_user_acc:%.4f'%(MM/(i+1),KAPPA/(i+1),T/(i+1),U/(i+1)))

#tezt set

# X_test,y_test=create_dataset(test_imgs,test_labels,Transform)
data_set_test=D.DataSet(test_imgs,test_labels)
data_loader_test=data.DataLoader(data_set_test,batch_size=4,shuffle=True)
MM,KAPPA,T,U=0,0,0,0
for i,(x,y) in enumerate(data_loader_test):
    x=x.cuda()
    pred=net(x)
    pred=pred.max(1,keepdim=True)[1].squeeze().cpu().detach().numpy()
    
    dst = measure.label(pred, connectivity=2)
    region=measure.regionprops(dst)
    pred=morphology.remove_small_objects(dst,min_size=dst_min_size,connectivity=2)
    pred[pred!=0]=1

    label=np.asanyarray(y).squeeze()
    mm=E.get_miou(pred,label)
    kappa=E.get_kappa(pred,label)
    t_acc=E.get_total_acc(pred,label)
    u_acc=E.get_user_acc(pred,label)
    MM+=mm
    KAPPA+=kappa
    T+=t_acc
    U+=u_acc
    print('Test:miou_mean:%.4f,kappa_mean:%.4f,total_acc:%.4f,user_acc:%.4f'%(mm,kappa,t_acc,u_acc))
print('total_miou_mean:%.4f,total_kappa_mean:%.4f,total_total_acc:%.4f,total_user_acc:%.4f'%(MM/(i+1),KAPPA/(i+1),T/(i+1),U/(i+1)))


# # prediction and combination

# In[3]:


