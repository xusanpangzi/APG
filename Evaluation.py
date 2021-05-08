#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.metrics import confusion_matrix
import numpy as np


# In[3]:


def get_confusion(pred,label):   #混淆矩阵
    pred_flat=pred.flatten()
    label_flat=label.flatten()
    confusion=confusion_matrix(label_flat,pred_flat)
    return confusion
def get_miou(pred,label):    #交并比
    classes=len(np.unique(label))
    confusion=get_confusion(pred,label)
    miou_mean=sum([confusion[c,c]/(confusion[c][c]+np.sum(confusion[c,:])+np.sum(confusion[:,c])-2*confusion[c,c]) for c in range(classes)])/classes
    return miou_mean

def get_kappa(pred,label):   #kappa系数
    classes=len(np.unique(label))
    confusion=get_confusion(pred,label)
    p0=np.trace(confusion)/np.sum(confusion)
    pe=sum([np.sum(confusion[i,:])*np.sum(confusion[:,i]) for i in range(classes)])/pow(np.sum(confusion),2)
    kappa=(p0-pe)/(1-pe)
    return kappa
def get_total_acc(pred,label):   #总体精度
    confusion=get_confusion(pred,label)
    total_acc=np.trace(confusion)/np.sum(confusion)
    return total_acc
def get_user_acc(pred,label):     #用户精度
    confusion=get_confusion(pred,label)
    user_acc=confusion[1,1]/np.sum(confusion[1,:])
    return user_acc


# In[ ]:




