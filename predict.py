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



net=M.UNetModel(3,2)
net.cuda()
net.eval()
net.load_state_dict(torch.load("unet_100_addimg3.pth"))



def predition(net,image_name,base_num):
    net.eval()
    img_big=tiff.imread(image_name)
    h,w=img_big.shape[:-1]
    rs=np.zeros((h,w))
    for i in range(h//base_num):
        for j in range(w//base_num):
#             print(i,j)
            if i==h//base_num-1 and j<w//base_num-1:
                new_img=img_big[-base_num*2:,base_num*j:base_num*j+base_num*2,:]
            elif j==w//base_num-1 and i<h//base_num-1:
                new_img=img_big[base_num*i:base_num*i+base_num*2:,-base_num*2:,:]
            elif i==h//base_num-1 and j==w//base_num-1:
                new_img=img_big[-base_num*2:,-base_num*2:,:]
            else:
                new_img=img_big[base_num*i:base_num*i+base_num*2,base_num*j:base_num*j+base_num*2,:]
            new_img=torch.from_numpy(new_img).unsqueeze(0).permute(0,3,1,2).cuda().float()
            pred=net(new_img)
            pred=pred.max(1,keepdim=True)[1].squeeze()
            pred=pred.cpu().detach().numpy()
            dst = measure.label(pred, connectivity=2)
            region=measure.regionprops(dst)
            pred=morphology.remove_small_objects(dst,min_size=50,connectivity=2)
            pred[pred!=0]=4
            pred[pred==0]=2
            if i==h//base_num-1 and j<w//base_num-1:
                rs[-base_num*2:,base_num*j:base_num*j+base_num*2]+=pred
            elif j==w//base_num-1 and i<h//base_num-1:
                rs[base_num*i:base_num*i+base_num*2:,-base_num*2:]+=pred
            elif i==h//base_num-1 and j==w//base_num-1:
                rs[-base_num*2:,-base_num*2:]+=pred
            else:
                rs[i*base_num:i*base_num+base_num*2,j*base_num:j*base_num+base_num*2]+=pred
            rs[rs==2]=0
            rs[rs==3]=1
            rs[rs==4]=1
            rs[rs==5]=1
#
    return rs

#batch prediction
images=glob.glob("shouguang/*")[4:]
print(images)
result="result3/"
for image in images:
    rs=predition(net,image,256)
    cv2.imwrite(result+image.split("/")[-1].split(".")[0]+".png",rs)
    print(result+image.split("/")[-1].split(".")[0]+".png")

