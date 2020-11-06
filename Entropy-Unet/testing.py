# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:36:07 2020

@author: Savvas
"""
from time import gmtime, strftime
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as skimage
import skimage.measure
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from random import shuffle
import scipy.misc
from PIL import Image
from tqdm import tqdm
import scipy.io
# import voc_dataset
import vanhingen_dataset
import argparse
import logging
import sys
from eval import eval_net, iou
from unet import UNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import subprocess
import custom_transforms
from custom_transforms import CustomToTensor
from PIL.ExifTags import TAGS
from datetime import datetime
from torch.distributions import Categorical
from delentropy import getDelEntropy, getDelEntropyGivenTensor 
from matplotlib import container
from main import getContiniousEntropy
from scipy.stats import pearsonr
from numpy.polynomial.polynomial import polyfit


PALETTE= [
            255, 0, 0,     # Clutter/background
            255, 255, 255, # ground 
            0, 255, 0,     # tree
            255, 255, 0,   # car
            0, 255, 255,   # low veg
            0, 0, 255      # building
]
imageDir = 'E:/Segmentation Images/testmaps'
maskDir = 'E:/Segmentation Images/testmasks-paletted'
checkDir = 'E:/Segmentation Images/checkpoints'
net = UNet(n_channels=3, n_classes=6, bilinear=True).cuda(0)
plt.rcParams["font.family"] = "Times New Roman"
def tensorMapToPILImage(t):
    with torch.no_grad():
        maxes, indices = torch.max(t, dim=0)
        imageToShow = indices.cpu().numpy()
        imageToShow[imageToShow==21] = 255
        eye = Image.fromarray(np.uint8(imageToShow))
        eye.putpalette(PALETTE)
        return eye
    
def maskToPILImage(t):
    t[t==21] = 255
    eye = Image.fromarray(np.uint8(t.cpu()))
    eye.putpalette(PALETTE)
    return eye

imageList = []

rows = 4
cols = 6
axes=[]
fig=plt.figure()
I_LISTI = ['λ = 0', 'λ = 0.1', 'λ = 1.0', 'λ = 2.0']

def showStuff():
    for a in range(rows):
        for b in range(cols):
            # [1 2 3] [4 5 6]
            # print(f'index {(a*cols + b) + 1}')
            axes.append( fig.add_subplot(rows, cols, (a*cols + b) + 1))
            
            if a == 0 and b != cols-1:
                subplot_title=("epoch "+str(b+1))
                axes[-1].set_title(subplot_title)  
            if b == 0:
                axes[-1].set_ylabel(I_LISTI[a], labelpad=15, rotation=0, size='large')
                # print('awe')
            if b == (cols-1):
                #For ground truth
                axes[-1].spines['bottom'].set_color('red')
                axes[-1].spines['top'].set_color('red')
                axes[-1].spines['left'].set_color('red')
                axes[-1].spines['right'].set_color('red')
                axes[-1].spines['bottom'].set_linewidth(3.0)
                axes[-1].spines['top'].set_linewidth(3.0)
                axes[-1].spines['left'].set_linewidth(3.0)
                axes[-1].spines['right'].set_linewidth(3.0)
                if a == 0:
                    axes[-1].set_title('target')  


            # axes[-1].axis('off')  
            axes[-1].set_yticklabels([])
            axes[-1].set_xticklabels([])
            axes[-1].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,  
                left=False,
                right=False,# ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
            
            plt.imshow(imageList[(a*cols + b)])
    # fig.tight_layout(w_pad=0, h_pad=0)  

    # fig.text(0.5, 0.01, 'common xlabel', ha='center', va='center')
    # fig.text(0.01, 0.5, 'common ylabel', ha='center', va='center', rotation='vertical')
    plt.savefig(f'E:/Segmentation Images/seg_res{IMAGE_INDEX}.png', dpi=300)
    # plt.subplots_adjust(wspace=0.2,hspace=0.0001)
    plt.show()

def displayImagesWithPredictions(net_in, ds, imageIndex):
    with torch.no_grad():
        img_sample, mask_sample = ds.__getitem__(imageIndex)
        ##### Display normal image
        inIm = transforms.ToPILImage()(img_sample)
        # inIm.show()
        ##### run image through NN
        t = net_in(img_sample.unsqueeze(0).cuda(0))
        im = tensorMapToPILImage(t.squeeze(0))
        imageList.append(im)
        _, indices = torch.max(t, dim=1)
        # print(iou.(indices, mask_sample))
        ##### Actual mask
        im = maskToPILImage(mask_sample)
        # im.show()
        
        
def appendMask(ds, imageIndex):
    img_sample, mask_sample = ds.__getitem__(imageIndex)
    im = maskToPILImage(mask_sample)
    imageList.append(im)
    # ax.spines['bottom'].set_color('0.5')
    # ax.spines['top'].set_color('0.5')
    # ax.spines['right'].set_color('0.5')
    # ax.spines['left'].set_color('0.5')

def iouCalcs(net_in, ds):
    iouList = []
    with torch.no_grad():
        for img_sample, mask_sample in ds:
            ##### Display normal image
            inIm = transforms.ToPILImage()(img_sample)
            # inIm.show()
            ##### run image through NN
            t = net_in(img_sample.unsqueeze(0).cuda(0))
            im = tensorMapToPILImage(t.squeeze(0))
            imageList.append(im)
            _, indices = torch.max(t, dim=1)
            iouList.append(iou(indices, mask_sample))
            ##### Actual mask
            im = maskToPILImage(mask_sample)
            # im.show()
    return iouList
# want multiple models, show progression of images
modelDirs = [f'contin_ent_lam2.0_r1' ,f'actual_ent_lam0.1_r1',f'contin_ent_lam1.0_r0', 'actual_ent_lam2.0_r1']
# modelsToLoad = [f'{checkDir}/CP_epoch0.pth', f'{checkDir}/CP_epoch1.pth', f'{checkDir}/CP_epoch2.pth']
# IMAGE_INDEX = 102
IMAGE_INDEX = 300
test_set = vanhingen_dataset.VaihingenDataset(imgDirPath=imageDir, gtDirPath=maskDir, shouldTransform=True)


######################## entropy vs del
state = torch.load(f'{checkDir}/{modelDirs[0]}/CP_epoch4.pth')
net.load_state_dict(state['state_dict'])
net.eval()
delEnt = []
oneDEnt = []
c =0 
for img_sample, mask_sample in test_set:
    ##### Display normal image
    inIm = transforms.ToPILImage()(img_sample)
    # inIm.show()
    ##### run image through NN
    t = net(img_sample.unsqueeze(0).cuda(0))
    # print(t.shape)
    _, indices = torch.max(t, dim=1)
    # print(indices)
    delEnt.append(getDelEntropyGivenTensor(indices.squeeze(0)))
    oneDEnt.append(getContiniousEntropy(t).item())
    # print(getContiniousEntropy(t).item())
    ##### Actual mask
    # im = maskToPILImage(mask_sample)
    # im.show()
    # c+=1
    # if c == 4:
    #     break
b, m = polyfit(oneDEnt, delEnt, 1)

corr, _ = pearsonr(oneDEnt, delEnt)
print('Pearsons correlation: %.3f' % corr)

# plt.scatter(oneDEnt, delEnt)
plt.plot(oneDEnt, delEnt, 'o')
plt.plot(oneDEnt, b+m*np.array(oneDEnt), '-', color='r', label=f'r = {round(corr, 4)}')
plt.xlabel('1D Entropy')
plt.ylabel('Delentropy')
plt.title('λ = 0')
plt.legend()
plt.savefig('E:/Segmentation Images/scatter-noent_e4_2.png', dpi=300)
plt.show()

################################# IOU
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlabel('Epochs', fontsize = 16)
# ax.set_ylabel('IoU', fontsize = 16)
# ax.set_ylim([0,1])

# linestyle = {"linestyle":"--", "linewidth":2, "markeredgewidth":2, "elinewidth":1, "capsize":5}


# colourArr = ['b', 'g', 'r', 'y']
# EPOCHS_TO_IOU = 10
# modelDirs = [f'contin_ent_lam2.0_r1' ,f'actual_ent_lam0.1_r1',f'contin_ent_lam1.0_r0']

# customXaxis = range(1,EPOCHS_TO_IOU+1)
# for ind, i in enumerate(modelDirs):     
#     means = []
#     stds = []
#     for e in range(EPOCHS_TO_IOU):
#         state = torch.load(f'{checkDir}/{i}/CP_epoch{e}.pth')
#         net.load_state_dict(state['state_dict'])
#         net.eval()
#         res = iouCalcs(net,test_set)
#         means.append(np.mean(res))
#         stds.append(np.std(res))
#         # means.append(np.mean([0.1,0.2,0.3,0.5,0.6,0.8]))
#         # stds.append(np.std([0.1,0.2,0.3,0.5,0.6,0.8]))

#     ax.errorbar(customXaxis, means, yerr = stds, color=colourArr[ind], label=I_LISTI[ind], **linestyle)
#     plt.xticks(customXaxis)
#     handles, labels = ax.get_legend_handles_labels()
#     handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
#     ax.legend(handles, labels)  
# plt.savefig('E:/Segmentation Images/ioures.png', dpi=300)
# plt.show()

#################################### CODE FOR IMAGES
# for i in modelDirs:       
#     for e in range(cols-1):
#         state = torch.load(f'{checkDir}/{i}/CP_epoch{e}.pth')
#         net.load_state_dict(state['state_dict'])
#         net.eval()
#         displayImagesWithPredictions(net,test_set,IMAGE_INDEX)
#     appendMask(test_set, IMAGE_INDEX)
# showStuff()










