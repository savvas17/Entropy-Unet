# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:11:44 2020

@author: Savvas
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImagePalette
from custom_transforms import CustomToTensor
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import os
import numpy as np

NUM_CLASSES = 6

vaihingenPalette= [
            255, 0, 0,     # Clutter/background
            255, 255, 255, # ground 
            0, 255, 0,     # tree
            255, 255, 0,   # car
            0, 255, 255,   # low veg
            0, 0, 255      # building
]

def quantizetopalette(silf, palette, dither=False):
    silf.load()
    palette.load()
    if palette.mode != "P":
        raise ValueError("bad mode for palette image")
    if silf.mode != "RGB" and silf.mode != "L":
        raise ValueError(
            "only RGB or L mode images can be quantized to a palette"
            )
    im = silf.im.convert("P", 1 if dither else 0, palette.im)
    try:
        return silf._new(im)
    except AttributeError:
        return silf._makeself(im)

class VaihingenDataset(Dataset):
    def __init__(self, imgDirPath, gtDirPath, shouldTransform=False):
        random.seed(0) # SEED 

        # # PALETTE WRITING
        # while len(vaihingenPalette) != 256*3:
        #     vaihingenPalette.append(0)
        
        # for file in os.listdir(gtDirPath):
        #     img = Image.open(f'{gtDirPath}/{file}')
        #     newImage = Image.new('P', img.size)
        #     newImage.putpalette(vaihingenPalette)
        #     newImage = quantizetopalette(img, newImage, dither=False)
        #     newImage.save(f'E:/Segmentation Images/testmasks-paletted/{file[:-4]}.png')
        #     newImage.close()
            
        self.file_names = []
        for file in os.listdir(imgDirPath):
            self.file_names.append(file[:-4])
        self.imgDirPath = imgDirPath
        self.gtDirPath = gtDirPath
        self.shouldTransform = shouldTransform
        print("Initiated Dataset")
        
    def transform(self, image, mask):
        # if(self.shouldTransform):
            # if random.random() > 0.5:
            #     image = TF.hflip(image)
            #     mask = TF.hflip(mask)
    
            # if random.random() > 0.5:
            #     image = TF.vflip(image)
            #     mask = TF.vflip(mask)
        return CustomToTensor()(image, mask)
        
    def __getitem__(self, index):
        c = self.file_names[index]
        img = Image.open(f'{self.imgDirPath}/{c}.tif')
        m = Image.open(f'{self.gtDirPath}/{c}.png')     
        return self.transform(img, m) # transform will perform custom validations 
    
    def __len__(self):
        return len(self.file_names)
    
def getEntropy(predictedTensor, isOutsideBatch=False, isMask=False):
    if isOutsideBatch:
        predictedTensor = predictedTensor.unsqueeze(0)
        
    if not isMask:
        _, indices = torch.max(predictedTensor, dim=1) #this dimension is the channel dimensions
    else:
        indices = predictedTensor
    totalCount = predictedTensor.shape[-1] * predictedTensor.shape[-2]
    # print(f'ismask {isMask} shape: {indices.shape}')
    uniqueTensor = []
    for t in indices:
        _ , counts = torch.unique(t, return_counts=True, dim=None)
        toPad = NUM_CLASSES - counts.shape[0]
        counts = F.pad(counts, (0, toPad), value=totalCount)
        uniqueTensor.append(counts)
    res = torch.stack(uniqueTensor)
    # totalCount = torch.sum(res[0], dim=0) # counting the number of vals
    
    # print('Stack: ')
    # print(res)
    # print('TotalCount: ')
    # print(totalCount)
    probTensor = 1.0*res / totalCount
    # print('Prob Tensor')
    # print(probTensor)
    inter = - probTensor * torch.log(probTensor) 
    # print('Inter Tensor')
    # print(inter)
    
    entro = torch.sum(inter, dim=1)
    # print('Sum Tensor')
    # print(entro)
    return entro

# d = VaihingenDataset('E:/Segmentation Images/Vanhingen/img', 'E:/Segmentation Images/Vanhingen/masks')
# import skimage.measure
# cartoon = Image.open('E:/Segmentation Images/Vanhingen/tj.jpg').convert('L')
# cartoon = transforms.CenterCrop((200,200))(cartoon)
# print(np.unique(cartoon))
# print(f'cartoon entropy: {skimage.measure.shannon_entropy(cartoon, base=None)}')
# print(f'cartoon entropy: {getEntropy(torch.tensor(np.array(cartoon)), isOutsideBatch=True, isMask=True)}')
# print(f'cartoon entropy: {np.array(cartoon)}')
# for i in range(len(d)):
#     image, mask = d.__getitem__(i)
#     randomArray = np.random.randint(0,6,(200,200))
    # print(randomArray)
    # entropy = getEntropy(torch.tensor(randomArray), True, True)
    # print(f'random entropy: {entropy}')    
    # entropy = getEntropy(mask, True, True)
    # print(f'mask entropy: {entropy}')    
    # if i == 30:
    #     break
    
#     imageH = TF.hflip(image)
#     maskH = TF.hflip(mask)
#     imageV = TF.vflip(image)
#     maskV = TF.vflip(mask)
#     imageHV = TF.vflip(imageH)
#     maskHV = TF.vflip(maskH)
#     imageH.save(f'E:/Segmentation Images/Vanhingen/img/aug_h_{i}.tif')
#     maskH.save(f'E:/Segmentation Images/Vanhingen/masks/aug_h_{i}.png')
#     imageV.save(f'E:/Segmentation Images/Vanhingen/img/aug_v_{i}.tif')
#     maskV.save(f'E:/Segmentation Images/Vanhingen/masks/aug_v_{i}.png')
#     imageHV.save(f'E:/Segmentation Images/Vanhingen/img/aug_hv_{i}.tif')
#     maskHV.save(f'E:/Segmentation Images/Vanhingen/masks/aug_hv_{i}.png')
    # break
# print(i)
# print(m)