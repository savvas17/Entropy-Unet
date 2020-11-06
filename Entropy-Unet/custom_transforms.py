# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:55:45 2020

@author: Savvas
"""
import numpy as np
import torch
import torchvision.transforms as transforms
    
class CustomToTensor(object):
    def __call__(self, img, target):
        m = torch.tensor(np.array(target))
        m[m==255] = 21 #Not predicting the void class
        return transforms.ToTensor()(img), m.long()
    
# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, image, target):
#         for t in self.transforms:
#             image = t(image, target) 
#             target = t(image, target)
#         return image, target