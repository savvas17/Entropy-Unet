# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 10:02:43 2020

@author: Savvas
"""
from skimage import measure
from PIL import Image
import numpy as np
import torch
import cmath
import sys
###########################################################
kernel = np.array([[-1+0j,0+1j],[0-1j,1+0j]])
maxRadius = 2048
radiusMargin = 8
histSize8BPP = 256
###########################################################


#returns a complex number
def MultiplyPixelByKernel(pix, right, below, belowRight):
    return kernel[0][0]*complex(pix,0) + kernel[0][1]*complex(right,0) + kernel[1][0]*complex(below,0) + kernel[1][1]*complex(belowRight,0) 

def FiniteDifferencesGrad(image : torch.Tensor):
    widthClipped = image.shape[-2] - 1 
    heightClipped = image.shape[-2] - 1
    
    minReal = float('inf')
    minImag = float('inf')
    maxReal = float('-inf')
    maxImag = float('-inf')
    maxMod = 0
    gradientPixels = []

    for x in range(heightClipped):
        for y in range(widthClipped):
            complexReturn = MultiplyPixelByKernel(image[x][y], image[x+1][y], image[x][y+1], image[x+1][y+1] )
            gradientPixels.append(complexReturn)
            real = complexReturn.real
            imag = complexReturn.imag
            modsq = real*real + imag*imag
            if modsq > maxMod:
                maxMod = modsq
    
    maxModSqrt = np.sqrt(maxMod)
    
    return gradientPixels, maxModSqrt

def getHistogram(gradientPixels, maxMod):
    #assuming radius is 0
    if maxMod > maxRadius:
        radius = maxRadius
    else:
        radius = int(maxMod + radiusMargin)
    
    binIndex = [None] * len(gradientPixels)
    stride = 2*radius +1
    histDataSize = stride * stride
    histBin = [0] * histDataSize
    factor = 1.0
    histMax = float('-inf')
    if maxMod > radius:
        factor = radius / float(maxMod)
    
    for index, pixel in enumerate(gradientPixels):
        u = np.floor(factor * pixel.real) + radius
        v = np.floor(factor * pixel.imag) + radius
        binIndex[index] =int(v*stride + u)
        histBin[binIndex[index]] += 1
        if histBin[binIndex[index]] > histMax:
            histMax = histBin[binIndex[index]]
    return histBin, histMax, len(gradientPixels)

def delentropy(histBin, histMax, gradientPixelLength):
    binDelentropy = [0] * (histMax + 1)
    total = gradientPixelLength
    delEntropy = 0
    
    for _, bin in enumerate(histBin):
        if bin != 0:
            if binDelentropy[bin] == 0:
                p = bin / float(total)
                binDelentropy[bin] = -1.0 * p * np.log2(p)
            delEntropy += binDelentropy[bin]
    return delEntropy /2.0

def getDelEntropy(image):
    im = torch.tensor(np.array(image, copy=True))
    print(type(im))
    print(im)
    gradientPixels, maxModSqrt =  FiniteDifferencesGrad(im)
    histBin, histMax, gradientPixelLength = getHistogram(gradientPixels, maxModSqrt)
    return delentropy(histBin, histMax, gradientPixelLength)

def getDelEntropyGivenTensor(tens):
    gradientPixels, maxModSqrt =  FiniteDifferencesGrad(tens)
    histBin, histMax, gradientPixelLength = getHistogram(gradientPixels, maxModSqrt)
    return delentropy(histBin, histMax, gradientPixelLength)
# 
# image1 = Image.open('E:/Segmentation Images/Delentropy test/in/ent_jumbled.png')
# image2 = Image.open('E:/Segmentation Images/Delentropy test/in/ent_organized.png')
# image3 = Image.open('E:/Segmentation Images/Delentropy test/in/2g.png')
# print(f'Jumbled pixels delentropy {getDelEntropy(image1)}')
# print(f'Organised pixels delentropy {getDelEntropy(image2)}')
# print(f'Almost empty image delentropy {getDelEntropy(image3)}')

