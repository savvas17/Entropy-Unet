# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 07:52:12 2020

@author: Savvas
"""
from PIL import Image
import os 

from PIL import Image

infile = 'E:/Segmentation Images/Vanhingen/img/top_mosaic_09cm_area1.tif'
chopsize = 250

img = Image.open(infile)
width, height = img.size

# Save Chops of original image
for x0 in range(0, width, chopsize):
   for y0 in range(0, height, chopsize):
      box = (x0, y0,
             x0+chopsize if x0+chopsize <  width else  width - 1,
             y0+chopsize if y0+chopsize < height else height - 1)
      print('%s %s' % (infile, box))
      img.crop(box).save(f'E:/Segmentation Images/izee/zcho{x0},{y0}.tif')
            
            
