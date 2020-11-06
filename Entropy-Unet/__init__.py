# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 12:40:29 2020

@author: Savvas
"""
import torch
import torch.nn as nn
from model_parts import *
from unet import UNet
from voc_dataset import VOC2012Dataset
