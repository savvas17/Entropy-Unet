# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 05:27:08 2020

@author: Savvas
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np    

def eval_net(net, loader, device):
    """Evaluation"""
    net.eval()
    n_val = len(loader.dataset)  # the number of batch
    tot = 0
    totalIOU = 0
    for (imgs, true_masks)  in loader:   
        imgs = imgs.to(device)
        true_masks = true_masks.to(device)

        with torch.no_grad():
            mask_pred = net(imgs)
            _, indices = torch.max(mask_pred, dim=1) #take argmax to get class predicted
            totalIOU += iou(indices, true_masks)
            tot += F.cross_entropy(mask_pred, true_masks).item()
    net.train()
    return tot / n_val, (totalIOU / n_val)

def iou(pred, target, n_classes = 6):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(0, n_classes): 
      pred_inds = pred == cls
      target_inds = target == cls
      intersection = (pred_inds[target_inds]).long().sum()  # Cast to long to prevent overflows
      union = pred_inds.long().sum() + target_inds.long().sum() - intersection
      if union == 0:
          ious.append(float( 'nan'))  # If there is no ground truth, do not include in evaluation
      else:
          ious.append(float(intersection) / float(max(union, 1)))
    
    outTensor = torch.tensor(ious)
    outTensor = outTensor[~torch.isnan(outTensor)]
    return outTensor.mean().detach().item()

# SMOOTH = 1e-6
# BATCH x H x W
# def iou_tensors(outputs: torch.Tensor, labels: torch.Tensor):
#     # outputs = outputs.roll(1, dims=2)
#     print('outputs')
#     print(outputs)
#     print('labels')
#     print(labels)
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
#     print(iou)
#     return iou.mean()