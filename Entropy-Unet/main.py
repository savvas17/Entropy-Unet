# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:32:17 2020

@author: Savvas
"""
from time import gmtime, strftime
import os
# import  cv2
import numpy as np
import matplotlib.pyplot as plt
# import imutils as imutils
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
from eval import eval_net
from unet import UNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import subprocess
import custom_transforms
from custom_transforms import CustomToTensor
from PIL.ExifTags import TAGS
from datetime import datetime
from torch.distributions import Categorical
from delentropy import getDelEntropy
from matplotlib.lines import Line2D 

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'
PALETTE = None
# if(sys.platform == 'win32'):
#     imageDir = 'E:/Segmentation Images/VOC2012/JPEGImages'
#     maskDir = 'E:/Segmentation Images/VOC2012/SegmentationClass'
#     textFilePathTrain ='E:/Segmentation Images/VOC2012/ImageSets/Segmentation/train.txt'
#     textFilePathVal ='E:/Segmentation Images/VOC2012/ImageSets/Segmentation/val.txt'
# else:
#     imageDir = '/mnt/lustre/users/spanagiotou/working_dir/VOC2012/JPEGImages'
#     maskDir = '/mnt/lustre/users/spanagiotou/working_dir/VOC2012/SegmentationClass'
#     textFilePathTrain ='/mnt/lustre/users/spanagiotou/working_dir/VOC2012/ImageSets/Segmentation/train.txt'
#     textFilePathVal ='/mnt/lustre/users/spanagiotou/working_dir/VOC2012/ImageSets/Segmentation/val.txt'
if(sys.platform == 'win32'):
    imageDir = 'E:/Segmentation Images/Vanhingen/img'
    maskDir = 'E:/Segmentation Images/Vanhingen/masks'
else:
    imageDir = '/mnt/lustre/users/spanagiotou/working_dir/Vanhingen/img'
    maskDir = '/mnt/lustre/users/spanagiotou/working_dir/Vanhingen/masks'
  
# def print_memory():
#     t = torch.cuda.get_device_properties(0).total_memory
#     c = torch.cuda.memory_reserved(0)
#     a = torch.cuda.memory_allocated(0)
#     f = c-a  # free inside cache
#     logger.info(f'''
#                     totmem: {t}
#                     cached: {c}
#                     free  : {f}
#                 ''')
####################################################################           
NUM_CLASSES = 6
MAX_PATIENCE = 100
LAMBDA = 1
USE_ENTROPY = True
####################################################################    
       
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def load_checkpoint(model, optimizer, filename):
    start_epoch = 0
    global_step = 0
    if os.path.isfile(filename):
        print("=> loading state '{}'".format(filename))
        state = torch.load(filename)
        start_epoch = state['epoch']
        global_step = state['global_step']
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
    else:
        print("=> no state found at '{}'".format(filename))

    return model, optimizer, start_epoch, global_step

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.0001,
              val_percent=0.1,
              save_cp=True,
              resume='',
              resumeEpoch=1):
    
    # dataset_train = voc_dataset.VOC2012Dataset(imgDirPath=imageDir, gtDirPath=maskDir, textFilePath=textFilePathTrain, shouldTransform=True)
    # dataset_val = voc_dataset.VOC2012Dataset(imgDirPath=imageDir, gtDirPath=maskDir, textFilePath=textFilePathVal, shouldTransform=False)
    dataset_trainVal = vanhingen_dataset.VaihingenDataset(imgDirPath=imageDir, gtDirPath=maskDir, shouldTransform=True)
    # dataset_val = vanhingen_dataset.VaihingenDataset(imgDirPath=imageDir, gtDirPath=maskDir, textFilePath=textFilePathVal, shouldTransform=False)

    # not using val percent
    n_val = int(len(dataset_trainVal) * val_percent)
    n_train = len(dataset_trainVal) - n_val
    train, val = random_split(dataset_trainVal, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # n_train = len(dataset_train)
    # n_val = len(dataset_val)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    writer = SummaryWriter(comment=f'{dir_checkpoint[12:-1]}_LR_{lr}_BS_{batch_size}_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}')

    global_step = 0
    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Resume:          {resume}
        Resume Epoch     {resumeEpoch}
    ''')
    
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    start_epoch = 0
    if(resume != ''):
        net, optimizer, start_epoch, global_step = load_checkpoint(net, optimizer, f'{dir_checkpoint}/CP_epoch{resumeEpoch}.pth')
        net.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    criterion = nn.CrossEntropyLoss(reduction='none') # dont aggregate the loss here
    # criterion = nn.CrossEntropyLoss() # dont aggregate the loss here

    # print_memory()
    
    oldIoU = 0
    patience = MAX_PATIENCE
    
    for epoch in range(start_epoch, epochs):
        net.train()
        epoch_loss = 0
        logger.info(f'epoch: {epoch}')
        for (imgs, true_masks) in train_loader:
            optimizer.zero_grad()
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            masks_pred = net(imgs)
            

            losses = criterion(masks_pred, true_masks) #raw losses for each pixel in image
            mean_of_losses = losses.mean(dim=(1,2)) # sample-wise mean
            if(USE_ENTROPY.strip() == 'True'):
                continiousEntropyPenalty = getContiniousEntropy(masks_pred)
                writer.add_scalar('Entropy/batch_ent_penalty_average', continiousEntropyPenalty.sum().item() / batch_size, global_step)
                loss_each = mean_of_losses + (LAMBDA * continiousEntropyPenalty) # sample-wise addition of entropy
            else:
                loss_each = mean_of_losses # no entropy here
            loss_combined = torch.mean(loss_each) #reducing the loss using the mean
            
            writer.add_scalar('Loss/train_batch/withEntropy', loss_combined.item(), global_step)
            epoch_loss += loss_combined.item()
            
            loss_combined.backward()
            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            val_score_loss, val_score_iou = eval_net(net, val_loader, device)
            # TRAIN SET EVAL train_score_loss, train_score_iou = eval_net(net, train_loader, device) # dont evaluate on the train set
            # scheduler.step(val_score_loss)
            
            logger.info('Validation loss no ent: {}'.format(val_score_loss))
            logger.info('Validation iou: {}'.format(val_score_iou))
            writer.add_scalar('Loss/val/crossentropy_only', val_score_loss, global_step)
            writer.add_scalar('IoU/val', val_score_iou, global_step)
            
            if(val_score_iou > oldIoU):
                oldvalscore = val_score_iou
                patience = MAX_PATIENCE
            elif(patience <= 0):
                state = {'epoch': epoch + 1, 'global_step': global_step, 'state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict()}
                torch.save(state, dir_checkpoint + f'Earlystop_CP_epoch{epoch}_globalstep_{global_step}.pth')
                logger.info(f'Early Stopping on IoU {oldIoU} epoch {epoch}')
                writer.close()
                return
            else:
                patience -= 1
                
            global_step += 1 #increment global step
            
            if save_cp and global_step % 50 == 0:
                try:
                    os.mkdir(dir_checkpoint) #every 50th batch save
                    logger.info('Created checkpoint directory')
                except OSError:
                    pass
                state = {'epoch': epoch + 1, 'global_step':global_step, 'state_dict': net.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, dir_checkpoint + f'CP_epoch{epoch}.pth')
                logger.info(f'Checkpoint {epoch} saved on step {global_step} !')
        writer.add_scalar('Loss/epoch_total/withEntropy', epoch_loss, global_step)
    writer.close()
    state = {'epoch': epochs, 'global_step':global_step, 'state_dict': net.state_dict(),
                         'optimizer': optimizer.state_dict()}
    torch.save(state, dir_checkpoint + f'epochs_finished_CP_epoch{epoch}.pth')

def getContiniousEntropy(predictedTensor):
    totalCount = predictedTensor.shape[-1] * predictedTensor.shape[-2]
    softmaxAccrossClasses = torch.softmax(predictedTensor, dim=1)
    sumSoftmax = (torch.sum(softmaxAccrossClasses, dim=(2,3)))
    probTensor = sumSoftmax / totalCount
    inter = - probTensor * torch.log(probTensor) 
    entro = torch.sum(inter, dim=1)
    return entro
    #This calculation is for entropy focussing on pixel classes rather than binning
    # softmaxAccrossClasses = torch.softmax(predictedTensor, dim=1)
    # print(softmaxAccrossClasses.shape)
    # inter = - softmaxAccrossClasses * torch.log(softmaxAccrossClasses) 
    # condensed = torch.sum(inter, dim=1)
    # entro = torch.mean(condensed, dim=(1,2))
    # return entro
    
    
####################################IGNORE -- NOT DIFFERENTIABLE
def getEntropy(predictedTensor, isOutsideBatch=False, isMask=False):
    if isOutsideBatch:
        predictedTensor = predictedTensor.unsqueeze(0)

    if not isMask:
        indices = torch.argmax(predictedTensor, dim=1) #this dimension is the channel dimensions
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
    # print(res)
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


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=2,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-ent', '--entropy', dest='useEntropy', type=str, default=True,
                        help='use entropy')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=15.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-r', '--runinfo', dest='run', default='',
                        help='folder name of the run')
    parser.add_argument('-c', '--resume', dest='resume', default='',
                        help='resume path')
    parser.add_argument('-ce', '--resumeEpoch', dest='resumeEpoch', type=int, default=1,
                        help='resume epoch')
    parser.add_argument('-lam', '--lambda', dest='lam',type=float, default=1,
                        help='factor to multiply entropy by')
    parser.add_argument('-s', '--seed', dest='seed',type=int, default=0,
                        help='random seed')

    return parser.parse_args()



def getMaskTensorFromOutputTensor(t):
    #shape must be CLASSES x W x H
    maxes, indices = torch.max(t, dim=0)
    imageToShow = indices.cpu().numpy()
    return torch.tensor(imageToShow, dtype=torch.long)


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


def displayImagesWithPredictions(net_in, ds, imageIndex):
    with torch.no_grad():
        img_sample, mask_sample = ds.__getitem__(imageIndex)
        ##### Display normal image
        transforms.ToPILImage()(img_sample).show()
        ##### run image through NN
        t = net_in(img_sample.unsqueeze(0).to(device))
        im = tensorMapToPILImage(t.squeeze(0)).show()
        ##### Actual mask
        im = maskToPILImage(mask_sample).show()

# i = Image.open('E:/Segmentation Images/VOC2012/SegmentationClass/2007_000033.png')
# PALETTE = i.getpalette()
PALETTE= [
            255, 0, 0,     # Clutter/background
            255, 255, 255, # ground 
            0, 255, 0,     # tree
            255, 255, 0,   # car
            0, 255, 255,   # low veg
            0, 0, 255      # building
]

if __name__ == '__main__':
    args = get_args()
    
    if args.run != '':
        dir_checkpoint += args.run + '/' #add subfolder to the checkpoint directory
        
    USE_ENTROPY = args.useEntropy
    LAMBDA = args.lam
    
    torch.manual_seed(args.seed) #seed which changes all randomness except validation set

        
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger(__name__)  
    logger.setLevel(logging.INFO)
    try:
        os.mkdir(dir_checkpoint) #every 50th batch save
    except OSError:
        pass
    file_handler = logging.FileHandler(f'{dir_checkpoint}/logfile.log')
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info(f'Using device {device}')
    
    # print_memory()

    net = UNet(n_channels=3, n_classes=6, bilinear=True)
    logger.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    logger.info(f'LAMBDA: {LAMBDA}')
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logger.info(f'Model loaded from {args.load}')
        
    
    logger.info(f'cuda: {torch.cuda.is_available()}')
    logger.info(f'num cuda devices: {torch.cuda.device_count()}')
    net.to(device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
        
    # EVALUATION
    # state = torch.load('checkpoints/ent_lam1.0_r0/CP_epoch1.pth')
    # net.load_state_dict(state['state_dict'])
    # net.eval()
    # dataset_trainVal = vanhingen_dataset.VaihingenDataset(imgDirPath=imageDir, gtDirPath=maskDir, shouldTransform=True)
    # n_val = int(len(dataset_trainVal) * 0.15)
    # n_train = len(dataset_trainVal) - n_val
    # train, val = random_split(dataset_trainVal, [n_train, n_val])
    # displayImagesWithPredictions(net,val,np.random.randint(0, n_val))
    # displayImagesWithPredictions(net,val,np.random.randint(0, n_val))
    # displayImagesWithPredictions(net,val,np.random.randint(0, n_val))
    logger.info(f"IS USING ENT: {USE_ENTROPY == 'True'}")
    
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  resume=args.resume,
                  resumeEpoch = args.resumeEpoch)
        logger.removeHandler(file_handler)
        del logger,file_handler
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logger.info('Saved interrupt')
        logger.removeHandler(file_handler)
        del logger,file_handler
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
