import logging
# Commented out IPython magic to ensure Python compatibility.
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.transforms import ToTensor, ToPILImage
from collections import defaultdict, deque
import datetime
import pickle
import time
import torch.distributed as dist
import errno
import collections
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import boxes as bx
ImageFile.LOAD_TRUNCATED_IMAGES = True
CUDA_LAUNCH_BLOCKING = 1
sys.path.append("./TorchvisionObjectDetection")
from engine import train_one_epoch, evaluate
import utils

sys.path.append(os.path.abspath("."))
from mask_rcnn_utils import *


#
PATH_TO_PRETRAINED_MODEL = sys.argv[1] # 'new' if you want a new model
NUM_EPOCHS = int(sys.argv[2])
PATH_TO_IMAGES = sys.argv[3]
PATH_TO_TRAIN_ANNOTATIONS = sys.argv[4]
PATH_TO_VAL_ANNOTATIONS = sys.argv[5] # "noval" if you don't want validation

MewModel = False
if PATH_TO_PRETRAINED_MODEL == 'new'
    NUM_CLASSES = int(sys.argv[6]) # including background - insert number of classes if you want a new model
    NewModel = True



Validation = (PATH_TO_VAL_ANNOTATIONS!="noval")



#### The Model

if NewModel
    number_classes = NUM_CLASSES #11 classes + background


    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

else:
    #Load your trained model
    model = torch.load(PATH_TO_PRETRAINED_MODEL)

model.to(device)

### DataSets

TrainSet = DocLayNetDataset(PATH_TO_IMAGES, PATH_TO_TRAIN_ANNOTATIONS)
data_loader_train = torch.utils.data.DataLoader(TrainSet, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

if Validation:
    ValSet = DocLayNetDataset(PATH_TO_IMAGES, PATH_TO_VAL_ANNOTATIONS)
    data_loader_valid = torch.utils.data.DataLoader(ValSet, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))    


### Training

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0001)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for n epochs
num_epochs = NUM_EPOCHS
		
for epoch in range(num_epochs):
    try:
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        if Validation:
            evaluate(model, data_loader_valid, device=device)
        torch.save(model,"trained_mask-rcnn-{}epochs".format(epoch+1))

# save the model
#torch.save(model, "trained_mask-rcnn-{}epochs".format(num_epochs))






