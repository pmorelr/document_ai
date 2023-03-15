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

# Prepare the DataSet

class DocLayNetDataset(torch.utils.data.Dataset):
	def __init__(self, imgDir, annotFile, original_img_size=1025):
		"""
		imgDir: Directory where documents are stored
		annotFile: Path to the annotations file + the name
	
		"""
		self.original_img_size = original_img_size
        	self.height = 256
        	self.width = 256
        	self.imgDir = imgDir 
        	self.annotations = load_dataset('json', data_files={'data': annotFile})['data']
        	
        	
	def __len__(self):
		return self.annotations.['data'].num_rows
        	
        def __getitem__(self, index):
		document = self.annotations[index]
		I = Image.open(self.imgDir+'/'+document['image_path']).convert("RGB")
		I = I.resize((self.width, self.height), resample=Image.BILINEAR)
		I = np.array(I.getdata()).reshape(I.size[0], I.size[1], 3)
		I = np.transpose(I, [2,0,1])/np.max(I)
		img = torch.tensor(I).float()
		img = img.cuda()
		
		target = {}
		
		target['image_id'] = torch.as_tensor(document['id'], dtype=torch.int64)
		
		bboxes_np = np.asarray(document['bboxes'])*(self.width/self.original_img_size)
		bboxes_tr = torch.as_tensor(bboxes_np, dtype=torch.float32)
		target['boxes'] = bboxes_tr
		
		labels_np =  np.asarray(document['tags'])
		labels_tr = torch.as_tensor(labels, dtype=torch.int64)
		target['labels']= labels_tr
        
		areas_np = np.asarray(document['areas'])*((self.width*self.height)/self.original_img_size**2)
		areas_tr = torch.as_tensor(areas_np, dtype=torch.float32)
		target['area'] = areas_tr
		
		target['masks'] = Ann_to_mask(document)
		
		return img, target
		
		
#### The Model

number_classes = 12 #11 classes + background


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


# or Load your trained model
#torch.load(model_path)


### Training

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0001)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for n epochs
num_epochs = 5
		
for epoch in range(num_epochs):
        try:
        # train for one epoch, printing every 10 iterations
        	train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        	lr_scheduler.step()
        	evaluate(model, data_loader_valid, device=device)
        	torch.save(model,"trained_mask-rcnn-{}".format(epoch+1))

# save the model
torch.save(model, "trained_mask-rcnn-{}".format(num_epochs))
		
		
		
