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


# function to get masks from bboxes
def ann2mask(bboxes, width=256, height=256, mode='xyxy'):    
    nb_instances = len(bboxes)
    if mode=='xywh':
        for i in range(len(bboxes)):
            box = bboxes[i]
            bboxes[i]=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
    #print(bboxes)
    mask = np.zeros((nb_instances, width, height))
    for n in range(nb_instances):
        bbox = bboxes[n]
        for i in range(width):
            if i>=bbox[0] and i<=bbox[2]:
                for j in range(height):
                    if j>=bbox[1] and j<=bbox[3]:
                        mask[n,i,j]=1
    return mask
    
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
		
        target['masks'] = torch.as_tensor(ann2mask(target['boxes']))
		
        return img, target
        
        

    
