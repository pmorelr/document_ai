# -*- coding: utf-8 -*-
"""for_train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/186RzCRWCj2nK5SGZIW6SZDAfSiOceFbx

### Imports
"""


# Commented out IPython magic to ensure Python compatibility.
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
import sys
### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.transforms import ToTensor, ToPILImage
# %matplotlib inline

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



"""### Prepare DataSet"""

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class DocLayNetDataset(torch.utils.data.Dataset):
      def __init__(self, imgDir, annotFile):
        self.height = 256
        self.width = 256
        self.imgDir = imgDir
        self.annotations = COCO(annotFile)
        self.imgIDs = self.annotations.getImgIds()
        self.catIDs = self.annotations.getCatIds()
        self.cats = self.annotations.loadCats(self.catIDs)

      def __getitem__(self, index):
        id = self.imgIDs[index]
        #print("id", id)
        img = self.annotations.loadImgs(id)[0]
        #print(img)
        #I = io.imread(imgDir+'/'+img['file_name'])/255.0
        I = Image.open(self.imgDir+'/'+img['file_name']).convert("RGB")
        #print(type(I))
        I = I.resize((self.width, self.height), resample=Image.BILINEAR)
        
        I = np.array(I.getdata()).reshape(I.size[0], I.size[1], 3)
        #print(I.shape)
      
        annIds = self.annotations.getAnnIds(id, self.catIDs, iscrowd=None)
        #print("self.cats", self.cats)
        #print("annIds",annIds)
        #print("self.annotations",self.annotations)
        anns = self.annotations.loadAnns(annIds)
        #print("anns", anns)
        boxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]
        areas = [ann['area']*((self.width/img["width"])*(self.height/img["height"])) for ann in anns]
        iscrowds = [ann['iscrowd'] for ann in anns]
        classes = [cat['name'] for cat in self.cats]
        #binary mask [N,H,W] N number of instances
        normal_mask = np.zeros((img['height'],img['width']))
        for i in range(len(anns)):
          className = getClassName(anns[i]['category_id'], self.cats)
          pixel_value = classes.index(className)+1
          normal_mask = np.maximum(self.annotations.annToMask(anns[i])*pixel_value, normal_mask)
        #print("normal mask", normal_mask.shape)
        
        bin_msk = Image.fromarray(normal_mask)
        bin_msk = bin_msk.resize((self.width, self.height), resample=Image.BILINEAR)
        normal_mask = np.array(bin_msk.getdata()).reshape(bin_msk.size[0], bin_msk.size[1])
        #print("normal", normal_mask.shape)
        #print("normal", np.max(normal_mask))

        binary_mask = np.zeros((len(anns),self.height,self.width))
        for instance in range(len(anns)):
          bin_msk = Image.fromarray(self.annotations.annToMask(anns[instance]))
          bin_msk = bin_msk.resize((self.width, self.height), resample=Image.BILINEAR)
          bin_msk = np.array(bin_msk.getdata()).reshape(bin_msk.size[0], bin_msk.size[1])
          binary_mask[instance,:,:] = bin_msk[:,:]
          
        #print("binary mask", binary_mask.shape)
        
        boxes_np = np.asarray(boxes)*(self.width/img["width"])
        boxes_tr = torch.as_tensor(boxes_np, dtype=torch.float32)
        #print("boxes_tr", boxes)
        labels_tr = torch.as_tensor(labels, dtype=torch.int64)
        #print("labels_tr", labels_tr)
        masks_tr = torch.as_tensor(binary_mask, dtype=torch.uint8)
        #print("masks_tr", masks_tr)
        areas_tr = torch.as_tensor(areas, dtype=torch.float32)
        iscrowds_tr = torch.as_tensor(iscrowds, dtype=torch.uint8)
        target = {}
        target['boxes'] = bx.box_convert(boxes_tr, 'xywh', 'xyxy')
        target['labels']= labels_tr
        target['masks'] = masks_tr
        target['image_id'] = torch.as_tensor(id, dtype=torch.int64)
        target['area'] = areas_tr
        target['iscrowd'] = iscrowds_tr
        

        I = np.transpose(I, [2,0,1])/np.max(I)
        img = torch.tensor(I).float()
        img = img.cuda()
        return img, target


      def __len__(self):
        return len(self.imgIDs)

TraindataSet = DocLayNetDataset('PNG', 'COCO/test.json')

"""### Data Loader"""

data_loader_train = torch.utils.data.DataLoader(TraindataSet, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


def convert_tensor_to_RGB(network_output):
   
    converted_tensor = torch.squeeze(network_output)

    return converted_tensor

model = torch.load("trained_mask-rcnn-5_epochs")
model.eval()


image, target = TraindataSet[0]

lista = [image]
output = model(lista)
labels = output[0]['labels']
#print(labels)
masks = output[0]['masks']
scores = output[0]['scores']
#print(scores)
#print(masks.shape)
output = convert_tensor_to_RGB(output[0].get('masks'))
output_cpu = output.cpu()
output_cpu = torch.nn.functional.softmax(output_cpu, dim=0)

output  = np.argmax(output_cpu.detach().numpy(), axis=0)
#print("after argmax", output.shape, np.min(output), np.max(output))

masko = np.zeros((1,256,256))
lbls = []
for i in range(len(labels)):
	if (scores[i]>0.65):
		masko = np.concatenate([masko, masks[i,:,:].detach().cpu()], axis=0)
		lbls.append(int(labels[i]))
masko = masko[1:,:,:]

maskor = np.where(masko<0.5, 0, 1)
#print("maskor", maskor.shape, np.min(maskor), np.max(maskor))


mask = np.zeros((256,256))

for i in range(maskor.shape[0]):
	mask = np.maximum(maskor[i,:,:]*lbls[i], mask)

#print("mask", mask.shape, np.max(mask), np.min(mask)) 

plt.imsave("pred.png",mask)


# # save the ground truth mask
m = TraindataSet[0][1]['masks'].cpu().detach().numpy()
labels = TraindataSet[0][1]['labels'].cpu().detach().numpy()
new_mask = np.zeros((m.shape[1],m.shape[2]))
for i in range(m.shape[1]):
  for j in range(m.shape[2]):
    if np.max(m[:,i,j]) == 1:
      new_mask[i,j] = labels[np.argmax(m[:,i,j])]
plt.imsave("gtruth.png",new_mask)

# # save original image
image = TraindataSet[0][0].cpu().numpy()
#print(image.shape)
image = np.transpose(image, [1,2,0])
plt.imsave("img.png", image)





