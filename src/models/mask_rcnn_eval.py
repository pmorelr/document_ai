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



PATH_TO_MODEL= sys.argv[1]
PATH_TO_TEST_ANNOTATIONS = sys.argv[2]
PATH_TO_IMAGES = sys.argv[3]


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Load your trained model
model = torch.load(PATH_TO_MODEL)
model.to(device)

#Data
TestSet = DocLayNetDataset(PATH_TO_IMAGES, PATH_TO_TEST_ANNOTATIONS)
data_loader_train = torch.utils.data.DataLoader(TestSet, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


#Evaluation
def eval(model, data_loader_valid, device=device)
    cocoEval evaluate(model, data_loader_valid, device=device)
	precision = np.mean(coc.coco_eval['bbox'].eval['precision'][0,:,:,0,2])
    recall = np.mean(coc.coco_eval['bbox'].eval['recall'][0,:,0,2])   
    f1score = (2*precision*recall)/(precision+recall)
    
    return precision, recall, f1score
if __name__ == "__main__":

    precision, recall, f1score = eval(model, data_loader_valid, device=device)
    print("for IoU=0.5, precision:{}, recall: {} and f1score: {}".format(precision, recall, f1score)
    


