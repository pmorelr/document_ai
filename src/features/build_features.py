from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset
from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import warnings
warnings.simplefilter("ignore")

# Declaration of arguments that will be used 
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default="doclaynet", help="Dataset to be cleaned and structured (doclaynet, docbank or publaynet)")
parser.add_argument("-m", "--mode", default='vision',  help="Type of model that will use the processed dataset (text, vision or multimodal)")
parser.add_argument("-p", "--partition", default='train',  help="Partition of the dataset to be processed (train, test or val)")
parser.add_argument("-s", "--save", default='json',  help="Format that will be used to save the dataset (json, csv, hf)")
args = vars(parser.parse_args())

DATASET = args['dataset']
MODE = args['mode']
PART = args['partition']
SAVE_TYPE = args['save']

# Acessing Dataset
if DATASET == 'doclaynet':
    data_path='../../data/raw/DocLayNet/DocLayNet_core/COCO/'

# Using HuggingFace's function to load the chosen dataset
ds_raw = load_dataset('json', data_files={'train': data_path+'train.json', 'test': data_path+'test.json', 'val': data_path+'val.json'}, field='annotations')
images = load_dataset('json', data_files={'train': data_path+'train.json', 'test': data_path+'test.json', 'val': data_path+'val.json'}, field='images')
categories = load_dataset('json', data_files={'train': data_path+'train.json', 'test': data_path+'test.json', 'val': data_path+'val.json'}, field='categories')

# Previewing the size of the dataset for the chosen partiton
print(f"Dataset size: {len(ds_raw[PART])}, using a total number of {len(images[PART])} images")

# Structuring the dataset per document
bboxes = [[]]
tags = [[]]
img_ids = []
image_path = []

img_id = ds_raw[PART][0]['image_id']
img_ids.append(img_id)

for i in range(len(ds_raw[PART])):

    if ds_raw[PART][i]['image_id'] == img_id:
        bboxes[-1].append(ds_raw[PART][i]['bbox'])
        tags[-1].append(ds_raw[PART][i]['category_id'])
    else:
        bboxes.append([])
        tags.append([])
        img_ids.append(ds_raw[PART][i]['image_id'])

    img_id = ds_raw[PART][i]['image_id']

j = 0
for i in range(len(img_ids)):

    while img_ids[i] != images[PART][j]['id']:
        j +=1
    
    image_path.append(images[PART][j]['file_name'])

# Creating a new dataset based on the previous process
my_dict = {'id': img_ids,
           'bboxes': bboxes,
           'tags': tags,
           'image_path': image_path}

dataset = Dataset.from_dict(my_dict)

# Saving the dataset to the local disk
dataset.to_json(f'{DATASET}_{MODE}_{PART}.{SAVE_TYPE}') 