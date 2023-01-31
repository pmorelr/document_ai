from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset
from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import warnings
warnings.simplefilter("ignore")

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default="doclaynet", help="Dataset to be cleaned and structured (doclaynet, docbank or publaynet)")
parser.add_argument("-m", "--mode", default='vision',  help="Specification of the type of model that will use the processed dataset (text, vision or multimodal)")
parser.add_argument("-p", "--partition", default='train',  help="Specification of the partition of the dataset to be processed (train, test or val)")
args = vars(parser.parse_args())

DATASET = args['dataset']
MODE = args['mode']
PART = args['partition']

if DATASET == 'doclaynet':
    data_path='../../data/raw/DocLayNet/DocLayNet_core/COCO/'

ds_raw = load_dataset('json', data_files={'train': data_path+'train.json', 'test': data_path+'test.json', 'val': data_path+'val.json'}, field='annotations')
images = load_dataset('json', data_files={'train': data_path+'train.json', 'test': data_path+'test.json', 'val': data_path+'val.json'}, field='images')
categories = load_dataset('json', data_files={'train': data_path+'train.json', 'test': data_path+'test.json', 'val': data_path+'val.json'}, field='categories')

print(f"Dataset size: {len(ds_raw[PART])}, using a total number of {len(images[PART])} images")

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

my_dict = {'id': img_ids,
           'bboxes': bboxes,
           'tags': tags,
           'image_path': image_path}

dataset = Dataset.from_dict(my_dict)

dataset.to_json(f'{DATASET}_{MODE}_{PART}.json') 