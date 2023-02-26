from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from features_tools import *
import numpy as np
import warnings
import json
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
    data_path='../../data/raw/DocLayNet/COCO/'
    data_path_text='../../data/raw/DocLayNet/JSON/'

# Using HuggingFace's function to load the chosen dataset
ds_raw = load_dataset('json', data_files={'train': data_path+'train.json', 'test': data_path+'test.json', 'val': data_path+'val.json'}, field='annotations')
images = load_dataset('json', data_files={'train': data_path+'train.json', 'test': data_path+'test.json', 'val': data_path+'val.json'}, field='images')

# Previewing the size of the dataset for the chosen partiton
print(f"Dataset size: {len(ds_raw[PART])}, using a total number of {len(images[PART])} images")

# Structuring the dataset per document
img_ids, bboxes, tags, image_path = vision_features(ds_raw, images, PART)

# Changing bbox convention from (x0, y0, width, height) to (x0, y0, x1, y1)
bboxes = organize_bboxes(bboxes)

# Eliminating blank documents from the dataset
img_ids, bboxes, tags, image_path = eliminate_blank(img_ids, bboxes, tags, image_path)

# Resizing bboxes from 1025x1025 to a 224x224 format
bboxes = resize_bboxes(bboxes, 224/1025.)

# Creating a new dataset based on the previous process
classes = ['Caption', 'Footnote', 'Formula', 'List-Item', 'Page-Footer', 'Page-Header', 'Picture','Section-Header', 'Table', 'Text', 'Title']

my_dict = {'id': img_ids,
           'bboxes': bboxes,
           'tags': tags,
           'image_path': image_path}

dataset = Dataset.from_dict(my_dict)

if MODE == "multimodal":

    words_bb = text_features(data_path_text, image_path)

    img_ids, bboxes, tags, image_path, words = multimodal_features(dataset, words_bb)

    # Eliminating blank documents from the dataset
    img_ids, bboxes, tags, image_path, words = eliminate_blank(img_ids, bboxes, tags, image_path, words)

    bboxes = resize_bboxes(bboxes, 24/1025.)

    my_dict = {'id': img_ids,
            'bboxes': bboxes,
            'words': words,
            'tags': tags,
            'image_path': image_path}

    dataset = Dataset.from_dict(my_dict)


# Saving the dataset to the local disk
dataset.to_json(f'{DATASET}_{MODE}_{PART}.{SAVE_TYPE}') 