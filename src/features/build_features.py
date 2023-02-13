from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence
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
    data_path_text='../../data/raw/DocLayNet/DocLayNet_extra/JSON/'

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
        tags[-1].append(ds_raw[PART][i]['category_id'] -1)
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
classes = ['Caption', 'Footnote', 'Formula', 'List-Item', 'Page-Footer', 'Page-Header', 'Picture','Section-Header', 'Table', 'Text', 'Title']

my_dict = {'id': img_ids,
           'bboxes': bboxes,
           'tags': tags,
           'image_path': image_path}

features = Features({'id': Value(dtype='int64', id=None),
'bboxes': Sequence(feature=Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None), length=-1, id=None),
'tags': Sequence(ClassLabel(num_classes=11, names=classes, id=None), length=-1, id=None),
'image_path': Value(dtype='string', id=None)})

dataset = Dataset.from_dict(my_dict, features=features)

if MODE == "multimodal":

    words_bb = []

    files_words = [data_path_text+image_path[i].strip('.png')+'.json' for i in range(len(image_path))]

    for j in range(len(files_words)):
        data = json.load(open(files_words[j], encoding='utf-8'))
        words_bb.append([[data['cells'][i]['text']] + [data['cells'][i]['bbox']] for i in range(len(data['cells']))])

    img_ids = []
    image_path = []
    bboxes = []
    tags = []
    words = []

    for i_doc in range(len(dataset)):

        img_ids.append(dataset[i_doc]['id'])
        image_path.append(dataset[i_doc]['image_path'])
        tags.append([])
        bboxes.append([])
        words.append([])

        for i_bb in range(len(dataset[i_doc]['bboxes'])):

            if dataset[i_doc]['tags'][i_bb] in [3,7]:
                tags[-1].append(dataset[i_doc]['tags'][i_bb])
                bboxes[-1].append(dataset[i_doc]['bboxes'][i_bb])
                words[-1].append('')

            else:
                for text in words_bb[i_doc]:
                    if check_bbox_in(dataset[i_doc]['bboxes'][i_bb], text[1]) and text[0] not in ['$', '.', '–', '_', '(', ')', '%', '#']:
                    tags[-1].append(dataset[i_doc]['tags'][i_bb])
                    bboxes[-1].append(text[1])
                    words[-1].append(text[0])

    my_dict = {'id': img_ids,
            'bboxes': bboxes,
            'words': words,
            'tags': tags,
            'image_path': image_path}

    features = Features({'id': Value(dtype='int64', id=None),
        'bboxes': Sequence(feature=Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None), length=-1, id=None),
        'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        'tags': Sequence(ClassLabel(num_classes=11, names=classes, id=None), length=-1, id=None),
        'image_path': Value(dtype='string', id=None)})

    dataset = Dataset.from_dict(my_dict, features=features)


# Saving the dataset to the local disk
dataset.to_json(f'{DATASET}_{MODE}_{PART}.{SAVE_TYPE}') 