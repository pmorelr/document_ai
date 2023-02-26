from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence
import numpy as np
import json

def vision_features(raw_dataset, part):
    """
    Extracts and structures features for vision only models. 

    Args:
        raw_dataset (`datasets.dataset_dict.DatasetDict`):
            Original version of the dataset loaded with HuggingFace's load_dataset method.
        part (`str`):
            Specification of the partition of the dataset to be processed. ('train', 'test', 'val').
    Returns:
        img_ids (`list`): 
            Contains an index that corresponds to a specific image.
        bboxes_raw (`list`): 
            Contains sublists with the bounding boxes of a specific image. Each sublist's index is in accordance with img_ids.
        tags (`list`): 
            Contains sublists with the labels of every bounding box of a specific image. Each sublist's index is in accordance with img_ids.
        image_path (`list`): 
            Contains the path to a specific image. Each index is in accordance with the image in every other list.
    """

    bboxes_raw, tags, img_ids, image_path = [[]], [[]], [], []

    img_id = raw_dataset[part][0]['image_id']
    img_ids.append(img_id)

    for i in range(len(raw_dataset[part])):
        if raw_dataset[part][i]['image_id'] == img_id:
            bboxes_raw[-1].append(raw_dataset[part][i]['bbox'])
            tags[-1].append(raw_dataset[part][i]['category_id'] -1)
        else:
            bboxes_raw.append([])
            tags.append([])
            img_ids.append(raw_dataset[part][i]['image_id'])
        img_id = raw_dataset[part][i]['image_id']

    j = 0
    for i in range(len(img_ids)):
        while img_ids[i] != images[part][j]['id']:
            j+=1
        image_path.append(images[part][j]['file_name'])

    return img_ids, bboxes_raw, tags, image_path


def organize_bboxes(bboxes_raw):
    """
    Transforms the representation of bounding boxes from (x0, y0, width, height) to (x0, y0, x1, y1).
    We consider (x0, y0) to be the coordinate on the lower left and (x1, y1) to be the coordinate on
    the upper right of the bounding box.

    Args:
        bboxes_raw (`list`):
            Bounding box data extracted from the raw dataset.
    Returns:
        bboxes (`list`): 
            Modified bounding box data, following the (x0, y0, x1, y1) convention.
    """
    bboxes = []
    for i_doc in range(len(bboxes_raw)):
        bboxes.append([])
        for i_box in range(len(bboxes_raw[i_doc])):
            box = bboxes_raw[i_doc][i_box]
            bboxes[-1].append([box[0], box[1], box[2]+box[0], box[3]+box[1]])
    return bboxes


def text_features(path):
    """
    Extracts the textual data alongside with its respective bounding boxes.

    Args:
        path (`str`):
            Folder path to the textual data.
    Returns:
        words_bb (`list`): 
            List contaning textual data and its respective bounding box, separated by document.
    """
    words_bb = []

    files_words = [path+image_path[i].strip('.png')+'.json' for i in range(len(image_path))]

    for j in range(len(files_words)):
        data = json.load(open(files_words[j], encoding='utf-8'))
        words_bb.append([[data['cells'][i]['text']] + [data['cells'][i]['bbox']] for i in range(len(data['cells']))])
    
    return words_bb

def check_bbox_in(b1, b2):
    """
    Checks if a bounding box is inside the other. Notably important to link annotated data with textual data, 
    as textual data is usually assigned to a bounding box inside the box of annotated data.

    Args:
        b1 (`list`):
            Exterior bounding box to be checked.
        b2 (`list`):
            Interior bounding box to be checked.
    Returns:
        (`bool`): 
    """
    if b1[0]<=b2[0] and b1[1]<=b2[1] and b1[0]+b1[2]>=b2[0]+b2[2] and b1[1]+b1[3]>=b2[1]+b2[3]:
        return True


def multimodal_features(vision_dataset, words_bb):
    """
    Structures features for multimodal models. 

    Args:
        vision_dataset (`datasets.dataset_dict.DatasetDict`):
            Dataset obtained after applying vision_features function to raw dataset, and structured as a HuggingFace DatasetDict.
        words_bb (`list`):
            Textual data alongside with its respective bounding boxes.
    Returns:
        img_ids (`list`): 
            Contains an index that corresponds to a specific image.
        bboxes (`list`): 
            Contains sublists with the bounding boxes of a specific image. Each sublist's index is in accordance with img_ids.
        tags (`list`): 
            Contains an index that corresponds to a specific image.
        bboxes_raw (`list`): 
            Contains sublists with the labels of every bounding box of a specific image. Each sublist's index is in accordance with img_ids.
        image_path (`list`): 
            Contains the path to a specific image. Each index is in accordance with the image in every other list.
        words (`list`): 
            Contains sublists with the text of a specfic bounding box. Each sublist's index is in accordance with img_ids.
    """
    img_ids = []
    image_path = []
    bboxes = []
    tags = []
    words = []

    for i_doc in range(len(vision_dataset)):

        img_ids.append(vision_dataset[i_doc]['id'])
        image_path.append(vision_dataset[i_doc]['image_path'])
        tags.append([])
        bboxes.append([])
        words.append([])

        for i_bb in range(len(vision_dataset[i_doc]['bboxes'])):

            if vision_dataset[i_doc]['tags'][i_bb] in [3,7]:
                tags[-1].append(vision_dataset[i_doc]['tags'][i_bb])
                bboxes[-1].append(vision_dataset[i_doc]['bboxes'][i_bb])
                words[-1].append('')

            else:
                for text in words_bb[i_doc]:
                    if check_bbox_in(vision_dataset[i_doc]['bboxes'][i_bb], text[1]) and text[0] not in ['$', '.', '–', '_', '(', ')', '%', '#']:
                        tags[-1].append(vision_dataset[i_doc]['tags'][i_bb])
                        bboxes[-1].append(text[1])
                        words[-1].append(text[0])

    return img_ids, bboxes_raw, tags, image_path, words 
