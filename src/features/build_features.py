from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from features_tools import *
from noise_management import *
import numpy as np
import warnings
import json
warnings.simplefilter("ignore")

if __name__ == "__main__":
    # Declaration of arguments that will be used 
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", default="doclaynet", help="Dataset to be cleaned and structured (doclaynet, docbank or publaynet)")
    parser.add_argument("-m", "--mode", default='vision',  help="Type of model that will use the processed dataset (text, vision or multimodal)")
    parser.add_argument("-p", "--partition", default='train',  help="Partition of the dataset to be processed (train, test or val)")
    parser.add_argument("-s", "--save", default='json',  help="Format that will be used to save the dataset (json, csv, hf)")
    parser.add_argument("-n", "--noise_manag", default='all',  help="Noise classes management that will be used to train the model (all, binary or triplet)")
    args = vars(parser.parse_args())

    DATASET = args['dataset']
    MODE = args['mode']
    PART = args['partition']
    SAVE_TYPE = args['save']
    NOISE_MANAG = args['noise_manag']

    pre_path='../../'

else:
    pre_path='./'

data_path = pre_path+'data/raw/DocLayNet/COCO/'
data_path_text= pre_path+'data/raw/DocLayNet/JSON/'

def run(DATASET, MODE, PART, SAVE_TYPE, NOISE_MANAG):

    # Using HuggingFace's function to load the chosen dataset
    ds_raw = load_dataset('json', data_files=data_path+PART+'.json', field='annotations', split='train[:100%]')
    images = load_dataset('json', data_files=data_path+PART+'.json', field='images', split='train[:100%]')

    # Previewing the size of the dataset for the chosen partiton
    print(f"Dataset size: {len(ds_raw)}, using a total number of {len(images)} images")

    # Structuring the vision dataset per document
    img_ids_raw, bboxes_raw, areas_raw, tags_raw, image_path_raw = vision_features(ds_raw, images)

    # Declaring a non normalized version of the dataset that will be used for the multimodal dataset structuring
    my_dict = {'id': img_ids_raw,
            'bboxes': bboxes_raw,
            'areas': areas_raw,
            'tags': tags_raw,
            'image_path': image_path_raw}

    non_normalized_dataset = Dataset.from_dict(my_dict)

    if MODE == "vision":

        # Managing noise classes
        noise_managed_dataset = noise_management(non_normalized_dataset, NOISE_MANAG)
        tags = noise_managed_dataset['tags']

        # Changing bbox convention from (x0, y0, width, height) to (x0, y0, x1, y1)
        bboxes = organize_bboxes(bboxes_raw)

        # Eliminating blank documents from the dataset
        img_ids, bboxes, areas, tags, image_path = eliminate_blank(img_ids_raw, bboxes, tags, image_path_raw, areas=areas_raw)

        # Resizing bboxes from 1025x1025 to a 224x224 format
        bboxes = resize_bboxes(bboxes, 1.) #224/1025.

        # Creating a final normalized dataset
        classes = ['Caption', 'Footnote', 'Formula', 'List-Item', 'Page-Footer', 'Page-Header', 'Picture',' Section-Header', 'Table', 'Text', 'Title']

        my_dict = {'id': img_ids,
                'bboxes': bboxes,
                'areas': areas,
                'tags': tags,
                'image_path': image_path}

        dataset = Dataset.from_dict(my_dict)

    if MODE == "multimodal":

        # Extracting bboxes and textual information from JSON folders
        words_bb = text_features(data_path_text, image_path_raw)

        # Structuring the multimodal dataset per document
        img_ids, bboxes, areas, tags, image_path, words = multimodal_features(non_normalized_dataset, words_bb)

        my_dict = {'id': img_ids,
            'bboxes': bboxes,
            'areas': areas,
            'tags': tags,
            'image_path': image_path,
            'words': words}

        multimodal_dataset_raw = Dataset.from_dict(my_dict)

        # Managing noise classes
        noise_managed_dataset = noise_management(multimodal_dataset_raw, NOISE_MANAG)
        tags = noise_managed_dataset['tags']

        # Changing bbox convention from (x0, y0, width, height) to (x0, y0, x1, y1)
        bboxes = organize_bboxes(bboxes)

        # Eliminating blank documents from the dataset
        img_ids, bboxes, areas, tags, image_path, words = eliminate_blank(img_ids, bboxes, tags, image_path, areas=areas, words=words)

        # Resizing bboxes from 1025x1025 to a 224x224 format
        bboxes = resize_bboxes(bboxes, 224/1025.)

        # Creating a final normalized dataset
        my_dict = {'id': img_ids,
                'bboxes': bboxes,
                'words': words,
                'tags': tags,
                'image_path': image_path}

        dataset = Dataset.from_dict(my_dict)


    # Saving the dataset to the local disk
    dataset.to_json(pre_path+f'data/processed/DocLayNet/{DATASET}_{MODE}_{PART}_{NOISE_MANAG}.{SAVE_TYPE}') 
    print('The features were successfully extracted and saved!')

if __name__ == "__main__":
    run(DATASET, MODE, PART, SAVE_TYPE, NOISE_MANAG)