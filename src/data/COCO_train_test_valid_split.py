import json
from pathlib import Path
import funcy
from sklearn.model_selection import train_test_split

# Source : https://github.com/dnfwlxo11/cocosplit_train_test_valid/blob/master/cocosplit_train_test_valid.py


def save_coco(file, dest_folder, info, licenses, images, annotations, categories):
    # Create destination folder if it doesn't exist
    if not Path(dest_folder).exists():
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
    # Save the COCO datasets to the destination folder
    with open(Path(dest_folder, file), 'w') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def split(input_annotations, dest_folder, train_ratio = 0.8, valid_ratio = 0.1, test_ratio = 0.1, trainJson_name = 'train.json', validJson_name = 'valid.json', testJson_name = 'test.json', remove_images_without_annotations = True):
    """
    Splits the COCO dataset into train, valid and test sets.
    
    Parameters
    ----------
    input_annotations : str
        Path to the COCO annotations file.
    dest_folder : str
        Path to the destination folder.
    train_ratio : float
        Ratio of the dataset to be used for training.
    valid_ratio : float
        Ratio of the dataset to be used for validation.
    test_ratio : float
        Ratio of the dataset to be used for testing.
    trainJson_name : str
        Name of the output file for the training set.
    validJson_name : str
        Name of the output file for the validation set.
    testJson_name : str
        Name of the output file for the test set.
    remove_images_without_annotations : bool
        If True, removes images without annotations from the dataset.
    
    Returns
    -------
    None
    
    """
    # Load the COCO annotations file
    with open(input_annotations) as coco_file:
        coco = json.load(coco_file)
    info = coco['info']
    licenses = coco['licenses']
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    number_of_images = len(images)

    images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

    if remove_images_without_annotations:
        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

    train_before, test = train_test_split(
        images, test_size=test_ratio)

    ratio_remaining = 1 - test_ratio
    ratio_valid_adjusted = valid_ratio / ratio_remaining

    train_after, valid = train_test_split(
        train_before, test_size=ratio_valid_adjusted)

    save_coco(trainJson_name, dest_folder, info, licenses, train_after, filter_annotations(annotations, train_after), categories)
    save_coco(testJson_name, dest_folder, info, licenses, test, filter_annotations(annotations, test), categories)
    save_coco(validJson_name, dest_folder, info, licenses, valid, filter_annotations(annotations, valid), categories)

    print("Saved {} entries in {} and {} in {} and {} in {}".format(len(train_after), trainJson_name, len(test), testJson_name, len(valid), validJson_name))