from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence, Array2D
from transformers import LayoutLMv2Processor, LayoutLMForTokenClassification, Trainer, TrainingArguments
from huggingface_hub import notebook_login, HfFolder
from PIL import Image, ImageDraw, ImageFont
from functools import partial
#import evaluate
import numpy as np
import torch
import os

HF_HUB = False
OCR = False
IMAGE_INDEX = 20

#repository_id = "../../models/LayoutLM/layoutlm-doclaynet" 
DATA_PATH = "../../data/processed/DocLayNet/multimodal/"
PNG_PATH = "../../data/raw/DocLayNet/PNG/"

classes = ['Caption', 'Footnote', 'Formula', 'List-Item', 'Page-Footer', 'Page-Header', 'Picture','Section-Header', 'Table', 'Text', 'Title']

features_raw = Features({'id': Value(dtype='int64', id=None),
    'bboxes': Sequence(feature=Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None), length=-1, id=None),
    'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'tags': Sequence(ClassLabel(num_classes=11, names=classes, id=None), length=-1, id=None),
    'image_path': Value(dtype='string', id=None)})

test_dataset = load_dataset('json', data_files=DATA_PATH+'doclaynet_multimodal_test.json', features=features_raw, split='train[:1%]')

model = LayoutLMForTokenClassification.from_pretrained("pmorelr/layoutlm-doclaynet-test")
processor = LayoutLMv2Processor.from_pretrained("pmorelr/layoutlm-doclaynet-test")

def unnormalize_box(bbox, width, height, OCR):
    if OCR == True:
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]
    else: 
        return [
           (bbox[0] / (224/1025.)),
           (bbox[1] / (224/1025.)),
           (bbox[2] / (224/1025.)),
           (bbox[3] / (224/1025.)),
        ]    

label2color = {
    "Caption": "brown",
    "Footnote": "gray",
    "Formula": "magenta",
    "List-Item": "purple",
    "Page-Footer": "black",
    "Page-Header": "black",
    "Picture": "orange",
    "Section-Header": "yellow",
    "Table": "green",
    "Text": "red",
    "Title": "blue"}

# draw results onto the image
def draw_boxes(image, boxes, predictions, OCR):
    width, height = image.size
    normalizes_boxes = [unnormalize_box(box, width, height, OCR) for box in boxes]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(predictions, normalizes_boxes):
        #draw.rectangle(box, outline="black")
        draw.rectangle(box, outline=label2color[prediction])
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image


# run inference
def run_inference(path, model=model, processor=processor, output_image=True, OCR=False):
    # create model input
    image = Image.open(path).convert("RGB")
    if OCR == True:
        encoding = processor(image, return_tensors="pt")
        del encoding["image"]
    else:
        for i in range(len(test_dataset)):
            if PNG_PATH+test_dataset[i]['image_path'] == path:
                words = test_dataset[i]['words']
                boxes_float = test_dataset[i]['bboxes']
                boxes = [[round(b[0]), round(b[1]), round(b[2]), round(b[3])] for b in boxes_float]
                tags = test_dataset[i]['tags']

        encoding = processor(
        image,
        words,
        boxes=boxes,
        word_labels=tags,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        )

        del encoding["image"]

    # run inference
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()

    # get labels
    labels = [model.config.id2label[prediction] for prediction in predictions]
    if output_image:
        return draw_boxes(image, encoding["bbox"][0], labels, OCR)
    else:
        return labels

img1 = run_inference(PNG_PATH+test_dataset[IMAGE_INDEX]["image_path"])
img1.save('result.png')