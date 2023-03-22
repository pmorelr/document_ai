from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence, Array2D
from transformers import LayoutLMv2Processor, LayoutLMForTokenClassification
from PIL import Image, ImageDraw, ImageFont
from functools import partial
import evaluate
import numpy as np
import torch
import os

repository_id = "pmorelr/layoutlm-doclaynet" #"../../models/LayoutLM/layoutlm-doclaynet" 

MODEL_ID = repository_id
PROCESSOR_ID= repository_id
DATA_PATH = "../../data/processed/DocLayNet/multimodal/"
PNG_PATH = "../../data/raw/DocLayNet/PNG/"

classes = ['Caption', 'Footnote', 'Formula', 'List-Item', 'Page-Footer', 'Page-Header', 'Picture','Section-Header', 'Table', 'Text', 'Title']

features_raw = Features(
    {'id': Value(dtype='int64', id=None),
    'bboxes': Sequence(feature=Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None), length=-1, id=None),
    'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'tags': Sequence(ClassLabel(num_classes=11, names=classes, id=None), length=-1, id=None),
    'image_path': Value(dtype='string', id=None)})

test_dataset = load_dataset('json', data_files=DATA_PATH+'doclaynet_multimodal_test.json', features=features_raw, split='train[:1%]')

labels = test_dataset.features['tags'].feature.names

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

features = Features(
    {"input_ids": Sequence(feature=Value(dtype="int64")),
     "attention_mask": Sequence(Value(dtype="int64")),
     "token_type_ids": Sequence(Value(dtype="int64")),
     "bbox": Array2D(dtype="int64", shape=(512, 4)),
     "labels": Sequence(ClassLabel(names=labels)),})


processor = LayoutLMv2Processor.from_pretrained(PROCESSOR_ID, apply_ocr=False)

model = LayoutLMForTokenClassification.from_pretrained(
    MODEL_ID, num_labels=len(labels), label2id=label2id, id2label=id2label)

# Preprocess function to prepare the correct encoding for the model.
def process(sample, processor=None):
    encoding = processor(
        images=Image.open(PNG_PATH + sample["image_path"]).convert("RGB"),
        text=sample["words"],
        boxes=sample["bboxes"],
        word_labels=sample["tags"],
        padding="max_length",
        truncation=True,
        max_length=512,)
    
    # As we are using LayoutLMv2's processor, the image encoding will be added. But for LMv1 we do not need it.
    del encoding["image"]
    return encoding

# Actual preprocessing and formating to pytorch.
proc_test_dataset = test_dataset.map(
    partial(process, processor=processor),
    remove_columns=["image_path", 'words', "tags", "id", "bboxes"],
    features=features,
).with_format("torch")

# load seqeval metric
metric = evaluate.load("seqeval")

# labels of the model
ner_labels = list(model.config.id2label.values())
ner_labels = ['B-' + s for s in ner_labels]

def evaluate(encoding, model):

    all_predictions = []
    all_labels = []
    
    outputs = model(input_ids=encoding['input_ids'], bbox=encoding['bbox'], attention_mask=encoding['attention_mask'], token_type_ids=encoding['token_type_ids'])
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    labels = encoding['labels']

    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(ner_labels[predicted_idx])
            all_labels.append(ner_labels[label_idx])

    scores = metric.compute(predictions=[all_predictions], references=[all_labels])

    return scores

s = evaluate(proc_test_dataset, model)


print("\n \n ---------- Printing the dictionary ------- \n \n")
print(s)

print("\n \n ---------- Printing the dict keys ------- \n \n")
print(s.keys())

print("\n \n ---------- Printing the dict values ------- \n \n")
print(s.values())