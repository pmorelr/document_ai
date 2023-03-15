from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence, Array2D
from transformers import LayoutLMv2Processor, LayoutLMForTokenClassification, Trainer, TrainingArguments
from huggingface_hub import notebook_login, HfFolder
from PIL import Image, ImageDraw, ImageFont
from functools import partial
import evaluate
import numpy as np
import torch
import os

HF_HUB = False

repository_id = "../../models/LayoutLM/layoutlm-doclaynet" 

if HF_HUB == True:
    hub_repository_id = "layoutlm-doclaynet"
    notebook_login()
    hub_args = dict(
        report_to="tensorboard",
        push_to_hub=True,
        hub_private_repo=True,
        hub_strategy="every_save",
        hub_model_id=hub_repository_id,
        hub_token=HfFolder.get_token()
    )
else:
    hub_args = dict(
        push_to_hub = False
    )

MODEL_ID = "microsoft/layoutlm-base-uncased"
PROCESSOR_ID= "microsoft/layoutlmv2-base-uncased"
DATA_PATH = "../../data/processed/DocLayNet/multimodal/"
PNG_PATH = "../../data/raw/DocLayNet/PNG/"

classes = ['Caption', 'Footnote', 'Formula', 'List-Item', 'Page-Footer', 'Page-Header', 'Picture','Section-Header', 'Table', 'Text', 'Title']

features_raw = Features(
    {'id': Value(dtype='int64', id=None),
    'bboxes': Sequence(feature=Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None), length=-1, id=None),
    'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'tags': Sequence(ClassLabel(num_classes=11, names=classes, id=None), length=-1, id=None),
    'image_path': Value(dtype='string', id=None)})

train_dataset = load_dataset('json', data_files=DATA_PATH+'doclaynet_multimodal_train.json', features=features_raw, split='train[:1%]')
test_dataset = load_dataset('json', data_files=DATA_PATH+'doclaynet_multimodal_test.json', features=features_raw, split='train[:1%]')

labels = train_dataset.features['tags'].feature.names

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
proc_train_dataset = train_dataset.map(
    partial(process, processor=processor),
    remove_columns=["image_path", 'words', "tags", "id", "bboxes"],
    features=features,
).with_format("torch")

proc_test_dataset = test_dataset.map(
    partial(process, processor=processor),
    remove_columns=["image_path", 'words', "tags", "id", "bboxes"],
    features=features,
).with_format("torch")

# -------- Evaluation  ----------
# Load seqeval metric.
metric = evaluate.load("seqeval")

# Labels of the model.
ner_labels = list(model.config.id2label.values())

# Metrics function that will be triggered at the end of each epoch.
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    all_predictions = []
    all_labels = []
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(ner_labels[predicted_idx])
            all_labels.append(ner_labels[label_idx])
    return metric.compute(predictions=[all_predictions], references=[all_labels])

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Defining Training Arguments
training_args = TrainingArguments(
    output_dir=repository_id,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    fp16=False,
    learning_rate=3e-5,
    logging_dir=f"{repository_id}/logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="overall_f1",
    **hub_args
)

# Creating Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=proc_train_dataset,
    eval_dataset=proc_test_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Saving processor locally
processor.save_pretrained(repository_id)

# Pushing ot HuggingFace hub
if HF_HUB == True:
    trainer.create_model_card()
    trainer.push_to_hub()
