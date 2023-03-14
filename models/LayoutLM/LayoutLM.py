from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence, Array2D
from PIL import Image
from functools import partial
from transformers import LayoutLMv2Processor, LayoutLMForTokenClassification
import torch

notebook_login()

PROCESSOR_ID= "microsoft/layoutlmv2-base-uncased"
DATA_PATH = "../data/processed/DocLayNet/multimodal/"
PNG_PATH = "../data/raw/DocLayNet/PNG/"
MODEL_ID = "microsoft/layoutlm-base-uncased"
PROCESSOR_ID =  "microsoft/layoutlmv2-base-uncased"

classes = ['Caption', 'Footnote', 'Formula', 'List-Item', 'Page-Footer', 'Page-Header', 'Picture','Section-Header', 'Table', 'Text', 'Title']

data_features = Features({'id': Value(dtype='int64', id=None),
    'bboxes': Sequence(feature=Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None), length=-1, id=None),
    'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'tags': Sequence(ClassLabel(num_classes=11, names=classes, id=None), length=-1, id=None),
    'image_path': Value(dtype='string', id=None)})

dataset = load_dataset('json', data_files={'test': DATA_PATH+'doclaynet_multimodal_test.json', 'train': DATA_PATH+'doclaynet_multimodal_train.json'}, features=data_features)

labels = dataset[PART].features['tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

features = Features(
    {
        "input_ids": Sequence(feature=Value(dtype="int64")),
        "attention_mask": Sequence(Value(dtype="int64")),
        "token_type_ids": Sequence(Value(dtype="int64")),
        "bbox": Array2D(dtype="int64", shape=(512, 4)),
        "labels": Sequence(ClassLabel(names=labels)),
    }
)

processor = LayoutLMv2Processor.from_pretrained(PROCESSOR_ID, apply_ocr=False)

def process(sample, processor=None):
    encoding = processor(
        images=Image.open(PNG_PATH + sample["image_path"]).convert("RGB"),
        text=sample["words"],
        boxes=sample["bboxes"],
        word_labels=sample["tags"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    del encoding["image"]
    return encoding

# process the dataset and format it to pytorch
proc_dataset = train_batch.map(
    partial(process, processor=processor),
    remove_columns=["image_path", 'words', "tags", "id", "bboxes"],
    features=features,
).with_format("torch")

model = LayoutLMForTokenClassification.from_pretrained(
    MODEL_ID, num_labels=len(labels), label2id=label2id, id2label=id2label
)



import evaluate
import numpy as np

# load seqeval metric
metric = evaluate.load("seqeval")

# labels of the model
ner_labels = list(model.config.id2label.values())


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

from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# hugging face parameter
repository_id = "layoutlm-doclaynet-test"

# Define training args
training_args = TrainingArguments(
    output_dir=repository_id,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    fp16=False,
    learning_rate=3e-5,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="overall_f1",
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=True,
    #hub_private_repo=True,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
)

# Create Trainer instance

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=proc_dataset,
    eval_dataset=proc_dataset,
    compute_metrics=compute_metrics,
    #tokenizer=tokenizer,
)

# Start training
trainer.train()

# change apply_ocr to True to use the ocr text for inference
p#rocessor.feature_extractor.apply_ocr = False

# Save processor and create model card
processor.save_pretrained(repository_id)
trainer.create_model_card()
trainer.push_to_hub()