from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence, Array2D
from transformers import LayoutLMv2Processor, LayoutLMForTokenClassification, Trainer, TrainingArguments
from huggingface_hub import notebook_login, HfFolder
from PIL import Image, ImageDraw, ImageFont
from functools import partial
import evaluate
import numpy as np
import torch
import os

if __name__ == "__main__":
    NOISE_MANAG = 'triplet'
    TRAIN_PART = '1'
    N_EPOCHS = 15
    REPOSITORY_ID = 'layoutlm-doclaynet-'+NOISE_MANAG
    HF_HUB = True
    HF_HUB_TOKEN = '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'

    pre_path = '../../'

else:
    pre_path = './'


def run(noise_manag, train_part, n_epochs, repository_id, hf_hub=False, hf_hub_token=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    local_repository_id = pre_path+'models/LayoutLM/'+repository_id

    if hf_hub == True:
        hub_args = dict(
            report_to="tensorboard",
            push_to_hub=True,
            hub_private_repo=True,
            hub_strategy="every_save",
            hub_model_id=repository_id,
            hub_token=hf_hub_token
        )
    else:
        hub_args = dict(
            push_to_hub = False
        )

    MODEL_ID = "microsoft/layoutlm-base-uncased"
    PROCESSOR_ID= "microsoft/layoutlmv2-base-uncased"
    DATA_PATH = pre_path+"data/processed/DocLayNet/"
    PNG_PATH = pre_path+"data/raw/DocLayNet/PNG/"

    if noise_manag == 'default':
        classes = ['Caption', 'Footnote', 'Formula', 'List-Item', 'Page-Footer', 'Page-Header', 'Picture','Section-Header', 'Table', 'Text', 'Title']
        num_classes = 11
    elif noise_manag == 'binary':
        classes = ['Info', 'Noise']
        num_classes = 2
    elif noise_manag == 'triplet':
        classes = ['Info', 'Footnote', 'Page-Footer', 'Page-Header']
        num_classes = 4

    features_raw = Features(
        {'id': Value(dtype='int64', id=None),
        'bboxes': Sequence(feature=Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None), length=-1, id=None),
        'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        'tags': Sequence(ClassLabel(num_classes=num_classes, names=classes, id=None), length=-1, id=None),
        'image_path': Value(dtype='string', id=None)})

    train_dataset = load_dataset('json', data_files=DATA_PATH+'doclaynet_multimodal_train_'+noise_manag+'.json', features=features_raw, split='train[:'+train_part+'%]')
    val_dataset = load_dataset('json', data_files=DATA_PATH+'doclaynet_multimodal_val_'+noise_manag+'.json', features=features_raw, split='train[:'+train_part+'%]')

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

    proc_val_dataset = val_dataset.map(
        partial(process, processor=processor),
        remove_columns=["image_path", 'words', "tags", "id", "bboxes"],
        features=features,
    ).with_format("torch")

    # -------- Evaluation  ----------
    # Load seqeval metric.
    metric = evaluate.load("seqeval")

    # Labels of the model.
    ner_labels = list(model.config.id2label.values())
    ner_labels = ['B-' + s for s in ner_labels]

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
        output_dir=local_repository_id,
        num_train_epochs=n_epochs,
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
        eval_dataset=proc_val_dataset,
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

if __name__ == "__main__":
    run(NOISE_MANAG, TRAIN_PART, N_EPOCHS, REPOSITORY_ID, HF_HUB, HF_HUB_TOKEN)