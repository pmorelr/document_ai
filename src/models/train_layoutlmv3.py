
from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence, Array2D,  Array3D
import torch
import json
from PIL import Image
import numpy as np 
import base64
from transformers import Trainer, TrainingArguments, AutoProcessor, LayoutLMv3ForTokenClassification
from io import BytesIO
import evaluate
import numpy as np
import os

if __name__ == "__main__":
    NOISE_MANAG = 'default'
    N_EPOCHS = 15
    REPOSITORY_ID = 'layoutlmv3-doclaynet-'+NOISE_MANAG
    HF_HUB = False
    HF_HUB_TOKEN = None

    pre_path = '../../'

else:
    pre_path = './'

def run(noise_manag, n_epochs, repository_id, hf_hub=False, hf_hub_token=None):

    local_repository_id = pre_path+'models/LayoutLMv3/'+repository_id

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

    PNG_PATH = pre_path+"data/raw/DocLayNet/PNG/"
    DATA_PATH = pre_path+"data/processed/DocLayNet/"
    PART = 'test'

    file_to_open = pre_path+'data/processed/DocLayNet/doclaynet_multimodal_test_'+noise_manag+'.json'

    if noise_manag == 'default':
        classes = ['Caption', 'Footnote', 'Formula', 'List-Item', 'Page-Footer', 'Page-Header', 'Picture','Section-Header', 'Table', 'Text', 'Title']
        num_classes = 11
    elif noise_manag == 'binary':
        classes = ['Info', 'Noise']
        num_classes = 2
    elif noise_manag == 'triplet':
        classes = ['Info', 'Footnote', 'Page-Footer', 'Page-Header']
        num_classes = 4

    features = Features({'id': Value(dtype='int64', id=None),
        'bboxes': Sequence(feature=Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None), length=-1, id=None),
        'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        'tags': Sequence(ClassLabel(num_classes=num_classes, names=classes, id=None), length=-1, id=None),
        'image_path': Value(dtype='string', id=None)})


    def load_data_from_json(file_to_open) :
        # Open the JSON file
        with open(file_to_open, 'r') as f:
            # Read the contents of the file
            file_contents = f.read()

        # Split the file contents by newline
        json_objects = file_contents.split('\n')

        # # Load each JSON object and store it in a list
        data = []

        json_objects = json_objects[:-1]
        for obj in json_objects:
            if obj.strip() != '':
                data.append(json.loads(obj))
                    
        data_dict_from_data = {
            'id': [d['id'] for d in data],
            'bboxes': [d['bboxes'] for d in data],
            'words': [d['words'] for d in data],
            'tags': [d['tags'] for d in data],
            'image_path': [PNG_PATH+d['image_path'] for d in data]}

        # # select the first 20 images
        data_dict_from_data['image_path'] = data_dict_from_data['image_path']
        data_dict_from_data['id'] = data_dict_from_data['id']
        data_dict_from_data['bboxes'] = data_dict_from_data['bboxes']
        data_dict_from_data['words'] = data_dict_from_data['words']
        data_dict_from_data['tags'] = data_dict_from_data['tags']

        dataset = Dataset.from_dict(data_dict_from_data, features=features)
        return dataset

    dataset = load_data_from_json(file_to_open)
    dataset.to_json(DATA_PATH+'test_multimodal_'+noise_manag+'.json')

    dataset =  load_data_from_json(file_to_open.replace('test', 'val'))
    dataset.to_json(DATA_PATH+'val_multimodal_'+noise_manag+'.json')

    dataset =  load_data_from_json(file_to_open.replace('test', 'train'))
    dataset.to_json(DATA_PATH+'train_multimodal_'+noise_manag+'.json')

    dataset = load_dataset('json', data_files={PART: DATA_PATH+'test_multimodal_'+noise_manag+'.json'}, features=features)
    dataset_val = load_dataset('json', data_files={PART: DATA_PATH+'val_multimodal_'+noise_manag+'.json'}, features=features)
    dataset_train = load_dataset('json', data_files={PART: DATA_PATH+'train_multimodal_'+noise_manag+'.json'}, features=features)


    # we'll use the Auto API here - it will load LayoutLMv3Processor behind the scenes,
    # based on the checkpoint we provide from the hub
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    labels = dataset['test'].features['tags'].feature.names

    features = dataset["test"].features
    column_names = dataset["test"].column_names
    image_column_name = "image_path"
    text_column_name = "words"
    boxes_column_name = "bboxes"
    label_column_name = "tags"

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        id2label = {k: v for k,v in enumerate(label_list)}
        label2id = {v: k for k,v in enumerate(label_list)}
    else:
        label_list = get_label_list(dataset["train"][label_column_name])
        id2label = {k: v for k,v in enumerate(label_list)}
        label2id = {v: k for k,v in enumerate(label_list)}
    num_labels = len(label_list)
    print(label_list)
    print(id2label)


    def prepare_examples(examples):
        images = examples[image_column_name]
        images = [Image.open(d).resize((224,224)) for d in images]
        words = examples[text_column_name]
        boxes = examples[boxes_column_name]
        word_labels = examples[label_column_name]

        encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                            truncation=True, padding="max_length",
                            max_length=512)

        return encoding

    # we need to define custom features for `set_format` (used later on) to work properly
    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        "labels": Sequence(ClassLabel(names=labels)),
    })


    train_dataset = dataset_train["test"].map(
        prepare_examples,
        batched=True,
        batch_size=8,
        remove_columns=column_names,
        features=features,
    )
    eval_dataset = dataset_val["test"].map(
        prepare_examples,
        batched=True,
        batch_size=4,
        remove_columns=column_names,
        features=features,
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=num_labels, label2id=label2id, id2label=id2label)

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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Define training args
    training_args = TrainingArguments(
        output_dir=local_repository_id,
        num_train_epochs=n_epochs,#15
        per_device_train_batch_size=8, # 16
        per_device_eval_batch_size=4, # 8
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
        **hub_args
    )


    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
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
    run(NOISE_MANAG, N_EPOCHS, REPOSITORY_ID)



''' 
test_dataset = dataset['test'].map(
    prepare_examples,   
    batched=True,
    batch_size=4,
    remove_columns=column_names,
    features=features,
).with_format("torch")

predictions, labels, _ = trainer.predict(test_dataset)
dico_metrics = compute_metrics((predictions, labels))
# save it in a json 
with open("metrics.json", "w") as f:
    json.dump(dico_metrics, f)


# same for dataset_val
predictions, labels, _ = trainer.predict(eval_dataset)
dico_metrics = compute_metrics((predictions, labels))
# save it in a json
with open("metrics_val.json", "w") as f:
    json.dump(dico_metrics, f)
'''