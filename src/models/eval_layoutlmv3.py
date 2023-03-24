from datasets import load_dataset, Dataset, Features, Value, ClassLabel, Sequence, Array2D, Array3D
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from PIL import Image, ImageDraw, ImageFont
import terminaltables
import scores_visualization
from functools import partial
import evaluate
import numpy as np
import torch
import os

if __name__ == "__main__":
    NOISE_MANAG = 'default'
    TEST_PART = '1'
    REPOSITORY_ID = 'Jacques2207/all_final_layoutlmv3-base-ner'

    pre_path = '../../'

else:
    pre_path = './'

def run(noise_manag, test_part, repository_id, hf_hub=True):

    if hf_hub == False:
        repository_id = pre_path+'models/LayoutLMv3/'+repository_id

    MODEL_ID = repository_id
    PROCESSOR_ID = repository_id
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

    test_dataset = load_dataset('json', data_files=DATA_PATH+'doclaynet_multimodal_test_'+noise_manag+'.json', features=features_raw, split='train[:'+test_part+'%]')

    image_column_name = "image_path"
    text_column_name = "words"
    boxes_column_name = "bboxes"
    label_column_name = "tags"
    column_names = test_dataset.column_names

    labels = test_dataset.features['tags'].feature.names

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    # we need to define custom features for `set_format` (used later on) to work properly
    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        "labels": Sequence(ClassLabel(names=labels)),
    })


    processor = AutoProcessor.from_pretrained(PROCESSOR_ID, apply_ocr=False)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_ID, num_labels=len(labels), label2id=label2id, id2label=id2label)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def prepare_examples(examples):
        images = examples[image_column_name]
        images = [Image.open(PNG_PATH+d).resize((224,224)) for d in images]
        words = examples[text_column_name]
        boxes = examples[boxes_column_name]
        word_labels = examples[label_column_name]

        encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                            truncation=True, padding="max_length",
                            max_length=512)

        return encoding
    
    proc_test_dataset = test_dataset.map(
        prepare_examples,
        batched=True,
        batch_size=8,
        remove_columns=column_names,
        features=features,
    )


    # load seqeval metric
    metric = evaluate.load("seqeval")

    # labels of the model
    ner_labels = list(model.config.id2label.values())
    ner_labels = ['B-' + s for s in ner_labels]

    def evaluate_test(encoding, model):

        all_predictions = []
        all_labels = []
        
        outputs = model(pixel_values=encoding['pixel_values'], input_ids=encoding['input_ids'], bbox=encoding['bbox'], attention_mask=encoding['attention_mask'])
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

    s = evaluate_test(proc_test_dataset, model)


    s_classif = [
        ['Entity', 'Precision', 'Recall', 'f1-score', 'Number'],
        ['Caption'] + list(s['Caption'].values()),
        ['Footnote'] + list(s['Footnote'].values()),
        ['Page-Footer'] + list(s['Page-Footer'].values()),
        ['Page-Header'] + list(s['Page-Header'].values()),
        ['Picture'] + list(s['Picture'].values()),
        ['Table'] + list(s['Table'].values()),
        ['Text'] + list(s['Text'].values()),
        ['Title'] + list(s['Title'].values()),
    ]

    s_overall = [
        ['Precision', 'Recall', 'Accuracy', 'f1-score'],
        [s['overall_precision'], s['overall_recall'], s['overall_accuracy'], s['overall_f1']]
    ]

    print('\n\n')

    table_classif = terminaltables.DoubleTable(s_classif, 'Scores per Entity')
    print(table_classif.table)

    print('\n\n')

    table_overall = terminaltables.DoubleTable(s_overall, 'Overall Scores')
    print(table_overall.table)

    print('\n\n')

    scores_visualization.plot_scores(s)

if __name__ == "__main__":
    run(NOISE_MANAG, TEST_PART, REPOSITORY_ID)