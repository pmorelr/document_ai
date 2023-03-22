# Import libraries


# Import evaluation functions
from src.models.evaluation_tools import *

# Pass arguments to the script : model, dataset, mode, partition, save, noise_manag
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", default="layoutlm", help="Model to be used for inference (layoutlm, maskrcnn, dit)")
parser.add_argument("-d", "--dataset", default="doclaynet", help="Dataset to be cleaned and structured (doclaynet, docbank or publaynet)")
parser.add_argument("-m", "--mode", default='vision',  help="Type of model that will use the processed dataset (text, vision or multimodal)")
parser.add_argument("-p", "--partition", default='train',  help="Partition of the dataset to be processed (train, test or val)")
parser.add_argument("-s", "--save", default='json',  help="Format that will be used to save the dataset (json, csv, hf)")
parser.add_argument("-n", "--noise_manag", default='all',  help="Noise classes management that will be used to train the model (all, merged or ignored)")
args = vars(parser.parse_args())

MODEL = args['model']
DATASET = args['dataset']
MODE = args['mode']
PART = args['partition']
SAVE_TYPE = args['save']
NOISE_MANAG = args['noise_manag']

# Retrieve test features
# --> We assume that the features have already been built and saved in the data/processed folder

# Perform inference on test data with the chosen model and save the results
# --> TODO : Implement the inference for each model
# LayoutLM = Classification -> Métriques de classification
# MaskRCNN = Detection -> Métriques de détection
# Comparaison uniquement sur les F1-Score

# Evaluate the results
#metrics = evaluate_seqeval(predictions, labels, ner_labels) #-> for LayoutLM

# Plot the metrics
