Deep Learning Based Document Layout Analysis
==============================
## What is this?
----------
This is a student project of ENSTA Paris in partnership with BNP Paribas focused on Document Layout Analysis using state-of-the-art AI tools. 

This work aims to build an integrated pipeline around the task of classifying entities in documents, using models based on Deep Learning. This comprises the initial steps of extracting features from the datasets we propose to use ([DocLayNet](https://github.com/DS4SD/DocLayNet), [DocBank](https://github.com/doc-analysis/DocBank)), pre-processing the data, training the models, evaluating them; and finally, inference tools with trained models. This same pipeline can be used for fine-tuning other models in the Document AI domain.

Additionally, we conducted a study aimed at the optimal classification of entities that often add noise during document classification, such as headers and footers. For this, we tested different data labeling techniques and made a comparative study about the best methodology to be followed.

----------
### Contributions
----------
Among the activities carried out during this project, we can mention:

- Development of feature extraction scripts from datasets such as [DocLayNet](https://github.com/DS4SD/DocLayNet) and [DocBank](https://github.com/doc-analysis/DocBank).
- Implementation of Mask R-CNN, LayoutLM, LayoutLMv3 and DiT models.
- Development of training, testing and evaluation scripts for the aforementioned models.
- Construction of a pipeline that integrates the different fronts of this project, from data pre-processing to inferences with trained models.
- Conduction a study on the ideal training approach for optimal classification of noisy entities (such as headers and footers).

----------
## How do I run this code?
----------

### 1. Data and Requirements

First, install all the requirements. Then, download [DocLayNet](https://github.com/DS4SD/DocLayNet). Both *core* and *extra* datasets should be downloaded. Their content should be allocated inside `/data/raw/DocLayNet` as follows:

------------
    *
    ├── COCO
    │   ├── test.json
    │   ├── train.json
    │   └── val.json
    ├── PNG
    │   ├── <hash>.png
    │   ├── ...
    ├── JSON
    │   ├── <hash>.json
    │   ├── ...


### 2. Run `run.py`
This is the main script of this project that will allow you to browse the tools and models implemented here, without having to run the base scripts (which is, of course, also an option).

Running this script will give the user 3 options:
- **Feature Extraction**: Preparation of the dataset from its *raw* version to a version with the most adapted structure for model input. You are given the option to get a dataset adapted for vision-only models (Mask R-CNN) or for multimodal models (of the LayoutLM type). It is also possible to select 3 different approaches for noise labeling (default, binary and all). The tools can be found in `/src/features`, where more freedom is given to the user.
- **Training**: Allows you to train the 3 models implemented here, and to select parameters such as the percentage of training data to be used and the number of epochs. Additionally, for models of the LayoutLM type, you are given the possibility to pull the trained model into the HuggingFace Hub. The original scripts can be found in `/src/models`, and the models will be saved locally in `/models`.
- **Evaluation**: Provides tools for analyzing your trained models (or similar models available on HuggingFace's Hub).  You are given the option to select the percentage of the test data that will be used. At the end, graphs and statistics linked to the models' score are provided. These scripts can also be found in `/src/models`.


----------------

## Authors
- [Youssef Ben Cheikh](https://github.com/YoussefBenCheikh)
- [Valentin Collumeau](https://github.com/ValColl)
- [Pedro Morel Rosa](https://github.com/pmorelr)
- [Jacques de Chevron Villette](https://github.com/jchevron)
- [Oumaima Ben Yemna](https://github.com/OumaimaBenYemna)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── train_scripts.py
    │   │   └── evaluation_scripts.py
    │   │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
