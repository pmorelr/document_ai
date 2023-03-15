Deep Learning Based Document Layout Analysis
==============================

## What is this?

This is a student project of ENSTA Paris in partnership with BNP Paribas focused on Document Layout Analysis using state-of-the-art AI tools. 

This work aims to build an integrated pipeline around the task of classifying entities in documents, using models based on Deep Learning. This comprises the initial steps of extracting features from the datasets we propose to use ([DocLayNet](https://github.com/DS4SD/DocLayNet), [DocBank](https://github.com/doc-analysis/DocBank)), pre-processing the data, training the models, evaluating them; and finally, inference tools with trained models. This same pipeline can be used for fine-tuning other models in the Document AI domain.

Additionally, we conducted a study aimed at the optimal classification of entities that often add noise during document classification, such as headers and footers. For this, we tested different data labeling techniques and made a comparative study about the best methodology to be followed.

### Contributions

Among the activities carried out during this project, we can mention:

- Development of feature extraction scripts from datasets such as [DocLayNet](https://github.com/DS4SD/DocLayNet) and [DocBank](https://github.com/doc-analysis/DocBank).
- Implementation of Mask R-CNN, LayoutLM, LayoutLMv3 and DiT models.
- Development of training, testing and evaluation scripts for the aforementioned models.
- Construction of a pipeline that integrates the different fronts of this project, from data pre-processing to inferences with trained models.
- Conduction a study on the ideal training approach for optimal classification of noisy entities (such as headers and footers).

----------
## TODO: How do I run this code?
----------

The processed data can be found in [this link](https://drive.google.com/drive/folders/167k-SrAM5qpCO3hkSZHenA6Z4Bqk3YB_?usp=sharing)

----------------

## Authors
- [Youssef Ben Cheikh](https://github.com/YoussefBenCheikh)
- [Valentin Collumeau](https://github.com/ValColl)
- [Pedro Morel](https://github.com/pmorelr)
- [Oumaima Ben Yemna](https://github.com/OumaimaBenYemna)
- [Jacques de Chevron Villette](https://github.com/jchevron)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
