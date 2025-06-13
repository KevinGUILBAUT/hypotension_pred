# Hypotension_pred
This project provides a data-driven pipeline to predict intraoperative hypotension using physiological signals from the [VitalDB](https://vitaldb.net/) dataset.

The original implementation was developed by Bob Aubouin [here](https://github.com/BobAubouin/hypotension_pred).
During a research internship, I extended and modified his codebase to explore various classifiers and feature extraction techniques. This repository contains the code corresponding to the most significant and conclusive results.

## Installation

Use a new virtual env and Python 3.11 (with pyenv) for maximal compatibility.

```bash
git clone https://github.com/KevinGUILBAUT/hypotension_pred hp_pred
cd hp_pred
pip install .
```

### Dev / Contribution

In addition, you can add the optional build `dev`. So you will download the Python packages required to develop the project (unit test, linter, formatter).

```bash
git clone https://github.com/KevinGUILBAUT/hypotension_pred hp_pred
cd hp_pred
pip install -e .[dev]
```

## Use

### Download raw data from VitalDB

The data used are from the [VitalDB](https://vitaldb.net/) open dataset. You must read the [Data Use Agreement](https://vitaldb.net/dataset/#h.vcpgs1yemdb5) before using it.

 To download the data you can use the package's command `python -m hp_pred.dataset_download`. The help command outputs the following:

```bash
usage: dataset_download.py [-h] [-l {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}] [-s GROUP_SIZE] [-o OUTPUT_FOLDER]

Download the VitalDB data for hypertension prediction.

options:
  -h, --help            show this help message and exit
  -l {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}, --log_level_name {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}
                        The logger level name to generate logs. (default: INFO)
  -s GROUP_SIZE, --group_size GROUP_SIZE
                        Amount of cases dowloaded and processed. (default: 950)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        The folder to store the data and logs. (default: data)
```

### Create the segmented dataset

The class `hp_pred.databuilder.DataBuilder` is used to create the segmented dataset with a sliding window approach. An example of use is given in the `scripts/dataset_build/base_dataset.py` scripts. If you do not want to use features extracted using linear regression you can check the `scripts/dataset_build/signal_dataset.py` script.

### Recreate JBHI results

The results associated with our paper can be replicated using the version of the git tagged [jbhi_XP](https://github.com/BobAubouin/hypotension_pred/releases/tag/jbhi_XP).

- First download data from VitalDB using the command `python -m hp_pred.dataset_download`. It will download the raw data in the `data/cases` foler.
- Then create the segmented dataset running the script `scripts/dataset_build/30_s_dataset.py`. It will create a new folder in `data/datasets` with the segmented data.
- Train the XGB model using the script `scripts/experiments/train_model.py`, approximately 1h. It will save the model in the `data/models` folder.
- Finally, you can show the results using the notebook `scripts/experiments/show_results.ipynb`.
- Study of the leading time influence can be done using the notebook `scripts/experiments/studyleading_time.ipynb`.

Results might slightly differ due to the randomness of the model.
Note that the results associated with data from Grenoble Hospital can not be replicated as the data is not public.

## Scripts Overview

This project includes a set of Python scripts used for feature extraction, visualization, and model training on physiological signals. Below is a summary of each script's purpose.

### Feature Extraction

- `feature_extraction_pipeline.py`  
  Extracts features from physiological signals using a 4-step pipeline:
  1. Raw regression features from drug-related signals  
  2. Holt's exponential smoothing for physiological forecasts  
  3. Linear regression on smoothed forecasts  
  4. Merging of all features into a unified dataset

- `feature_extraction_pipeline_categorical.py`  
  Variant of `feature_extraction_pipeline.py` that includes:
  - Handling of categorical parameters  
  - Cleaning of missing (NaN) values  
  - Same pipeline with an added preprocessing step

- `features_extraction_reg_raw_signal.py`  
  Extracts trend-based features by applying linear regression over multiple time windows (2, 6, and 20 time steps). Also integrates categorical variables.

- `features_extraction_tsfel.py`  
  Uses the TSFEL library to extract statistical, temporal, and spectral features from physiological signals.

### Visualization & Dimensionality Reduction

- `graphiques.py`  
  Generates a figure illustrating the output of `feature_extraction_pipeline.py` for a specific patient.

- `PCA.py`  
  Applies PCA to reduce the dimensionality of a given dataset. Currently applied to features extracted via TSFEL.

### Model Training & Optimization

- `stacking_model.py`  
  Builds and saves a stacked ensemble classifier using pre-trained models (Random Forest, XGBoost, Rotation Forest), with Logistic Regression as the meta-learner.

- `train_model_new_separation.py`  
  Generates 10 randomized train/test splits and trains a new XGBoost classifier on each training set. Useful for robustness and generalization evaluation.

- `train_super_model.py`  
  Performs 11-fold cross-validated hyperparameter optimization of an XGBoost classifier using Optuna, based on features from `feature_extraction_pipeline.py`.  
  The resulting model is referred to as `SuperModelXGBoost` and is implemented in `src/hp_pred/supermodel.py`.

- `train_XGBoost_model.py`  
  Optimizes a single XGBoost classifier using Optuna on the training set produced by `feature_extraction_pipeline.py`.

## Citation

If you use this code in your research, please cite Bob Aubouin's paper.


## Acknowledgments
Original codebase developed by Bob Aubouin.
Subsequent extensions and experiments carried out during a research internship under Bob's supervision.
