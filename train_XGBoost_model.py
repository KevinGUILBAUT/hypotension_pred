from pathlib import Path

import optuna
import pandas as pd
import xgboost as xgb

from hp_pred.experiments import objective_xgboost

optuna.logging.set_verbosity(optuna.logging.WARNING)



model_filename = "xgb_30_s_test_04_06.json"


#Import the data already processed
test = pd.read_parquet("data/features_extraction_test_categoricals/test.parquet")
train = pd.read_parquet("data/features_extraction_test_categoricals/train.parquet")

# control reproducibility
rng_seed = 42

FEATURE_NAME = list(test)[:-4]

train = train.dropna(subset=FEATURE_NAME)
test = test.dropna(subset=FEATURE_NAME)

print(
    f"{len(train):,d} train samples, "
    f"{len(test):,d} test samples, "
    f"{test['label'].mean():.2%} positive rate."
)

# Set model file, create models folder if does not exist.
model_folder = Path("./data/models")
if not model_folder.exists():
    model_folder.mkdir()
model_file = model_folder / model_filename


if model_file.exists():
    model = xgb.XGBClassifier()
    model.load_model(model_file)
else:
    number_fold = len(train.cv_split.unique())
    data_train_cv = [train[train.cv_split != f'cv_{i}'] for i in range(number_fold)]
    data_test_cv = [train[train.cv_split == f'cv_{i}'] for i in range(number_fold)]
    # create an optuna study
    sampler = optuna.samplers.TPESampler(seed=rng_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective_xgboost(trial, data_train_cv, data_test_cv, FEATURE_NAME),
        n_trials=100,
        show_progress_bar=True,
    )

    # get the best hyperparameters
    best_params = study.best_params
    model = xgb.XGBClassifier(**best_params)

    # refit the model with best parameters
    model.fit(train[FEATURE_NAME], train.label, verbose=1)

    # save the model
    model.save_model(model_file)
