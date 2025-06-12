"""
This script generates 10 new train/test splits and optimizes an XGBoost Classifier on each training set.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb
import optuna
from hp_pred.experiments import objective_xgboost
import hp_pred.split as spt
from multiprocessing import Pool, cpu_count
from functools import partial

# --- Parameters ---
TRAIN_RATIO = 0.7  
N_FOLDS = 3
N_DATASETS = 10  
RNG_SEED = 23 
MODEL_FOLDER = Path("./data/models_xgb_sets")  
DATA_FOLDER = Path("./data/split_datasets_xgb_sets")  
DATA_PATH = "data/features_extraction_full_data/data.parquet"
TOL_SEGMENT = 0.01  
TOL_LABEL = 0.005  
MAX_ITER = 200_000

# --- Load the data ---
data = pd.read_parquet(DATA_PATH)
FEATURE_NAME = list(data.columns[:-8])
data = data.drop(columns='cv_split')

def shuffle_caseids_and_sort_within(df, seed=23):
    """
    Shuffle the case IDs randomly and sort the data within each case by time.
    """
    np.random.seed(seed)
    shuffled_caseids = np.random.permutation(df['caseid'].unique())
    df['caseid'] = pd.Categorical(df['caseid'], categories=shuffled_caseids, ordered=True)
    df = df.sort_values(by=['caseid', 'time']).reset_index(drop=True)
    return df

def process_single_dataset(i, data, feature_name, train_ratio, n_folds, rng_seed, 
                          tol_segment, tol_label, max_iter, model_folder, data_folder):
    """
    Process a single dataset: shuffle, split, perform CV, optimize model with Optuna,
    train the model and save both the dataset and model.
    """
    print(f"\n--- Generating dataset {i + 1} ---")

    # Shuffle and sort
    df = shuffle_caseids_and_sort_within(data.copy(), seed=rng_seed + i)

    # Compute label summary per case ID
    caseid_summary = df.groupby("caseid").agg(label=("label", "max")).reset_index()

    # Stratified split by case ID
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=rng_seed + i)
    train_idx, test_idx = next(splitter.split(caseid_summary["caseid"], caseid_summary["label"]))
    train_caseids = caseid_summary.iloc[train_idx]["caseid"]
    test_caseids = caseid_summary.iloc[test_idx]["caseid"]

    # Separate training and test data
    train_data = df[df["caseid"].isin(train_caseids)].reset_index(drop=True)
    test_data = df[df["caseid"].isin(test_caseids)].reset_index(drop=True)
    test_data["cv_split"] = "test"

    label_stats = train_data.groupby("caseid").agg(
        segment_count=("label", "count"),
        label_count=("label", "sum")
    )

    # Create balanced CV splits
    cv_label_stats = spt.create_cv_balanced_split(
        label_stats=label_stats,
        general_ratio_segment=spt.compute_ratio_segment(label_stats, label_stats),
        n_cv_splits=n_folds,
        tolerance_segment_split=tol_segment,
        tolerance_label_split=tol_label,
        n_max_iter_split=max_iter
    )

    # Assign CV splits to training data
    caseid_to_fold = {}
    for fold_id, fold_stats in enumerate(cv_label_stats):
        for cid in fold_stats.index:
            caseid_to_fold[cid] = f"cv_{fold_id}"
    train_data["cv_split"] = train_data["caseid"].map(caseid_to_fold)

    # Save full dataset
    full_dataset = pd.concat([train_data, test_data], ignore_index=True)
    dataset_path = data_folder / f"data_split_{i}.parquet"
    full_dataset.to_parquet(dataset_path, index=False)

    print(f"Dataset {i} saved to: {dataset_path}")

    # Create CV folds for Optuna optimization
    data_train_cv = [train_data[train_data.cv_split != f'cv_{j}'] for j in range(n_folds)]
    data_test_cv = [train_data[train_data.cv_split == f'cv_{j}'] for j in range(n_folds)]

    # Run Optuna optimization to find best model hyperparameters
    print(f"Optimizing XGBoost model for dataset {i}...")
    sampler = optuna.samplers.TPESampler(seed=rng_seed + i)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Start the hyperparameter optimization
    study.optimize(
        lambda trial: objective_xgboost(trial, data_train_cv, data_test_cv, feature_name),
        n_trials=100,
        show_progress_bar=True,
    )

    # Train final model with best parameters
    best_params = study.best_params
    best_params['n_jobs'] = 1  # Avoid nested parallelization
    model = xgb.XGBClassifier(**best_params)
    model.fit(train_data[feature_name], train_data.label, verbose=0)

    # Save the trained model
    model_path = model_folder / f"xgboost_model_{i}.json"
    model.save_model(model_path)
    print(f"Model {i} saved to: {model_path}")
    
    return f"Dataset {i} completed"

def main():
    # --- Create necessary folders ---
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    # Partial function with fixed parameters
    process_func = partial(
        process_single_dataset,
        data=data,
        feature_name=FEATURE_NAME,
        train_ratio=TRAIN_RATIO,
        n_folds=N_FOLDS,
        rng_seed=RNG_SEED,
        tol_segment=TOL_SEGMENT,
        tol_label=TOL_LABEL,
        max_iter=MAX_ITER,
        model_folder=MODEL_FOLDER,
        data_folder=DATA_FOLDER
    )

    # Determine number of processes (not more than number of datasets)
    n_processes = min(cpu_count(), N_DATASETS)
    print(f"Using {n_processes} processes")

    # Parallel dataset processing
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_func, range(N_DATASETS))

    print("\n--- All datasets have been processed ---")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
