"""
This script performs 11-fold cross-validated optimization of an XGBoost classifier using Optuna on the training set from `feature_extraction_pipeline.py`.  
This classifier is referred to as SuperModelXGBoost in `src/hp_pred/supermodel.py`.
"""

from pathlib import Path
import optuna
import pandas as pd
from hp_pred.supermodel import SuperModelXGBoost


if __name__ == "__main__":
    dataset_name = '30_s_dataset_reg_signal'
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Parameter to choose optimization mode
    OPTIMIZE_HYPERPARAMS = False    # Do not change it
    
    model_filename = f"super_xgb_11_folds_test_{'optimized' if OPTIMIZE_HYPERPARAMS else 'default'}_params.pkl"
    
    # Load data
    test = pd.read_parquet("data/features_extraction_integrated_2windows/test.parquet")
    train = pd.read_parquet("data/features_extraction_integrated_2windows/train.parquet")
    FEATURE_NAME = list(test)[:-4]

    print(f"{len(train):,d} train samples, {len(test):,d} test samples, {test['label'].mean():.2%} positive rate.")
    
    # Create and train the super model
    model_folder = Path("./data/models")
    model_folder.mkdir(exist_ok=True)
    model_file = model_folder / model_filename
    
    super_model = SuperModelXGBoost(n_folds=11, threshold=6, random_state=42, optimized=OPTIMIZE_HYPERPARAMS)
    
    if model_file.exists():
        print("Loading existing model...")
        super_model.load(model_file)
    else:
        if OPTIMIZE_HYPERPARAMS:
            print("Training super model with per-fold optimization...")
            super_model.fit(train, FEATURE_NAME, n_trials=50)
        else:
            print("Training super model with default parameters...")
            super_model.fit(train, FEATURE_NAME)
        super_model.save(model_file)