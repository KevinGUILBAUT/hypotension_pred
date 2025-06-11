
from pathlib import Path
import optuna
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
from sklearn.model_selection import KFold
import os
from hp_pred.experiments import objective_xgboost
import hp_pred.split as spt

class SuperModelXGBoost:
    """
    Super model composed of 11 XGBoost models trained on different folds.
    The final model predicts 1 if at least 6 out of 11 models predict 1.
    Currently, optimize_hyperparameters_for_fold is not functional.
    """
    
    def __init__(self, n_folds=11, threshold=6, random_state=42, optimized=False):
        self.n_folds = n_folds
        self.threshold = threshold
        self.random_state = random_state
        self.optimized = optimized
        self.models = []
        self.fold_indices = []
        self.best_params_per_fold = []
        
        # Predefined parameters for non-optimized models
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'min_child_weight': 1,
            'gamma': 0,
            'nthread': os.cpu_count(),
            'random_state': self.random_state,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
    def create_folds(self, data):
        label_stats = (
            data.groupby("caseid")
            .agg(segment_count=("label", "count"), label_count=("label", "sum"))
        )

        # Create balanced folds
        cv_label_stats = spt.create_cv_balanced_split(
            label_stats=label_stats,
            general_ratio_segment=spt.compute_ratio_segment(label_stats, label_stats),
            n_cv_splits=self.n_folds,
            tolerance_segment_split=0.01,
            tolerance_label_split=0.005,
            n_max_iter_split=500_000
        )

        # Associate each split to its data
        fold_data = []
        self.fold_indices = []

        for fold_stats in cv_label_stats:
            case_ids = fold_stats.index
            fold = data[data["caseid"].isin(case_ids)].copy()
            fold_data.append({'train': fold})
            self.fold_indices.append(case_ids)

        return fold_data
    
    def get_default_params_for_fold(self, fold_num):
        """Returns predefined parameters for a specific fold."""
        params = self.default_params.copy()
        params['random_state'] = self.random_state + fold_num
        return params
    
    def optimize_hyperparameters_for_fold(self, fold_train_data, feature_names, 
                                          fold_num, n_trials=50):
        """Optimizes hyperparameters for a specific fold."""
        print(f"Optimizing hyperparameters for fold {fold_num+1}/{self.n_folds}...")
        
        # Create internal validation on the training fold
        kf = KFold(n_splits=3, shuffle=False, random_state=self.random_state + fold_num)
        train_idx, val_idx = next(kf.split(fold_train_data))
        
        data_train_opt = fold_train_data.iloc[train_idx]
        data_val_opt = fold_train_data.iloc[val_idx]

        sampler = optuna.samplers.TPESampler(seed=self.random_state + fold_num)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
                lambda trial: objective_xgboost(
                    trial, data_train_opt, data_val_opt, feature_names
                ),
                n_trials=n_trials,
                show_progress_bar=True,
            )
        
        best_params = study.best_params
        best_params.update({
            "nthread": os.cpu_count(),
            "random_state": self.random_state + fold_num,
        })
        
        print(f"Fold {fold_num+1} - Score: {study.best_value:.4f}")
        return best_params
    
    def fit(self, train_data, feature_names, n_trials=50):
        """
        Trains the model on 11 folds with or without individual optimization.
        """
        if self.optimized:
            print(f"Training the super model with {self.n_folds} folds and hyperparameter optimization...")
            print("Each fold will have its own optimized hyperparameters.")
        else:
            print(f"Training the super model with {self.n_folds} folds and default parameters...")
            print("All folds will use the same default parameters.")
        
        # Create folds
        fold_data = self.create_folds(train_data)
        
        # Train each model
        self.models = []
        self.best_params_per_fold = []
        
        for i, fold in enumerate(fold_data):
            print(f"\n--- Fold {i+1}/{self.n_folds} ---")
            
            if self.optimized:
                # Optimize hyperparameters for this specific fold
                fold_params = self.optimize_hyperparameters_for_fold(
                    fold['train'], feature_names, i, n_trials
                )
            else:
                # Use predefined parameters
                fold_params = self.get_default_params_for_fold(i)
                print(f"Using default parameters for fold {i+1}")
            
            self.best_params_per_fold.append(fold_params)
            
            # Train the model with parameters (optimized or default)
            print(f"Training model {i+1}...")
            model = xgb.XGBClassifier(**fold_params)
            
            X_train = fold['train'][feature_names]
            y_train = fold['train']['label']
            
            model.fit(X_train, y_train, verbose=0)
            self.models.append(model)
        
        print("\nTraining complete!")
        mode_text = "with individually optimized hyperparameters" if self.optimized else "with default parameters"
        print(f"Total of {len(self.models)} models trained {mode_text}.")
    
    def predict_proba(self, X_test):
        if not self.models:
            raise ValueError("The model must be trained before making predictions")
        
        X_test = X_test
        n_samples = len(X_test)
        
        # Store predictions of each model
        individual_preds = np.zeros((n_samples, self.n_folds))
        
        for i, model in enumerate(self.models):
            # Predictions for each model
            prob_preds = model.predict_proba(X_test)[:, 1]
            individual_preds[:, i] = prob_preds

        # Descending sort of predictions for each sample
        sorted_preds = np.sort(individual_preds, axis=1)[:, ::-1]
        final_proba = sorted_preds[:, self.threshold - 1]
            
        proba_array = np.column_stack([1 - final_proba, final_proba])

        return proba_array
    
    def predict(self, test_data, feature_names):
        """Makes binary predictions."""
        X_test = test_data[feature_names]
        n_samples = len(X_test)

        # Store predictions of each model
        individual_preds = np.zeros((n_samples, self.n_folds))

        for i, model in enumerate(self.models):
            # Predictions for each model
            fold_preds = model.predict(X_test)
            individual_preds[:, i] = fold_preds

        # Majority vote: at least 6 models must predict 1
        votes = np.sum(individual_preds, axis=1)
        final_predictions = (votes >= self.threshold).astype(int)

        return final_predictions
    
    def get_fold_parameters(self):
        """Returns optimized parameters for each fold."""
        if not self.best_params_per_fold:
            return None
        
        params_summary = {}
        for i, params in enumerate(self.best_params_per_fold):
            params_summary[f'fold_{i+1}'] = params
        
        return params_summary
    
    def save(self, filepath):
        """Saves the super model."""
        model_data = {
            'models': self.models,
            'best_params_per_fold': self.best_params_per_fold,
            'n_folds': self.n_folds,
            'threshold': self.threshold,
            'random_state': self.random_state,
            'optimized': self.optimized,
            'fold_indices': self.fold_indices,
            'default_params': self.default_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        mode_text = "optimized" if self.optimized else "with default parameters"
        print(f"Model {mode_text} saved to {filepath}")
    
    def load(self, filepath):
        """Loads a saved super model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.best_params_per_fold = model_data.get('best_params_per_fold', [])
        self.n_folds = model_data['n_folds']
        self.threshold = model_data['threshold']
        self.random_state = model_data['random_state']
        self.optimized = model_data.get('optimized', True)
        self.fold_indices = model_data.get('fold_indices', [])
        self.default_params = model_data.get('default_params', self.default_params)
        
        mode_text = "optimized" if self.optimized else "with default parameters"
        print(f"Model {mode_text} loaded from {filepath}")
        print(f"Loaded {len(self.models)} models")
