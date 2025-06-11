from pathlib import Path
import multiprocessing as mp
import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import auc, roc_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from aeon.classification.sklearn import RotationForestClassifier
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from hp_pred.databuilder import DataBuilder
from optuna import Trial
from tqdm import tqdm
from scipy.stats import shapiro


NUMBER_CV_FOLD = 3
N_INTERPOLATION = 1000

def objective_super_model_xgboost(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for XGBoost model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
    Returns:
        float: Mean AUC score of the XGBoost model.

    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.01, 1.0, log=True
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "eval_metric": trial.suggest_categorical("eval_metric", ["auc", "aucpr", "logloss", "map"]),
        "objective": "binary:logistic",
        "nthread": os.cpu_count(),
        # "scale_pos_weight": data.label.value_counts()[0] / data.label.value_counts()[1],
    }

    X_train = data_train[feature_name]
    y_train = data_train.label

    X_validate = data_test[feature_name]
    y_validate = data_test.label

    optuna_model = XGBClassifier(**params)
    optuna_model.fit(X_train, y_train)
    # Make predictions
    y_pred = optuna_model.predict_proba(X_validate)[:, 1]

    # Evaluate predictions with AP score
    score = average_precision_score(y_validate, y_pred)

    return score

def objective_tabnet(
    trial: Trial,
    data_train: List[pd.DataFrame],
    data_test: List[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean Average Precision score for TabNet model using Optuna hyperparameter optimization.
    
    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (List[pd.DataFrame]): List of training data sets.
        data_test (List[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
        feature_name (List[str]): List of feature names to use for training.
    Returns:
        float: Mean Average Precision score of the TabNet model.
    """
    params = {
        "n_d": trial.suggest_int("n_d", 8, 64),
        "n_a": trial.suggest_int("n_a", 8, 64),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "max_epochs": trial.suggest_int("max_epochs", 50, 200),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
    }
    
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    # Number of epchs without improvement of the metric
    patience = 10
    
    fold_number = 0
    ap_scores = np.zeros(number_cv_fold)
    
    for i in range(number_cv_fold):
        X_train = data_train[i][feature_name]
        y_train = data_train[i].label
        
        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_validate_scaled = scaler.transform(X_validate)
        
        optuna_model = TabNetClassifier(
            n_d=params["n_d"],
            n_a=params["n_a"],
            n_steps=params["n_steps"],
            gamma=params["gamma"],
            lambda_sparse=params["lambda_sparse"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=params["learning_rate"]),
            mask_type="entmax",
            scheduler_params=dict(
                mode="min",
                patience=patience//2,
                min_lr=1e-5,
                factor=0.5
            ),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            verbose=0
        )
        
        optuna_model.fit(
            X_train=X_train_scaled,
            y_train=y_train,
            eval_set=[(X_validate_scaled, y_validate)],
            max_epochs=params["max_epochs"],
            patience=patience,
            batch_size=params["batch_size"],
            eval_metric=['auc']
        )
        
        y_pred = optuna_model.predict_proba(X_validate_scaled)[:, 1]
        
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1
    
    return ap_scores.mean()


def objective_rotationforest(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for RotationForest model using Optuna hyperparameter optimization.
    
    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
        feature_name (List[str]): List of feature names to use for training.
    Returns:
        float: Mean AUC score of the RotationForest model.
    
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_group": trial.suggest_int("min_group", 3, 3),  # Fixed value of 3 to avoid errors
        "max_group": trial.suggest_int("max_group", 3, 3),  # Fixed value of 3 to avoid errors
        "remove_proportion": trial.suggest_float("remove_proportion", 0.0, 0.9),
        "pca_solver": trial.suggest_categorical("pca_solver", ["auto", "full", "arpack", "randomized"]),
        "n_jobs": os.cpu_count(),
    }
    
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."
    
    fold_number = 0
    # separate training in folds
    ap_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):
        
        X_train = data_train[i][feature_name]
        y_train = data_train[i].label
        
        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label
        
        optuna_model = RotationForestClassifier(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]
        
        # Evaluate predictions with AP score
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1
    
    return ap_scores.mean()



def objective_randomforest(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for RandomForest model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
        feature_name (List[str]): List of feature names to use for training.
    Returns:
        float: Mean AUC score of the RandomForest model.

    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "n_jobs": os.cpu_count(),
    }
    
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    fold_number = 0
    # separate training in folds
    ap_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):

        X_train = data_train[i][feature_name]
        y_train = data_train[i].label

        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label

        optuna_model = RandomForestClassifier(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions with AP score
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1

    return ap_scores.mean()

def objective_mlp(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for MLP model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
        feature_name (List[str]): List of feature names to use for training.
    Returns:
        float: Mean AUC score of the MLP model.

    """

    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layers = []
    for i in range(n_layers):
        hidden_layers.append(trial.suggest_int(f"n_units_l{i}", 10, 200))
    
    params = {
        "hidden_layer_sizes": tuple(hidden_layers),
        "activation": trial.suggest_categorical("activation", ["tanh", "relu", "logistic"]),
        "solver": trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),  # L2 regularization
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive", "invscaling"]),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "early_stopping": trial.suggest_categorical("early_stopping", [True, False]),
        "batch_size": trial.suggest_categorical("batch_size", ["auto", 32, 64, 128, 256]),
    }
    
    # Add momentum only when solver is 'sgd'
    if params["solver"] == "sgd":
        params["momentum"] = trial.suggest_float("momentum", 0.0, 0.9)
    
    # Validation fraction only makes sense with early stopping
    if params["early_stopping"]:
        params["validation_fraction"] = trial.suggest_float("validation_fraction", 0.1, 0.3)
    
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    fold_number = 0
    # separate training in folds
    ap_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):

        X_train = data_train[i][feature_name]
        y_train = data_train[i].label

        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label

        optuna_model = MLPClassifier(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions with AP score
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1

    return ap_scores.mean()

def objective_logistic_regression(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for Logistic Regression model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
        feature_name (List[str]): List of feature names to use for training.
    Returns:
        float: Mean AUC score of the Logistic Regression model.

    """
    params = {
        "C": trial.suggest_float("C", 1e-5, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None]),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "random_state": 42,
        "n_jobs": os.cpu_count(),
    }
    
    # Ensure compatible solver and penalty combinations
    if params["penalty"] == "elasticnet" and params["solver"] != "saga":
        params["solver"] = "saga"
    elif params["penalty"] == "l1" and params["solver"] not in ["liblinear", "saga"]:
        params["solver"] = "saga"
    elif params["penalty"] is None and params["solver"] == "liblinear":
        params["solver"] = "lbfgs"
    
    # Add l1_ratio only when elasticnet is selected
    if params["penalty"] == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    fold_number = 0
    # separate training in folds
    ap_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):

        X_train = data_train[i][feature_name]
        y_train = data_train[i].label

        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label

        optuna_model = LogisticRegression(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions with AP score
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1

    return ap_scores.mean()


def objective_xgboost(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for XGBoost model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
    Returns:
        float: Mean AUC score of the XGBoost model.

    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.01, 1.0, log=True
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "eval_metric": trial.suggest_categorical("eval_metric", ["auc", "aucpr", "logloss", "map"]),
        "objective": "binary:logistic",
        "nthread": os.cpu_count(),
        # "scale_pos_weight": data.label.value_counts()[0] / data.label.value_counts()[1],
    }
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    fold_number = 0
    # separate training in 3 folds
    ap_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):

        X_train = data_train[i][feature_name]
        y_train = data_train[i].label

        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label

        optuna_model = XGBClassifier(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions with AP score
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1

    return ap_scores.mean()


def precision_event_recall(y_true: np.ndarray, y_pred: np.ndarray, label_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision, recall and thresholds for precision-recall curve.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted probabilities.
        label_id (np.ndarray): Label IDs.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following arrays:
            - precision (np.ndarray): Precision values.
            - recall (np.ndarray): Recall values.
            - thresholds (np.ndarray): Thresholds for precision.
    """
    desc_score_indices = np.argsort(y_pred, kind="mergesort")[::-1]
    y_pred = y_pred[desc_score_indices]
    y_true = y_true[desc_score_indices]
    label_id = label_id[desc_score_indices]

    # Find distinct value indices
    distinct_value_indices = np.where(np.diff(y_pred))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Compute true positive counts at each threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = np.cumsum(1 - y_true)[threshold_idxs]
    ps = tps + fps
    # Compute precision at each threshold
    precision = np.divide(tps, ps, where=(ps != 0))

    # Reduce the number of thresholds by removing those that do not change precision
    # Compute changes in precision
    precision_changes = np.abs(np.diff(precision))
    # Extend changes array to match threshold_idxs size
    precision_changes = np.r_[precision_changes, precision_changes[-1]]

    # Define the number of thresholds to select
    num_selected_thresholds = 1000

    if len(precision_changes) > num_selected_thresholds:

        # Normalize changes to use as probabilities
        weights = precision_changes / np.sum(precision_changes)

        # Randomly select indices based on weights, ensuring some spread
        selected_threshold_idxs = np.random.choice(
            threshold_idxs, size=num_selected_thresholds, replace=False, p=weights)

        # Ensure the first and last thresholds are included
        selected_threshold_idxs = np.unique(np.r_[selected_threshold_idxs, threshold_idxs[0], threshold_idxs[-1]])

    else:
        selected_threshold_idxs = threshold_idxs

    # Recompute precision at selected thresholds
    precision_label = np.cumsum(y_true)[selected_threshold_idxs] / \
        np.cumsum(np.ones_like(y_true))[selected_threshold_idxs]

    # Compute recall at selected thresholds without explicit for loop
    nb_unique_label = len(np.unique(label_id[np.where(y_true == 1)[0]]))

    # Converting to a single-line computation using list comprehension
    recall_label = np.fromiter((len(np.unique(label_id[np.where(y_true[:threshold_idx + 1] == 1)[0]]))
                               for threshold_idx in selected_threshold_idxs), dtype=float) / nb_unique_label

    precision_label = np.hstack([1, precision_label])
    recall_label = np.hstack([0, recall_label])

    return precision_label, recall_label, y_pred[selected_threshold_idxs]


def get_all_stats(
    y_true: np.ndarray, y_pred: np.ndarray, label_id: np.ndarray, strategy: str = "max_precision", target: float = None
) -> tuple[
    float, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Calculate various statistics for binary classification.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted probabilities.
        label_id (np.ndarray): Label IDs.

    Returns:
        tuple: A tuple containing the following statistics:
            - auc_ (float): Area under the ROC curve.
            - sensitivity (float): Sensitivity.
            - specificity (float): Specificity.
            - ppv (float): Positive predictive value.
            - npv (float): Negative predictive value.
            - fpr (np.ndarray): False positive rates.
            - tpr (np.ndarray): True positive rates.
            - thr (np.ndarray): Thresholds.
            - precision (np.ndarray): Precision values.
            - recall (np.ndarray): Recall values.
            - thr_pr (np.ndarray): Thresholds for precision.
            - ap (float): Average precision score.
            - auprc (float): Area under the precision-recall curve.
            - precision_threshold (float): Precision threshold.
            - recall_threshold (float): Recall threshold.
    """
    precision, recall, thr_pr = precision_event_recall(y_true, y_pred, label_id)
    ap = average_precision_score(y_true, y_pred)

    fpr, tpr, thr = roc_curve(y_true, y_pred)
    auc_ = float(auc(fpr, tpr))
    auprc = auc(recall, precision)

    if strategy == 'max_precision':
        # find the threshold that optimize the precision after a recall of 0.1
        max_precision = np.max(precision[recall > 0.02])
        id_thresh_opt_prc = int(np.argmin(np.abs(precision - max_precision)))
    elif strategy == 'targeted_precision':
        id_thresh_opt_prc = int(np.argmin(np.abs(precision - target)))
    elif strategy == 'targeted_recall':
        id_thresh_opt_prc = int(np.argmin(np.abs(recall - target)))
    elif strategy == 'fixed_threshold':
        id_thresh_opt_prc = int(np.argmin(np.abs(thr_pr - target)))

    threshold_opt = thr_pr[id_thresh_opt_prc]

    precision_threshold = precision[id_thresh_opt_prc]
    recall_threshold = recall[id_thresh_opt_prc]

    nb_unique_label = len(np.unique(label_id[np.where(y_true == 1)[0]]))
    prevalence = nb_unique_label / (nb_unique_label + np.sum(1-y_true))

    term1 = prevalence / recall_threshold
    term2 = (1 - precision_threshold) / precision_threshold

    # Calculate specificity
    specificity = (1 - prevalence) / (term1 * term2 + (1 - prevalence))

    npv = (specificity * (1 - prevalence)) / (specificity * (1 - prevalence) + (1 - recall_threshold) * prevalence)

    f1 = 2 * (precision_threshold * recall_threshold) / (precision_threshold + recall_threshold)

    return auc_, specificity, npv, fpr, tpr, thr, precision, recall, thr_pr, ap, auprc, precision_threshold, recall_threshold, f1, threshold_opt


def bootstrap_worker(args):
    y_true, y_pred, y_label_id, xs_interpolation, recall_levels, seed, len_y_pred, strategy, target = args
    rng = np.random.default_rng(seed)
    while True:
        indices = rng.integers(0, len_y_pred, len_y_pred)
        if len(np.unique(y_true[indices])) < 2:
            continue

        (
            auc,
            specificity,
            npv,
            fpr,
            tpr,
            thr,
            precision,
            recall,
            thr_pr,
            ap,
            auprc,
            precision_threshold,
            recall_threshold,
            f1,
            threshold_opt,
        ) = get_all_stats(y_true[indices], y_pred[indices], y_label_id[indices], strategy, target)

        tpr_interpolated = np.interp(xs_interpolation, fpr, tpr)
        thr_interpolated = np.interp(xs_interpolation, fpr, np.clip(thr, -100, 100))
        precision_interpolated = np.interp(recall_levels, recall, precision)
        precision_interpolated[0] = 1
        thr_pr_interpolated = np.interp(recall_levels, recall[1:], thr_pr)

        return (
            auc,
            specificity,
            npv,
            ap,
            tpr_interpolated,
            thr_interpolated,
            precision_interpolated,
            thr_pr_interpolated,
            auprc,
            precision_threshold,
            recall_threshold,
            f1,
            threshold_opt,
        )


def bootstrap_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_label_id: np.ndarray,
    n_bootstraps: int = 200,
    rng_seed: int = 42,
    strategy: str = "max_precision",
    target: float = None,
) -> tuple[Dict[str, pd.DataFrame], np.ndarray]:
    """
    Perform bootstrap testing for evaluating a regression model's performance.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_label_id (np.ndarray): Label IDs.
        n_bootstraps (int, optional): Number of bootstrap iterations. Defaults to 200.
        rng_seed (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing the DataFrame with evaluation metrics and the interpolated TPR values.
    """
    np.random.seed(rng_seed)
    seeds = np.random.randint(0, np.iinfo(np.int32).max, size=n_bootstraps)

    # for each subgoup of data, create a regressor and evaluate it
    xs_interpolation = np.linspace(0, 1, N_INTERPOLATION)
    recall_levels = np.linspace(0, 1, N_INTERPOLATION)

    results = []
    args = [(y_true, y_pred, y_label_id, xs_interpolation, recall_levels, seeds[i], len(y_pred), strategy, target)
            for i in range(n_bootstraps)]

    with mp.Pool() as pool:
        for result in tqdm(pool.imap(bootstrap_worker, args), total=n_bootstraps):
            results.append(result)

    # Initialize arrays to store the results
    aucs = np.zeros(n_bootstraps, dtype=float)
    tprs_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    thrs_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    precision_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    thrs_interpolated_precision = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)

    specificities = np.zeros(n_bootstraps, dtype=float)
    npvs = np.zeros(n_bootstraps, dtype=float)
    aps = np.zeros(n_bootstraps, dtype=float)
    auprcs = np.zeros(n_bootstraps, dtype=float)
    precision_thresholds = np.zeros(n_bootstraps, dtype=float)
    recall_thresholds = np.zeros(n_bootstraps, dtype=float)
    f1s = np.zeros(n_bootstraps, dtype=float)
    threshold_opts = np.zeros(n_bootstraps, dtype=float)

    for i, result in enumerate(results):
        (
            auc,
            specificity,
            npv,
            ap,
            tpr_interpolated,
            thr_interpolated,
            precision_interpolated_,
            thr_pr_interpolated,
            auprc,
            precision_threshold,
            recall_threshold,
            f1,
            threshold_opt,
        ) = result

        aucs[i] = auc
        specificities[i] = specificity
        npvs[i] = npv
        aps[i] = ap
        auprcs[i] = auprc
        precision_thresholds[i] = precision_threshold
        recall_thresholds[i] = recall_threshold
        f1s[i] = f1
        threshold_opts[i] = threshold_opt

        tprs_interpolated[i] = tpr_interpolated
        thrs_interpolated[i] = thr_interpolated
        precision_interpolated[i] = precision_interpolated_
        thrs_interpolated_precision[i] = thr_pr_interpolated

    dict = {
        "fprs": xs_interpolation,
        "tpr": tprs_interpolated,
        "threshold": thr_interpolated,
        "aucs": aucs,
        "specificity": specificities,
        "npvs": npvs,
        "recall": recall_levels,
        "precision": precision_interpolated,
        "thr_precision": thrs_interpolated_precision,
        "aps": aps,
        "auprcs": auprcs,
        "precision_threshold": precision_thresholds,
        "recall_threshold": recall_thresholds,
        "f1": f1s,
        "threshold_opt": threshold_opts,
    }
    return dict, precision_interpolated, thrs_interpolated_precision


def load_labelized_cases(
    dataset_path: Path,
    caseid: int,
):
    """
    Labelize the IOH event of a single case.
    Args:
        databuilder (Path): Path of the dataset.
        caseid (int): Case ID.
    Returns:
        pd.DataFrame: Labeled case data.
    """
    databuilder = DataBuilder.from_json(dataset_path)
    databuilder.raw_data_folder = databuilder.raw_data_folder / f"case-{caseid:04d}.parquet"
    case_data, _ = databuilder._import_raw()
    case_data = case_data.reset_index("caseid", drop=True)
    preprocess_case = databuilder._preprocess(case_data)
    label, _ = databuilder._labelize(preprocess_case)
    preprocess_case["label"] = label

    return preprocess_case


def print_one_stat(series: pd.Series, percent: bool = False) -> bool:
    """ Test if a series is normally distributed using the Shapiro-Wilk test.

    If it is return mean (sd), otherwise return median (Q1, Q3).
    I percent is True, the values are returned as percentages.

    Args:
        series (pd.Series): Series to test.

    Returns:
        bool: True if the series is normally distributed, False otherwise.
    """
    stat, p = shapiro(series)
    if p > 0.05:
        if percent:
            return f"{series.mean():.1%} ({series.std():.1%})"
        else:
            return f"{series.mean():.2f} ({series.std():.2f})"
    else:
        if percent:
            return f"{series.median():.1%} [{series.quantile(0.25):.1%}, {series.quantile(0.75):.1%}]"
        else:
            return f"{series.median():.2f} [{series.quantile(0.25):.2f}, {series.quantile(0.75):.2f}]"


def print_statistics(dict: Dict[str, pd.DataFrame]) -> None:
    """
    Print the evaluation statistics of the model.
    Args:
        df (pd.DataFrame): DataFrame containing the evaluation metrics.
    """
    df = pd.DataFrame()
    for key in ["aucs", "aps", "auprcs", "threshold_opt", "recall_threshold", "precision_threshold", "specificity", "npvs", "f1"]:
        df[key] = dict[key]
    print(f"AUC: {print_one_stat(df.aucs, False)}")
    print(f"AP: {print_one_stat(df.aps, False)}")
    print(f"AUPRC: {print_one_stat(df.auprcs, False)}")
    print(f"Threshold: {print_one_stat(df.threshold_opt, False)}")
    print(f"Recall: {print_one_stat(df.recall_threshold, True)}")
    print(f"Precision: {print_one_stat(df.precision_threshold, True)}")
    print(f"Specificity: {print_one_stat(df.specificity, True)}")
    print(f"NPV: {print_one_stat(df.npvs, True)}")
    print(f"F1-score: {print_one_stat(df.f1, True)}")
    return
