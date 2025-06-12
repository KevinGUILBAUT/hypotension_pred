"""
This script extracts trend features from physiological signals by applying linear regression over 
multiple time-window scales (2, 6, and 20 time steps). It also includes categorical parameters.

"""

from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')


def process_row(args):
    """
    Process a single row for linear trend features extraction.
    """
    idx, row_data, window_sizes, signal_types, n_timesteps = args
    
    valid_row = True
    row_features = {}
    
    for window in window_sizes:
        half_time = max(window, 2)
        X = np.arange(-half_time, 0).reshape(-1, 1)
        
        for signal in signal_types:
            y = np.array([row_data[f"{signal}_{t}"] for t in range(n_timesteps)])[-half_time:]
            
            if signal not in ["mac", "pp_ct"]:
                valid_mask = ~np.isnan(y)
                X_fit = X[valid_mask]
                y_fit = y[valid_mask]
            else:
                X_fit = X
                y_fit = y
            
            if len(y_fit) < 2:
                slope = np.nan
                intercept = np.nan
                std_err = np.nan
                valid_row = False
            else:
                model = LinearRegression()
                model.fit(X_fit, y_fit)
                y_pred = model.predict(X_fit)
                residuals = y_fit - y_pred
                slope = model.coef_[0]
                intercept = model.intercept_
                std_err = residuals.std()
            
            row_features[f"{signal}_slope_{window}"] = slope
            row_features[f"{signal}_intercept_{window}"] = intercept
            row_features[f"{signal}_std_{window}"] = std_err
    
    return valid_row, row_features, idx

def compute_linear_trends_parallel(data, 
                                 window_sizes=[2, 6, 20],
                                 signal_types=['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2',
                                              'etco2', 'mac', 'pp_ct', 'rf_ct', 'body_temp'],
                                 output_filename=None,
                                 n_jobs=None):
    """
    Compute linear trend features in parallel for time series data.
    """
    n_timesteps = 20
    
    data = data.copy()
    data.reset_index(drop=True, inplace=True)
    
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    print(f"Using {n_jobs} parallel processors")
    
    args_list = [(idx, data.iloc[idx], window_sizes, signal_types, n_timesteps) 
                 for idx in range(len(data))]
    
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(process_row, args_list), 
            total=len(args_list),
            desc="Linear Trends Feature Extraction"
        ))
    
    valid_results = [(features_dict, idx) for valid_row, features_dict, idx in results if valid_row]
    
    if not valid_results:
        return pd.DataFrame()
    
    # Separate the features and indices
    features_list, row_indices = zip(*valid_results)
    
    # Initialize the features DataFrame
    features_dict = {f"{signal}_{stat}_{w}": []
                    for w in window_sizes
                    for signal in signal_types
                    for stat in ["slope", "intercept", "std"]}
    
    # features dictionary
    for row_features in features_list:
        for feature, value in row_features.items():
            features_dict[feature].append(value)
    
    # Create DataFrame
    features_df = pd.DataFrame(features_dict, dtype="Float32")
    
    # Add the additional columns
    features_df["age"] = data.loc[list(row_indices), "age"].values
    features_df["bmi"] = data.loc[list(row_indices), "bmi"].values
    features_df["asa"] = data.loc[list(row_indices), "asa"].values

    features_df["last_map_value"] = data.loc[list(row_indices), "last_map_value"].values
    features_df["label"] = data.loc[list(row_indices), "label"].values
    features_df["label_id"] = data.loc[list(row_indices), "label_id"].values
    features_df["cv_split"] = data.loc[list(row_indices), "cv_split"].values

    
    if output_filename:
        output_dir = 'data/features_extraction_trends'
        os.makedirs(output_dir, exist_ok=True)
        
        full_csv_path = os.path.join(output_dir, output_filename)
        full_parquet_path = full_csv_path.replace('.csv', '.parquet')
        
        features_df.to_csv(full_csv_path, index=False)
        features_df.to_parquet(full_parquet_path)
    
    return features_df


if __name__ == "__main__":
    # Configuration
    dataset_name = '30_s_dataset_reg_signal'

    # Load main dataset and metadata
    data = pd.read_parquet(f'./data/datasets/{dataset_name}/cases/', engine='pyarrow')
    static = pd.read_parquet(f'./data/datasets/{dataset_name}/meta.parquet', engine='pyarrow')

    # Merge data with metadata and apply filters
    data = data.merge(static, on='caseid')
    data = data[data['intervention_hypo'] == 0]
    data["last_map_value"] = data["mbp_19"]     # Set last MAP value as target

    # Split data into training and test sets
    train = data[data['split'] == 'train']
    test = data[data['split'] == 'test']
    
    # Apply additional filters to test set
    test = test[test['ioh_in_leading_time'] == 0]  # No IOH in leading time
    test = test[test['ioh_at_time_t'] == 0]        # No IOH at current time
    
    test_features = compute_linear_trends_parallel(
        test, 
        output_filename='test.csv'
    )
    
    train_features = compute_linear_trends_parallel(
        train, 
        output_filename='train.csv'
    )
    
    