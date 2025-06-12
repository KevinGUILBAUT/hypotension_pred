"""
This script extracts features from physiological signals using a specific pipeline configuration.

Pipeline steps:
    1. Extract raw regression features from drug-related signals.
    2. Apply Holt's exponential smoothing to physiological signals for forecasting.
    3. Perform linear regression analysis on the Holt forecasts.
    4. Merge all extracted features into a single dataset.
"""

from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import Holt
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')

def process_raw_regression_row(args):
    """
    Process a single row for raw regression feature extraction.
    
    Computes linear regression statistics (slope, intercept, MSE, std error) 
    for different window sizes on time series signals.
    """
    idx, row_data, window_sizes, signal_types, n_timesteps = args
    
    valid_row = True
    row_features = {}
    
    # Process each window size
    for window in window_sizes:
        half_time = max(window, 2)  # Ensure minimum window size of 2
        X = np.arange(-half_time, 0).reshape(-1, 1)  # Time indices (negative for past values)
        
        # Process each signal type
        for signal in signal_types:
            # Extract time series values for current signal
            y = np.array([row_data[f"{signal}_{t}"] for t in range(n_timesteps)])[-half_time:]
            
            if signal not in ["mac", "pp_ct"]:
                valid_mask = ~np.isnan(y)
                X_fit = X[valid_mask]
                y_fit = y[valid_mask]
            else:
                # For drug-related signals, keep all values (including NaN)
                X_fit = X
                y_fit = y
            
            # Perform linear regression if sufficient data points
            if len(y_fit) < 2:
                # Insufficient data for regression
                slope = np.nan
                intercept = np.nan
                std_err = np.nan
                valid_row = False
                mse = np.nan
            else:
                # Fit linear regression model
                model = LinearRegression()
                model.fit(X_fit, y_fit)
                y_pred = model.predict(X_fit)
                residuals = y_fit - y_pred
                
                # Extract regression statistics
                slope = model.coef_[0]
                intercept = model.intercept_
                std_err = residuals.std()
                mse = np.mean((y_fit - y_pred) ** 2)  # Mean Squared Error

            # Store features with descriptive names
            row_features[f"{signal}_raw_mse_{window}"] = mse
            row_features[f"{signal}_raw_slope_{window}"] = slope
            row_features[f"{signal}_raw_intercept_{window}"] = intercept
            row_features[f"{signal}_raw_std_{window}"] = std_err
    
    return valid_row, row_features, idx


def process_holt_row(args):
    """
    Process a single row for Holt exponential smoothing forecasting.
    
    Applies Holt's exponential smoothing method to generate forecasts
    for the next 20 time steps for each signal.
    
    """
    idx, row_data, signal_types, n_timesteps, alpha, beta, damped = args
    valid_row = True
    row_features = {}

    for signal in signal_types:
        try:
            # Extract time series for current signal
            y = np.array([row_data[f"{signal}_{t}"] for t in range(n_timesteps)])
            
            # Apply Holt exponential smoothing model
            model = Holt(y, damped=damped)
            model_fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=True)
            preds = model_fit.forecast(steps=20)  # Forecast next 20 time steps
            
            # Store all predicted values
            for t in range(20):
                row_features[f"{signal}_holt_t+{t+1}"] = preds[t]
                
        except Exception as e:
            # Handle forecasting failures by setting NaN values
            for t in range(20):
                row_features[f"{signal}_holt_t+{t+1}"] = np.nan
            valid_row = False

    return valid_row, row_features, idx


def process_linreg_holt_row(args):
    """
    Process linear regression on Holt forecast predictions.
    
    Splits the 20-step Holt forecasts into two halves (first 10, last 10)
    and performs linear regression on each half to capture trend characteristics.
    """
    idx, row_data, signal_types, forecast_horizon = args
    valid_row = True
    row_features = {}

    for signal in signal_types:
        try:
            # Extract Holt forecast predictions for current signal
            all_forecast_values = np.array([row_data[f"{signal}_holt_t+{t+1}"] for t in range(forecast_horizon)])
            
            # Split forecasts into two halves
            first_half = all_forecast_values[:forecast_horizon//2]   # First 10 steps
            second_half = all_forecast_values[forecast_horizon//2:]  # Last 10 steps
            
            # Process first half
            mask_first = ~np.isnan(first_half)
            n_valid_first = np.sum(mask_first)
            
            # Process second half
            mask_second = ~np.isnan(second_half)
            n_valid_second = np.sum(mask_second)
            
            # Linear regression on first half (early forecast period)
            if n_valid_first >= 2:  # Need at least 2 points for linear regression
                if np.isnan(first_half).any():
                    # Handle missing values by using only valid indices
                    valid_indices = np.where(mask_first)[0]
                    X_first = valid_indices.reshape(-1, 1)
                    y_first = first_half[valid_indices]
                else:
                    # Use all points if no missing values
                    X_first = np.arange(forecast_horizon//2).reshape(-1, 1)
                    y_first = first_half
                
                # Fit linear regression model
                model_first = LinearRegression()
                model_first.fit(X_first, y_first)
                
                # Extract regression coefficients
                row_features[f"{signal}_linreg_slope_first"] = model_first.coef_[0]
                row_features[f"{signal}_linreg_intercept_first"] = model_first.intercept_
                
                # Calculate regression statistics
                y_pred_first = model_first.predict(X_first)
                mse_first = np.mean((y_first - y_pred_first) ** 2)
                row_features[f"{signal}_linreg_mse_first"] = mse_first
                
                # Standard error of residuals
                std_err_first = np.std(y_first - y_pred_first) if len(y_first) > 1 else np.nan
                row_features[f"{signal}_linreg_std_first"] = std_err_first
                
                # Predicted value at midpoint of first half
                full_X_first = np.arange(forecast_horizon//2).reshape(-1, 1)
                full_pred_first = model_first.predict(full_X_first)
                row_features[f"{signal}_linreg_pred_mid_first"] = full_pred_first[forecast_horizon//4]
            else:
                # Insufficient data - set all features to NaN
                row_features[f"{signal}_linreg_slope_first"] = np.nan
                row_features[f"{signal}_linreg_intercept_first"] = np.nan
                row_features[f"{signal}_linreg_mse_first"] = np.nan
                row_features[f"{signal}_linreg_std_first"] = np.nan
                row_features[f"{signal}_linreg_pred_mid_first"] = np.nan
                valid_row = False
            
            # Linear regression on second half (later forecast period)
            if n_valid_second >= 2: 
                if np.isnan(second_half).any():
                    # Handle missing values by using only valid indices
                    valid_indices = np.where(mask_second)[0]
                    X_second = (valid_indices + forecast_horizon//2).reshape(-1, 1)  # Adjust indices
                    y_second = second_half[valid_indices]
                else:
                    # Use all points if no missing values
                    X_second = (np.arange(forecast_horizon//2) + forecast_horizon//2).reshape(-1, 1)
                    y_second = second_half
                
                # Fit linear regression model
                model_second = LinearRegression()
                model_second.fit(X_second, y_second)
                
                # Extract regression coefficients
                row_features[f"{signal}_linreg_slope_second"] = model_second.coef_[0]
                row_features[f"{signal}_linreg_intercept_second"] = model_second.intercept_
                
                # Calculate regression statistics
                y_pred_second = model_second.predict(X_second)
                mse_second = np.mean((y_second - y_pred_second) ** 2)
                row_features[f"{signal}_linreg_mse_second"] = mse_second
                
                # Standard error of residuals
                std_err_second = np.std(y_second - y_pred_second) if len(y_second) > 1 else np.nan
                row_features[f"{signal}_linreg_std_second"] = std_err_second
                
                # Predicted value at midpoint of second half
                full_X_second = (np.arange(forecast_horizon//2) + forecast_horizon//2).reshape(-1, 1)
                full_pred_second = model_second.predict(full_X_second)
                mid_second_idx = forecast_horizon//4  # Midpoint of second half
                row_features[f"{signal}_linreg_pred_mid_second"] = full_pred_second[mid_second_idx]
            else:
                # Insufficient data - set all features to NaN
                row_features[f"{signal}_linreg_slope_second"] = np.nan
                row_features[f"{signal}_linreg_intercept_second"] = np.nan
                row_features[f"{signal}_linreg_mse_second"] = np.nan
                row_features[f"{signal}_linreg_std_second"] = np.nan
                row_features[f"{signal}_linreg_pred_mid_second"] = np.nan
                valid_row = False
            
        except Exception as e:
            # Handle any processing errors by setting all features to NaN
            row_features[f"{signal}_linreg_slope_first"] = np.nan
            row_features[f"{signal}_linreg_intercept_first"] = np.nan
            row_features[f"{signal}_linreg_mse_first"] = np.nan
            row_features[f"{signal}_linreg_std_first"] = np.nan
            row_features[f"{signal}_linreg_pred_mid_first"] = np.nan
            
            row_features[f"{signal}_linreg_slope_second"] = np.nan
            row_features[f"{signal}_linreg_intercept_second"] = np.nan
            row_features[f"{signal}_linreg_mse_second"] = np.nan
            row_features[f"{signal}_linreg_std_second"] = np.nan
            row_features[f"{signal}_linreg_pred_mid_second"] = np.nan
            
            valid_row = False

    return valid_row, row_features, idx


def compute_integrated_features(data, 
                             window_sizes=[2, 6, 20],
                             signal_types=['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2',
                                        'etco2', 'mac', 'pp_ct', 'rf_ct', 'body_temp'],
                             output_prefix=None,
                             n_jobs=None,
                             alpha=0.8,
                             beta=0.2,
                             damped=True):
    """
    Main function to compute integrated time series features using multiple methods.
    
    This function performs a three-step feature extraction pipeline:
    1. Raw regression features on drug-related signals
    2. Holt exponential smoothing forecasts on physiological signals
    3. Linear regression analysis on the Holt forecasts
    4. Merge all features into a single dataset

    """

    # Configuration parameters
    n_timesteps = 20        # Number of historical time steps
    forecast_horizon = 20   # Number of forecast steps
    
    # Prepare data
    data = data.copy()
    data.reset_index(drop=True, inplace=True)
    
    # Set up parallel processing
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    print(f"Using {n_jobs} parallel processors")
    
    # Create output directories if needed
    if output_prefix:
        os.makedirs('data/features_extraction_integrated', exist_ok=True)

    # Define signal types for different processing steps
    signal_types_drugs = ['mac', 'pp_ct', 'rf_ct']  # Drug-related signals
    
    # STEP 1: Process raw regression features on drug signals
    print("Step 1: Computing raw regression features...")
    raw_args_list = [(idx, data.iloc[idx], window_sizes, signal_types_drugs, n_timesteps) 
                     for idx in range(len(data))]
    
    with Pool(processes=n_jobs) as pool:
        raw_results = list(tqdm(
            pool.imap(process_raw_regression_row, raw_args_list), 
            total=len(raw_args_list),
            desc="Raw Regression Feature Extraction"
        ))
    
    # Filter valid results
    valid_raw_results = [(features_dict, idx) for valid_row, features_dict, idx in raw_results if valid_row]
    raw_features_list, raw_indices = zip(*valid_raw_results) if valid_raw_results else ([], [])
    
    if not valid_raw_results:
        print("No valid rows found for raw regression.")
        return pd.DataFrame()

    print(f"{len(valid_raw_results)} valid rows after raw regression")
    
    # Define physiological signals for Holt forecasting
    signal_types_physio = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'body_temp']
    
    # STEP 2: Process Holt exponential smoothing forecasts
    print("Step 2: Computing Holt forecasts...")
    holt_args_list = [(idx, data.iloc[idx], signal_types_physio, n_timesteps, alpha, beta, damped) 
                      for idx in range(len(data))]
    
    with Pool(processes=n_jobs) as pool:
        holt_results = list(tqdm(
            pool.imap(process_holt_row, holt_args_list), 
            total=len(holt_args_list),
            desc="Holt Feature Extraction"
        ))

    # Filter valid results
    valid_holt_results = [(features_dict, idx) for valid_row, features_dict, idx in holt_results if valid_row]
    holt_features_list, holt_indices = zip(*valid_holt_results) if valid_holt_results else ([], [])
    
    if not valid_holt_results:
        print("No valid rows found for Holt forecasting.")
        return pd.DataFrame()
    
    # Create DataFrame for Holt features
    holt_features_df = pd.DataFrame(holt_features_list, dtype="Float32")
    print(f"{len(valid_holt_results)} valid rows after Holt forecasting")

    # STEP 3: Process linear regression on Holt forecasts
    print("Step 3: Computing linear regression on Holt forecasts...")
    
    # Check for common valid indices between raw and Holt processing
    common_indices = list(set(raw_indices).intersection(set(holt_indices)))
    if not common_indices:
        print("No common valid rows between raw regression and Holt forecasting.")
        return pd.DataFrame()
    
    # Prepare data for linear regression processing
    holt_data = holt_features_df.copy()
    holt_data["last_map_value"] = data.loc[list(holt_indices), "last_map_value"].values
    
    # Prepare arguments for linear regression processing
    linreg_args_list = [(idx, holt_data.iloc[i], signal_types_physio, forecast_horizon) 
                       for i, idx in enumerate(holt_indices)]
    
    with Pool(processes=n_jobs) as pool:
        linreg_results = list(tqdm(
            pool.imap(process_linreg_holt_row, linreg_args_list), 
            total=len(linreg_args_list),
            desc="Linear Regression on Holt Feature Extraction"
        ))
    
    # Filter valid results
    valid_linreg_results = [(features_dict, idx) for valid_row, features_dict, idx in linreg_results if valid_row]
    linreg_features_list, linreg_indices = zip(*valid_linreg_results) if valid_linreg_results else ([], [])
    
    if not valid_linreg_results:
        print("No valid rows found for linear regression on Holt forecasts.")
        return pd.DataFrame()

    print(f"{len(valid_linreg_results)} valid rows after linear regression")
        
    # STEP 4: Merge all feature sets
    print("Step 4: Merging all feature sets...")
    
    # Find common valid rows across all three feature extraction methods
    common_indices = list(set(raw_indices).intersection(set(holt_indices)).intersection(set(linreg_indices)))
    
    if not common_indices:
        print("No common valid rows across all three feature extraction methods.")
        return pd.DataFrame()
    
    # Create final merged DataFrame
    merged_features = pd.DataFrame()

    # Create dictionaries for feature lookup
    raw_features_dict = {idx: features for features, idx in valid_raw_results}
    linreg_features_dict = {idx: features for features, idx in valid_linreg_results}
    # Note: We don't include Holt predictions directly, only the derived linear regression features

    # Merge features for each common valid row
    for idx in tqdm(common_indices, desc="Merging Features"):
        row = {
            **raw_features_dict[idx],      # Raw regression features
            **linreg_features_dict[idx]    # Linear regression on Holt features
        }
        merged_features = pd.concat([merged_features, pd.DataFrame([row])], ignore_index=True)
    
    # Add additional columns from original data
    for col in ["label", "label_id", "cv_split", 'last_map_value']:
        if col in data.columns:
            merged_features[col] = data.loc[common_indices, col].values
    
    # Save results if output prefix is provided
    if output_prefix:
        output_dir = 'data/features_extraction_integrated'
        csv_path = os.path.join(output_dir, f"{output_prefix}.csv")
        parquet_path = os.path.join(output_dir, f"{output_prefix}.parquet")
        
        merged_features.to_csv(csv_path, index=False)
        merged_features.to_parquet(parquet_path)
        
        print(f"Saved integrated features to {csv_path} and {parquet_path}")
    
    return merged_features


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
    
    # Compute features for different datasets
    test_features = compute_integrated_features(test, output_prefix='test')
    train_features = compute_integrated_features(train, output_prefix='train')

    # Process external CHU dataset
    #data_chu = pd.read_parquet('./data/datasets/clean_chu_trends/data.parquet')
    #data_features = compute_integrated_features(data_chu, output_prefix='data')