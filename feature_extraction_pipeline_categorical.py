"""
This script extracts features from physiological signals using a specific pipeline configuration.

It is very similar to `feature_extraction_pipeline.py`, but includes additional handling for categorical parameters and missing (NaN) values.

Pipeline steps:
    1. Clean NaN values in raw signals
    2. Raw regression features on drug-related signals
    3. Holt exponential smoothing forecasts on physiological signals
    4. Linear regression analysis on the Holt forecasts
    5. Merge all features into a single dataset
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

def clean_nan_raw_signals_row(args):
    """
    Clean NaN values in raw signal data for a single row.
    
    """
    idx, row_data, signal_types, n_timesteps = args
    cleaned_row = row_data.copy()
    valid_row = True

    for signal in signal_types:
        # Extract time series values for this signal
        values = np.array([row_data.get(f"{signal}_{t}", np.nan) for t in range(n_timesteps)])
        valid_mask = ~np.isnan(values)

        # If at least 2 non-NaN points exist, replace NaN values with mean
        # TODO: Consider using a better imputation method
        if valid_mask.sum() >= 2:
            mean_val = np.nanmean(values)
            filled_values = np.where(valid_mask, values, mean_val)
        else:
            # Mark row as invalid if signal has insufficient valid data
            valid_row = False
            break  # Stop immediately if a signal is invalid

        # Update cleaned row with filled values
        for t in range(n_timesteps):
            cleaned_row[f"{signal}_{t}"] = filled_values[t]

    return valid_row, cleaned_row, idx

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
        X = np.arange(-half_time, 0).reshape(-1, 1)  # Time indices (negative, ending at 0)
        
        for signal in signal_types:
            # Extract the last 'half_time' values for this signal
            y = np.array([row_data[f"{signal}_{t}"] for t in range(n_timesteps)])[-half_time:]
            
            # For non-drug signals, handle NaN values by filtering
            if signal not in ["mac", "pp_ct"]:
                valid_mask = ~np.isnan(y)
                X_fit = X[valid_mask]
                y_fit = y[valid_mask]
            else:
                # For drug signals, use all values (assuming they're already clean)
                X_fit = X
                y_fit = y
            
            # Check if we have enough points for regression
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
                
                # Extract regression parameters
                slope = model.coef_[0]
                intercept = model.intercept_
                std_err = residuals.std()
                
                # Calculate Mean Squared Error
                mse = np.mean((y_fit - y_pred) ** 2)

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
            # Extract time series for this signal
            y = np.array([row_data[f"{signal}_{t}"] for t in range(n_timesteps)])
            
            # Fit Holt exponential smoothing model
            model = Holt(y, damped=damped)
            model_fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=True)
            
            # Generate forecasts for the next 20 time steps
            preds = model_fit.forecast(steps=20)
            
            # Store all predicted values
            for t in range(20):
                row_features[f"{signal}_holt_t+{t+1}"] = preds[t]
                
        except Exception as e:
            # If Holt fitting fails, fill with NaN values
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
            # Extract Holt forecast values for this signal
            all_forecast_values = np.array([row_data[f"{signal}_holt_t+{t+1}"] for t in range(forecast_horizon)])
            
            # Split forecasts into two halves
            first_half = all_forecast_values[:forecast_horizon//2]   # First 10 steps
            second_half = all_forecast_values[forecast_horizon//2:]  # Last 10 steps
            
            # Create masks for valid (non-NaN) values
            mask_first = ~np.isnan(first_half)
            n_valid_first = np.sum(mask_first)
            mask_second = ~np.isnan(second_half)
            n_valid_second = np.sum(mask_second)
            
            # Process first half (beginning of forecast)
            if n_valid_first >= 2:  # Need at least 2 points for linear regression
                if np.isnan(first_half).any():
                    # Handle sparse data by using only valid indices
                    valid_indices = np.where(mask_first)[0]
                    X_first = valid_indices.reshape(-1, 1)
                    y_first = first_half[valid_indices]
                else:
                    # Use all data points
                    X_first = np.arange(forecast_horizon//2).reshape(-1, 1)
                    y_first = first_half
                
                # Fit linear regression for first half
                model_first = LinearRegression()
                model_first.fit(X_first, y_first)
                
                # Extract regression coefficients
                row_features[f"{signal}_linreg_slope_first"] = model_first.coef_[0]
                row_features[f"{signal}_linreg_intercept_first"] = model_first.intercept_
                
                # Calculate predictions and error metrics
                y_pred_first = model_first.predict(X_first)
                mse_first = np.mean((y_first - y_pred_first) ** 2)
                row_features[f"{signal}_linreg_mse_first"] = mse_first
                
                # Standard error of residuals
                std_err_first = np.std(y_first - y_pred_first) if len(y_first) > 1 else np.nan
                row_features[f"{signal}_linreg_std_first"] = std_err_first
                
                # Predict value at midpoint of first half
                full_X_first = np.arange(forecast_horizon//2).reshape(-1, 1)
                full_pred_first = model_first.predict(full_X_first)
                row_features[f"{signal}_linreg_pred_mid_first"] = full_pred_first[forecast_horizon//4]
            else:
                # Insufficient data for first half regression
                row_features[f"{signal}_linreg_slope_first"] = np.nan
                row_features[f"{signal}_linreg_intercept_first"] = np.nan
                row_features[f"{signal}_linreg_mse_first"] = np.nan
                row_features[f"{signal}_linreg_std_first"] = np.nan
                row_features[f"{signal}_linreg_pred_mid_first"] = np.nan
                valid_row = False
            
            # Process second half (end of forecast)
            if n_valid_second >= 2: 
                if np.isnan(second_half).any():
                    # Handle sparse data
                    valid_indices = np.where(mask_second)[0]
                    X_second = (valid_indices + forecast_horizon//2).reshape(-1, 1)  # Adjust indices
                    y_second = second_half[valid_indices]
                else:
                    # Use all data points with proper indexing
                    X_second = (np.arange(forecast_horizon//2) + forecast_horizon//2).reshape(-1, 1)
                    y_second = second_half
                
                # Fit linear regression for second half
                model_second = LinearRegression()
                model_second.fit(X_second, y_second)
                
                # Extract regression coefficients
                row_features[f"{signal}_linreg_slope_second"] = model_second.coef_[0]
                row_features[f"{signal}_linreg_intercept_second"] = model_second.intercept_
                
                # Calculate predictions and error metrics
                y_pred_second = model_second.predict(X_second)
                mse_second = np.mean((y_second - y_pred_second) ** 2)
                row_features[f"{signal}_linreg_mse_second"] = mse_second
                
                # Standard error of residuals
                std_err_second = np.std(y_second - y_pred_second) if len(y_second) > 1 else np.nan
                row_features[f"{signal}_linreg_std_second"] = std_err_second
                
                # Predict value at midpoint of second half
                full_X_second = (np.arange(forecast_horizon//2) + forecast_horizon//2).reshape(-1, 1)
                full_pred_second = model_second.predict(full_X_second)
                mid_second_idx = forecast_horizon//4  # Midpoint of second half
                row_features[f"{signal}_linreg_pred_mid_second"] = full_pred_second[mid_second_idx]
            else:
                # Insufficient data for second half regression
                row_features[f"{signal}_linreg_slope_second"] = np.nan
                row_features[f"{signal}_linreg_intercept_second"] = np.nan
                row_features[f"{signal}_linreg_mse_second"] = np.nan
                row_features[f"{signal}_linreg_std_second"] = np.nan
                row_features[f"{signal}_linreg_pred_mid_second"] = np.nan
                valid_row = False
            
        except Exception as e:
            # In case of error, fill with NaN for both halves
            for half in ['first', 'second']:
                row_features[f"{signal}_linreg_slope_{half}"] = np.nan
                row_features[f"{signal}_linreg_intercept_{half}"] = np.nan
                row_features[f"{signal}_linreg_mse_{half}"] = np.nan
                row_features[f"{signal}_linreg_std_{half}"] = np.nan
                row_features[f"{signal}_linreg_pred_mid_{half}"] = np.nan
            
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
    
    This function orchestrates a multi-step feature extraction pipeline:
    1. Clean NaN values in raw signals
    2. Raw regression features on drug-related signals
    3. Holt exponential smoothing forecasts on physiological signals
    4. Linear regression analysis on the Holt forecasts
    5. Merge all features into a single dataset
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
        os.makedirs('data/features_extraction_test_categoricals', exist_ok=True)

    # Step 0: Clean NaN values in raw signals
    print("Step 0: Cleaning NaN values in raw signals...")
    args_list = [(idx, row, signal_types, n_timesteps) for idx, row in data.iterrows()]

    with Pool() as pool:
        cleaned_results = pool.map(clean_nan_raw_signals_row, args_list)

    # Keep only valid rows after cleaning
    cleaned_results = [r for r in cleaned_results if r[0]]
    cleaned_rows = [r[1] for r in cleaned_results]

    data = pd.DataFrame(cleaned_rows)
    data.reset_index(drop=True, inplace=True)

    # Define drug-related signals for raw regression analysis
    signal_types_drugs = ['mac', 'pp_ct', 'rf_ct']
    
    # Step 1: Process raw regression features (for drug signals)
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
    signal_types = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'body_temp']
    
    # Step 2: Process Holt forecasting (for physiological signals)
    print("Step 2: Computing Holt forecasts...")
    holt_args_list = [(idx, data.iloc[idx], signal_types, n_timesteps, alpha, beta, damped) 
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
    
    # Create dataframe for Holt features
    holt_features_df = pd.DataFrame(holt_features_list, dtype="Float32")
    print(f"{len(valid_holt_results)} valid rows after Holt forecasting")

    # Step 3: Process linear regression on Holt forecasts
    print("Step 3: Computing linear regression on Holt forecasts...")
    
    # Find common indices between raw and Holt results
    common_indices = list(set(raw_indices).intersection(set(holt_indices)))
    if not common_indices:
        print("No common valid rows between raw regression and Holt forecasting.")
        return pd.DataFrame()
    
    # Prepare data for linear regression processing
    holt_data = holt_features_df.copy()
    holt_data["last_map_value"] = data.loc[list(holt_indices), "last_map_value"].values
    
    # Set up arguments for linear regression processing
    linreg_args_list = [(idx, holt_data.iloc[i], signal_types, forecast_horizon) 
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
        
    # Step 4: Merge all feature sets
    print("Step 4: Merging all feature sets...")
    
    # Find common valid rows across all three feature extraction methods
    common_indices = list(set(raw_indices).intersection(set(holt_indices)).intersection(set(linreg_indices)))
    
    if not common_indices:
        print("No common valid rows across all three feature extraction methods.")
        return pd.DataFrame()
    
    # Initialize final merged dataframe
    merged_features = pd.DataFrame()

    # Create feature dictionaries
    raw_features_dict = {idx: features for features, idx in valid_raw_results}
    # Note: Holt predictions are not included in final output, only used for linear regression
    linreg_features_dict = {idx: features for features, idx in valid_linreg_results}

    # Merge features for each valid row
    for idx in tqdm(common_indices, desc="Merging Features"):
        row = {
            **raw_features_dict[idx],      # Raw regression features
            **linreg_features_dict[idx]    # Linear regression on Holt features
        }
        merged_features = pd.concat([merged_features, pd.DataFrame([row])], ignore_index=True)
    
    # Add metadata columns from original dataset
    for col in ["age", "bmi", "asa", "label", "label_id", "cv_split", 'last_map_value']:
        if col in data.columns:
            merged_features[col] = data.loc[common_indices, col].values
    
    # Save results if output prefix is provided
    if output_prefix:
        output_dir = 'data/features_extraction_test_categoricals'
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
    
    # Compute features for test and train set
    test_features = compute_integrated_features(
        test, 
        output_prefix='test'
    )
    
    train_features = compute_integrated_features(
        train, 
        output_prefix='train'
    )

    ### EXTERNAL DATASET PROCESSING ###
    #data_chu = pd.read_parquet('./data/datasets/clean_chu_trends/data.parquet')

    # Compute features for external dataset
    #data_features = compute_integrated_features(
    #    data_chu, 
    #    output_prefix='data'
    #)