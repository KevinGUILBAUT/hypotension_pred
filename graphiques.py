"""
This script generates the 'feature_extraction_pipeline.py' figure for a specific patient.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import Holt
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def plot_signal_with_predictions_and_regression(row_data, signal="mbp"):
    # Observed data (first 20 timesteps)
    observed = np.array([row_data[f"{signal}_{t}"] for t in range(20)])

    # Holt predictions (next 20 timesteps)
    predicted = np.array([row_data.get(f"{signal}_holt_t+{t+1}", np.nan) for t in range(20)])

    # X values for plotting
    x_observed = np.arange(20)
    x_pred = np.arange(20, 40)

    # Linear regression line for first half of predictions
    slope_first = row_data.get(f"{signal}_linreg_slope_first", np.nan)
    intercept_first = row_data.get(f"{signal}_linreg_intercept_first", np.nan)
    reg_line_first = slope_first * np.arange(10) + intercept_first if not np.isnan(slope_first) else [np.nan]*10

    # Linear regression line for second half of predictions
    slope_second = row_data.get(f"{signal}_linreg_slope_second", np.nan)
    intercept_second = row_data.get(f"{signal}_linreg_intercept_second", np.nan)
    reg_line_second = slope_second * np.arange(10, 20) + intercept_second if not np.isnan(slope_second) else [np.nan]*10

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_observed, observed, label="Observed", color="blue", marker="o")
    plt.plot(x_pred, predicted, label="Predicted (Holt)", color="orange", marker="x")
    plt.plot(x_pred[:10], reg_line_first, label="First Linear Regression", color="green", linestyle="--")
    plt.plot(x_pred[10:], reg_line_second, label="Second Linear Regression", color="red", linestyle="--")

    plt.axvline(19.5, color='gray', linestyle=':', label="Observed / Predicted Split")
    plt.title(f"Signal {signal.upper()}: Observed vs. Predicted + Linear Regression")
    plt.xlabel("Timestep (1 step = 30s)")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def process_raw_regression_row(row_data, window_sizes, signal_types, n_timesteps):

    valid_row = True
    row_features = {}
    
    for window in window_sizes:
        half_time = max(window, 2)  # Ensure minimum window size
        X = np.arange(-half_time, 0).reshape(-1, 1)  # Time indices (negative for past)
        
        for signal in signal_types:
            # Extract the last 'half_time' values for this signal
            y = np.array([row_data[f"{signal}_{t}"] for t in range(n_timesteps)])[-half_time:]
            
            # Handle NaN values differently for different signal types
            if signal not in ["mac", "pp_ct"]:
                # For most signals, exclude NaN values from fitting
                valid_mask = ~np.isnan(y)
                X_fit = X[valid_mask]
                y_fit = y[valid_mask]
            else:
                # For MAC and pp_ct, use all values (assuming no NaNs)
                X_fit = X
                y_fit = y
            
            # Fit linear regression if enough valid points
            if len(y_fit) < 2:
                slope = np.nan
                intercept = np.nan
                std_err = np.nan
                valid_row = False
                mse = np.nan
            else:
                model = LinearRegression()
                model.fit(X_fit, y_fit)
                y_pred = model.predict(X_fit)
                residuals = y_fit - y_pred
                slope = model.coef_[0]
                intercept = model.intercept_
                std_err = residuals.std()
                mse = np.mean((y_fit - y_pred) ** 2)

            # Store computed features
            row_features[f"{signal}_raw_mse_{window}"] = mse
            row_features[f"{signal}_raw_slope_{window}"] = slope
            row_features[f"{signal}_raw_intercept_{window}"] = intercept
            row_features[f"{signal}_raw_std_{window}"] = std_err
    
    return valid_row, row_features


def process_holt_row(row_data, signal_types, n_timesteps, alpha, beta, damped):

    valid_row = True
    row_features = {}

    for signal in signal_types:
        try:
            # Extract time series data for this signal
            y = np.array([row_data[f"{signal}_{t}"] for t in range(n_timesteps)])
            
            # Check for too many missing values
            if np.isnan(y).sum() > 18:
                raise ValueError("Too many NaNs in signal data")
            
            # Fit Holt model and generate forecasts
            model = Holt(y, damped=damped)
            model_fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=True)
            preds = model_fit.forecast(steps=20)  # Forecast next 20 timesteps
            
            # Store predictions
            for t in range(20):
                row_features[f"{signal}_holt_t+{t+1}"] = preds[t]
                
        except Exception as e:
            # If forecasting fails, fill with NaN
            for t in range(20):
                row_features[f"{signal}_holt_t+{t+1}"] = np.nan
            valid_row = False

    return valid_row, row_features


def process_linreg_holt_row(row_data, signal_types, forecast_horizon):
    """
    Fit linear regression models to Holt forecast data.
    
    This function takes the 20 Holt forecast values and fits separate linear
    regression models to the first 10 and last 10 predictions. This helps
    capture different trend patterns in early vs. late forecast periods.
    
    Parameters:
    - row_data: Data containing Holt forecast values
    - signal_types: Signals to process
    - forecast_horizon: Number of forecast timesteps (20)
    
    Returns:
    - valid_row: Boolean indicating successful regression fitting
    - row_features: Dictionary of regression coefficients and statistics
    """
    valid_row = True
    row_features = {}

    for signal in signal_types:
        try:
            # Extract all forecast values for this signal
            all_forecast_values = np.array([row_data[f"{signal}_holt_t+{t+1}"] for t in range(forecast_horizon)])
            
            # Split forecasts into first and second halves
            first_half = all_forecast_values[:forecast_horizon//2]  # First 10 predictions
            mask_first = ~np.isnan(first_half)
            n_valid_first = np.sum(mask_first)
            
            second_half = all_forecast_values[forecast_horizon//2:]  # Last 10 predictions
            mask_second = ~np.isnan(second_half)
            n_valid_second = np.sum(mask_second)
            
            # Process first half regression
            if n_valid_first >= 2:
                if np.isnan(first_half).any():
                    # Handle missing values by using only valid indices
                    valid_indices = np.where(mask_first)[0]
                    X_first = valid_indices.reshape(-1, 1)
                    y_first = first_half[valid_indices]
                else:
                    # Use all values if no missing data
                    X_first = np.arange(forecast_horizon//2).reshape(-1, 1)
                    y_first = first_half
                
                # Fit linear regression model
                model_first = LinearRegression()
                model_first.fit(X_first, y_first)
                
                # Store regression coefficients
                row_features[f"{signal}_linreg_slope_first"] = model_first.coef_[0]
                row_features[f"{signal}_linreg_intercept_first"] = model_first.intercept_
                
                # Calculate and store performance metrics
                y_pred_first = model_first.predict(X_first)
                mse_first = np.mean((y_first - y_pred_first) ** 2)
                row_features[f"{signal}_linreg_mse_first"] = mse_first
                
                std_err_first = np.std(y_first - y_pred_first) if len(y_first) > 1 else np.nan
                row_features[f"{signal}_linreg_std_first"] = std_err_first
                
                # Generate prediction at midpoint of first half
                full_X_first = np.arange(forecast_horizon//2).reshape(-1, 1)
                full_pred_first = model_first.predict(full_X_first)
                row_features[f"{signal}_linreg_pred_mid_first"] = full_pred_first[forecast_horizon//4]
            else:
                # Not enough valid data points for regression
                row_features[f"{signal}_linreg_slope_first"] = np.nan
                row_features[f"{signal}_linreg_intercept_first"] = np.nan
                row_features[f"{signal}_linreg_mse_first"] = np.nan
                row_features[f"{signal}_linreg_std_first"] = np.nan
                row_features[f"{signal}_linreg_pred_mid_first"] = np.nan
                valid_row = False
            
            # Process second half regression (similar logic)
            if n_valid_second >= 2: 
                if np.isnan(second_half).any():
                    valid_indices = np.where(mask_second)[0]
                    X_second = (valid_indices + forecast_horizon//2).reshape(-1, 1)
                    y_second = second_half[valid_indices]
                else:
                    X_second = (np.arange(forecast_horizon//2) + forecast_horizon//2).reshape(-1, 1)
                    y_second = second_half
                
                model_second = LinearRegression()
                model_second.fit(X_second, y_second)
                
                row_features[f"{signal}_linreg_slope_second"] = model_second.coef_[0]
                row_features[f"{signal}_linreg_intercept_second"] = model_second.intercept_
                
                y_pred_second = model_second.predict(X_second)
                mse_second = np.mean((y_second - y_pred_second) ** 2)
                row_features[f"{signal}_linreg_mse_second"] = mse_second
                
                std_err_second = np.std(y_second - y_pred_second) if len(y_second) > 1 else np.nan
                row_features[f"{signal}_linreg_std_second"] = std_err_second
                
                full_X_second = (np.arange(forecast_horizon//2) + forecast_horizon//2).reshape(-1, 1)
                full_pred_second = model_second.predict(full_X_second)
                mid_second_idx = forecast_horizon//4
                row_features[f"{signal}_linreg_pred_mid_second"] = full_pred_second[mid_second_idx]
            else:
                row_features[f"{signal}_linreg_slope_second"] = np.nan
                row_features[f"{signal}_linreg_intercept_second"] = np.nan
                row_features[f"{signal}_linreg_mse_second"] = np.nan
                row_features[f"{signal}_linreg_std_second"] = np.nan
                row_features[f"{signal}_linreg_pred_mid_second"] = np.nan
                valid_row = False
            
        except Exception as e:
            # Fill all features with NaN if processing fails
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

    return valid_row, row_features


def compute_integrated_features(data, 
                                 window_sizes=[2, 6, 20],
                                 signal_types=['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2',
                                               'etco2', 'mac', 'pp_ct', 'rf_ct', 'body_temp'],
                                 alpha=0.8,
                                 beta=0.2,
                                 damped=True):
    n_timesteps = 20
    forecast_horizon = 20
    
    # Ensure single row input
    if len(data) != 1:
        raise ValueError("This function is designed to process a single row only.")
    
    row_data = data.iloc[0]
    
    # Step 1: Raw regression features for specific signal types
    signal_types_raw_regression = ['mac', 'pp_ct', 'rf_ct']
    print("Step 1: Computing raw regression features...")
    valid_raw_row, raw_features = process_raw_regression_row(row_data, window_sizes, signal_types_raw_regression, n_timesteps)
    
    if not valid_raw_row:
        print("No valid raw regression features found for this row.")
        return pd.DataFrame()
    
    # Step 2: Holt forecasting for main physiological signals
    signal_types_holt_linreg = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2',
                               'etco2', 'body_temp']
    print("Step 2: Computing Holt forecasts...")
    valid_holt_row, holt_features = process_holt_row(row_data, signal_types_holt_linreg, n_timesteps, alpha, beta, damped)
    
    if not valid_holt_row:
        print("No valid Holt forecasts found for this row. Returning raw features only.")
        features_df = pd.DataFrame([raw_features], dtype="Float32")
        features_df["last_map_value"] = row_data["last_map_value"]
        for col in ["label", "label_id", "cv_split"]:
            if col in data.columns:
                features_df[col] = row_data[col]
        return features_df

    # Merge Holt features for linear regression processing
    temp_row_data_for_linreg = {**row_data.to_dict(), **holt_features}

    # Step 3: Linear regression on Holt forecasts
    print("Step 3: Computing linear regression on Holt forecasts...")
    valid_linreg_row, linreg_features = process_linreg_holt_row(temp_row_data_for_linreg, signal_types_holt_linreg, forecast_horizon)
    
    # Step 4: Merge all feature sets
    print("Step 4: Merging all feature sets...")
    merged_features_dict = {}
    
    # Always add raw features if valid
    if valid_raw_row:
        merged_features_dict.update(raw_features)
    
    # Add Holt predictions if valid
    if valid_holt_row:
        merged_features_dict.update(holt_features)
    
    # Add linear regression on Holt forecasts if valid
    if valid_linreg_row:
        merged_features_dict.update(linreg_features)
    
    if not merged_features_dict:
        print("No features could be computed for this row.")
        return pd.DataFrame()

    merged_features = pd.DataFrame([merged_features_dict], dtype="Float32")
    
    # Add important columns from the original row
    merged_features["last_map_value"] = row_data["last_map_value"]
    
    # Preserve metadata columns if they exist
    for col in ["label", "label_id", "cv_split"]:
        if col in data.columns:
            merged_features[col] = row_data[col]
    
    print("Feature computation complete for the single row.")
    
    return merged_features


if __name__ == "__main__":
    # Main execution block for testing and demonstration
    dataset_name = '30_s_dataset_reg_signal'

    # Load the dataset
    data = pd.read_parquet(f'./data/datasets/{dataset_name}/cases/', engine='pyarrow')
    static = pd.read_parquet(f'./data/datasets/{dataset_name}/meta.parquet', engine='pyarrow')

    # Merge data with metadata
    data = data.merge(static, on='caseid')
    data = data[data['intervention_hypo']==0] 
    data["last_map_value"] = data["mbp_19"]

    # Select a specific patient and observation for analysis
    patient_ids = data['caseid'].unique()
    case_id_index = 63
    selected_caseid = patient_ids[case_id_index]
    
    df_selected_case = data[data['caseid'] == selected_caseid].reset_index(drop=True)
    selected_row_index_in_case = 50

    print(f"\nAnalyzing patient with caseid: {selected_caseid}")
    print(f"Number of observations for this patient: {len(df_selected_case)}")
    print(f"Processing observation at index {selected_row_index_in_case} for this patient.")

    # Process the selected observation
    df_to_process = pd.DataFrame([df_selected_case.iloc[selected_row_index_in_case]])
    
    # Compute integrated features
    single_patient_features = compute_integrated_features(df_to_process)
    
    print(f"\nFeatures computed for the selected patient and index:")
    if not single_patient_features.empty:
        print(f"Feature shape: {single_patient_features.shape}")
        print(single_patient_features.T)
        
        # Create visualization
        row_for_plot = {**df_to_process.iloc[0].to_dict(), **single_patient_features.iloc[0].to_dict()}
        plot_signal_with_predictions_and_regression(row_for_plot, signal="sbp")
    else:
        print("No features could be computed for this observation.")