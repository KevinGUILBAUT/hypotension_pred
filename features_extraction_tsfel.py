from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
import warnings
from tsfel import get_features_by_domain, time_series_features_extractor

warnings.filterwarnings('ignore')

def process_row(args):
    idx, row_data, signal_types, feature_names, cfg = args
    valid_row = True
    sample_features_df_list = []
    
    for n, signal in enumerate(signal_types):
        try:
            # Extract the time series for this signal
            current_signal_serie = pd.Series([row_data[f"{signal}_{t}"] for t in range(20)])

            # If there are too many NaNs, skip this row
            if current_signal_serie.isna().sum() > 18:
                raise ValueError("Too many NaNs")
            
            # Feature extraction using TSFEL
            features_df = time_series_features_extractor(cfg, current_signal_serie, verbose=0)
            features_df = features_df.add_prefix(f'{signal}_')
            sample_features_df_list.append(features_df)
            
        except Exception as e:
            valid_row = False
            break
    
    if valid_row and sample_features_df_list:
        sample_features_df = pd.concat(sample_features_df_list, axis=1)
        return valid_row, sample_features_df, idx
    else:
        return False, None, idx


def compute_tsfel_features_parallel(data, 
                                   output_filename=None,
                                   signal_types=['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2',
                                                'etco2', 'mac', 'pp_ct', 'rf_ct', 'body_temp'],
                                   n_jobs=None):
    
    feature_names = []
    for signal in signal_types:
        for t in range(20):
            feature_names.append(f"{signal}_{t}")
    
    data = data.copy()
    data.reset_index(drop=True, inplace=True)
    
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    print(f"Utilisation de {n_jobs} processeurs parallèles")
    
    cfg = get_features_by_domain()
    
    args_list = [(idx, data.iloc[idx], signal_types, feature_names, cfg) 
                 for idx in range(len(data))]
    
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(process_row, args_list), 
            total=len(args_list),
            desc="TSFEL Feature Extraction"
        ))
    
    valid_results = [(features_df, idx) for valid_row, features_df, idx in results if valid_row]
    
    if not valid_results:
        return pd.DataFrame()
    
    features_df_list, row_indices = zip(*valid_results)
    
    all_features_df = pd.concat(features_df_list, axis=0).reset_index(drop=True)
    
    # Now add the other columns using the stored indices to match rows
    all_features_df["last_map_value"] = data.loc[list(row_indices), "last_map_value"].values
    all_features_df["label"] = data.loc[list(row_indices), "label"].values
    all_features_df["label_id"] = data.loc[list(row_indices), "label_id"].values
    all_features_df["cv_split"] = data.loc[list(row_indices), "cv_split"].values
    
    if output_filename:
        output_dir = os.path.join('data', 'features_extraction_tsfel')
        os.makedirs(output_dir, exist_ok=True)
        
        full_csv_path = os.path.join(output_dir, output_filename)
        full_parquet_path = full_csv_path.replace('.csv', '.parquet')
        
        all_features_df.to_csv(full_csv_path, index=False)
        all_features_df.to_parquet(full_parquet_path)
    
    return all_features_df


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

    test_features_tsfel = compute_tsfel_features_parallel(
        test, 
        output_filename='test.csv'
    )
    
    train_features_tsfel = compute_tsfel_features_parallel(
        train, 
        output_filename='train.csv'
    )
    
    print(test_features_tsfel.head())