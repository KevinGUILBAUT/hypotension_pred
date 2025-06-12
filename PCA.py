"""
This script applies an ACP on a specific dataset.
Currently, it's applied to the TSFEL-extracted features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def load_data(data_path):
    print(f"Loading data from: {data_path}")
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

def remove_high_nan_columns(df, non_feature_cols, threshold=0.75):
    # Select only feature columns by excluding non-feature columns
    feature_cols = [col for col in train_data.columns if col not in non_feature_cols]

    # Calculate percentage of NaNs in each feature column
    na_percent = df[feature_cols].isna().mean()
    # Identify columns exceeding the threshold
    columns_to_drop = na_percent[na_percent > threshold].index.tolist()
    print(f"Dropping {len(columns_to_drop)} columns with more than {threshold*100}% NaNs")
    
    return df.drop(columns=columns_to_drop)

def apply_pca(train_data, test_data, target_variance=0.8, non_feature_cols=['last_map_value', 'label', 'label_id', 'cv_split']):
    # Identify feature columns by excluding non-feature columns
    feature_cols = [col for col in train_data.columns if col not in non_feature_cols]
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[feature_cols])
    X_test = scaler.transform(test_data[feature_cols])
    
    # Apply PCA keeping enough components to explain target_variance (e.g., 80%)
    pca = PCA(n_components=target_variance, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Create DataFrames from PCA-transformed data
    pca_cols = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
    train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols)
    test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols)
    
    # Add back non-feature columns (labels, ids, splits, etc.)
    for col in non_feature_cols:
        if col in train_data.columns:
            train_pca_df[col] = train_data[col].reset_index(drop=True)
        if col in test_data.columns:
            test_pca_df[col] = test_data[col].reset_index(drop=True)
    
    # Print PCA summary info
    print(f"Number of components retained: {pca.n_components_}")
    print(f"Cumulative explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    return train_pca_df, test_pca_df, pca, scaler

def plot_variance_explained(pca):
    plt.figure(figsize=(10, 6))
    
    # Plot explained variance for each principal component
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance by Component')
    
    # Plot cumulative explained variance with threshold line
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum)+1), cumsum, marker='o')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join('data', 'features_extraction_tsfel', 'pca_results')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'variance_explained.png'))
    plt.close()

def save_results(train_pca_df, test_pca_df, train_data, test_data, pca, scaler):
    output_dir = os.path.join('data', 'features_extraction_tsfel', 'pca_results')
    os.makedirs(output_dir, exist_ok=True)

    # Save transformed train and test data as CSV and parquet
    train_pca_df.to_csv(os.path.join(output_dir, 'train_pca.csv'), index=False)
    train_pca_df.to_parquet(os.path.join(output_dir, 'train_pca.parquet'))
    
    test_pca_df.to_csv(os.path.join(output_dir, 'test_pca.csv'), index=False)
    test_pca_df.to_parquet(os.path.join(output_dir, 'test_pca.parquet'))
    
    # Save PCA components matrix
    pca_components_df = pd.DataFrame(
        pca.components_,
        columns=feature_cols,
        index=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    pca_components_df.to_csv(os.path.join(output_dir, 'pca_components.csv'))
    
    # Save explained variance ratios and cumulative variance ratios
    variance_df = pd.DataFrame({
        'variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_)
    })
    variance_df.index = [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))]
    variance_df.to_csv(os.path.join(output_dir, 'variance_explained.csv'))
    
    print(f"Results saved to folder: {output_dir}")

if __name__ == "__main__":
    input_dir = os.path.join('data', 'features_extraction_tsfel')
    train_path = os.path.join(input_dir, 'train.csv')
    test_path = os.path.join(input_dir, 'test.csv')
    
    # Load the datasets
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    # TEST: Uncomment to use a smaller subset for quick tests
    # train_data = train_data.head(500)
    # test_data = test_data.head(500)

    print(f"Initial shapes - Train: {train_data.shape}, Test: {test_data.shape}")
    
    # Columns to exclude from feature processing (e.g. labels, ids, split info)
    non_feature_cols = ['last_map_value', 'label', 'label_id', 'cv_split']
    
    # Remove columns with too many missing values
    train_data = remove_high_nan_columns(train_data, non_feature_cols, threshold=0.75)
    test_data = remove_high_nan_columns(test_data, non_feature_cols, threshold=0.75)
    
    print(f"Shapes after dropping high-NaN columns - Train: {train_data.shape}, Test: {test_data.shape}")
    
    # Fill remaining NaNs with column means (important to avoid data loss)
    feature_cols = [col for col in train_data.columns if col not in non_feature_cols]

    # For tsfel features, dropping rows with NaNs loses too much data,
    # so filling NaNs with mean is preferred
    train_data[feature_cols] = train_data[feature_cols].fillna(train_data[feature_cols].mean())
    test_data[feature_cols] = test_data[feature_cols].fillna(train_data[feature_cols].mean())  

    # Apply PCA to reduce dimensionality while preserving target variance
    train_pca_df, test_pca_df, pca, scaler = apply_pca(train_data, test_data, target_variance=0.8, non_feature_cols=non_feature_cols)
    
    print(f"Shapes after PCA - Train: {train_pca_df.shape}, Test: {test_pca_df.shape}")
    
    # Plot explained variance graphs
    plot_variance_explained(pca)
    
    # Save all results including transformed data and PCA metadata
    save_results(train_pca_df, test_pca_df, train_data, test_data, pca, scaler)
