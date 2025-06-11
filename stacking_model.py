from pathlib import Path
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Filenames for pre-trained base models and stacking model
rf_model_filename = "RF_30s_reg_holt_med.pkl"
xgb_model_filename = "xgb_30_s_reg_holt_med.json"
rotf_model_filename = "rotation_classifier_opt_reg_holt_med.pkl"
stacking_model_filename = "STACK_30s_reg_holt_med_2.pkl"

# Load training and test datasets
test = pd.read_parquet("data/features_extraction_integrated_2windows/test.parquet")
train = pd.read_parquet("data/features_extraction_integrated_2windows/train.parquet")

# Ensure reproducibility
rng_seed = 42

FEATURE_NAME = list(test)[:-4]

print(
    f"{len(train):,d} training samples, "
    f"{len(test):,d} test samples, "
    f"{test['label'].mean():.2%} positive class rate."
)

# Set the model folder path and create it if it does not exist
model_folder = Path("./data/models")
if not model_folder.exists():
    model_folder.mkdir()

# Load pre-trained Random Forest model
with open(model_folder / rf_model_filename, 'rb') as file:
    rf_model = pickle.load(file)

# Load pre-trained XGBoost model
xgb_model = XGBClassifier()
xgb_model.load_model(model_folder / xgb_model_filename)

# Load pre-trained Rotation Forest model
with open(model_folder / rotf_model_filename, 'rb') as file:
    rotf_model = pickle.load(file)

# Create the stacking classifier using Logistic Regression as the final estimator
final_estimator = LogisticRegression(max_iter=1000)

print("Creating stacking classifier...")
stacking_model = StackingClassifier(
    estimators=[
        ('rf', rf_model),        # Random Forest
        ('xgb', xgb_model),      # XGBoost
        ('rotf', rotf_model)     # Rotation Forest
    ],
    final_estimator=final_estimator,
    cv=5,  # Number of folds for internal cross-validation
    stack_method='predict_proba',  # Use class probabilities from base models
    passthrough=False,  # Do not include original features in final estimator
    n_jobs=-1, 
    verbose=1  
)

print("Stacking model created. Fitting the model...")
stacking_model.fit(train[FEATURE_NAME], train.label)

# Save the trained stacking model
with open(model_folder / stacking_model_filename, 'wb') as file:
    pickle.dump(stacking_model, file)

print("Stacking model successfully saved.")
