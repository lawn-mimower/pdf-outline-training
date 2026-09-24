import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import joblib
import argparse

# --- STEP 1: LOAD AND PREPARE THE DATA ---
# This expects the CSV written by AIH_preprocessor/stage3.ipynb:
# columns feature_0 ... feature_11 (the 12 span features) and 'label'.
parser = argparse.ArgumentParser(description="Train the final LightGBM heading classifier.")
parser.add_argument("--data", default="final_training_dataset.csv", help="CSV from stage3.ipynb")
parser.add_argument("--model-out", default="lgbm_document_model_final.joblib", help="Where to save the model")
args = parser.parse_args()
DATA_PATH = args.data
MODEL_SAVE_PATH = args.model_out

print(f"Loading data from '{DATA_PATH}'...")
df = pd.read_csv(DATA_PATH)

# The extractor produces exactly 12 features per span, so train on those.
feature_columns = [f"feature_{i}" for i in range(12)]
missing = [c for c in feature_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns {missing}. All rows must have exactly 12 features.")

X = df[feature_columns].to_numpy(dtype=float)
y = df['label']

print("Data loaded and features extracted successfully.")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")


# --- STEP 2: HANDLE MISSING VALUES AND CLASS IMBALANCE ---

# Impute missing values (e.g., NaN) that might exist in the feature data
print("Applying imputer to handle any potential missing values...")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Use SMOTE to handle class imbalance
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

print("Imputation and SMOTE applied.")
print(f"Shape after resampling: {X_resampled.shape}")


# --- STEP 3: TRAIN THE LIGHTGBM MODEL ON THE ENTIRE DATASET ---
print("Training the LightGBM model on the full, pre-processed dataset...")

# Best parameters can be found using GridSearchCV, but these are sensible defaults
lgbm = LGBMClassifier(
    objective='multiclass',
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=31
)

lgbm.fit(X_resampled, y_resampled)

print("✅ Model training complete.")


# --- STEP 4: SAVE THE FINAL MODEL ---
print(f"Saving the final trained model to '{MODEL_SAVE_PATH}'...")
joblib.dump(lgbm, MODEL_SAVE_PATH)
print(f"🎉 Success! Model saved and ready for use in '{MODEL_SAVE_PATH}'.")

# Optional: You can uncomment the following lines to split data and see a
# classification report on a hold-out set for evaluation purposes.
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
# lgbm.fit(X_train, y_train)
# y_pred = lgbm.predict(X_test)
# print("\n--- Classification Report (on a test split) ---")
# print(classification_report(y_test, y_pred))

