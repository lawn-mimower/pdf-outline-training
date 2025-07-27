import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import joblib
import ast

# --- STEP 1: LOAD AND PREPARE THE DATA ---
# This assumes you have already run a script to generate this dataset.
# The 'features' column should contain a string representation of a list of 12 features.
DATA_PATH = './AIH_model/AIH_model/final_training_dataset.csv' 
MODEL_SAVE_PATH = 'lgbm_document_model_final.joblib'

print(f"Loading data from '{DATA_PATH}'...")
df = pd.read_csv(DATA_PATH)

# Convert the string representation of feature lists back into actual lists
# and then into separate columns for the model.
df['features'] = df['features'].apply(ast.literal_eval)

# Check for consistency: Ensure all feature lists have the same length (12)
# This is critical for creating a valid DataFrame.
if df['features'].apply(len).nunique() != 1 or df['features'].apply(len).iloc[0] != 12:
    raise ValueError("Inconsistent number of features found. All rows must have exactly 12 features.")

X = np.array(df['features'].tolist())
y = df['label_encoded']

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

