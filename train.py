#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# 1. Configuration & Parameters
# ==========================================
DATA_PATH = 'data/aug_train.csv'
OUTPUT_FILE = 'model.bin'

# Random Forest Hyperparameters (based on tuning results)
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 50,
    'min_samples_split': 5,
    'min_samples_leaf': 4,
    'random_state': 7,
    'n_jobs': -1
}


# ==========================================
# 2. Data Cleaning Functions
# ==========================================
def clean_dataframe(df):
    """
    Performs initial data cleaning and type conversion
    before passing data into the Scikit-Learn pipeline.
    """
    df = df.copy()

    # 1. Handle Target (if present) and ID
    if 'target' in df.columns:
        df['target'] = df['target'].astype(int)

    # We drop enrollee_id as it's not a feature
    if 'enrollee_id' in df.columns:
        df = df.drop(columns=['enrollee_id'])

    # 2. Standardize Ordinal Strings to Numbers
    # Experience: '>20' -> 21, '<1' -> 0
    experience_map = {
        '>20': '21',
        '<1': '0'
    }
    df['experience'] = df['experience'].replace(experience_map)
    # Convert to float to handle NaNs gracefully (int doesn't support NaN)
    df['experience'] = pd.to_numeric(df['experience'], errors='coerce')

    # Last New Job: '>4' -> 5, 'never' -> 0
    last_new_job_map = {
        '>4': '5',
        'never': '0'
    }
    df['last_new_job'] = df['last_new_job'].replace(last_new_job_map)
    df['last_new_job'] = pd.to_numeric(df['last_new_job'], errors='coerce')

    # 3. Clean Numerical Precision (Optional but good for consistency)
    if 'city_development_index' in df.columns:
        df['city_development_index'] = df['city_development_index'].round(3)

    return df


# ==========================================
# 3. Training Script
# ==========================================
def train():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    print("Cleaning data...")
    df_clean = clean_dataframe(df)

    # Separate Feature Matrix (X) and Target Vector (y)
    y = df_clean['target']
    X = df_clean.drop(columns=['target'])

    # Define Feature Groups
    # Note: 'experience' and 'last_new_job' are now numerical after cleaning
    numerical_features = [
        'city_development_index',
        'training_hours',
        'experience',
        'last_new_job'
    ]

    categorical_features = [
        'city',
        'gender',
        'relevent_experience',
        'enrolled_university',
        'education_level',
        'major_discipline',
        'company_size',
        'company_type'
    ]

    # --- Build the Pipeline ---
    print("Building pipeline...")

    # Preprocessing for Numerical Data: Impute Median -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for Categorical Data: Impute Mode -> OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine them
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not explicitly listed
    )

    # Final Pipeline: Preprocessor + Random Forest
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**RF_PARAMS))
    ])

    # --- Fit the Model ---
    print("Training Random Forest model on full dataset...")
    model_pipeline.fit(X, y)
    print("Training complete.")

    # --- Save the Model ---
    print(f"Saving model to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump(model_pipeline, f_out)

    print(f"Success! Model saved.")


if __name__ == "__main__":
    train()