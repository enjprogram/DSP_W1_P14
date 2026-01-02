# procedure processing for predict module
import ml_module as ml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from pathlib import Path

# Base directory where app.py is located
#BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path.cwd()          # Jupyter-safe
MODELS_DIR = BASE_DIR / "models" 

# # Safe paths to the pickle files
# scaler_path = BASE_DIR / 'models' / 'scaler.pkl'
# selector_path = BASE_DIR / 'models' / 'selector.pkl'
# imputer_path = BASE_DIR / 'models' / 'imputer.pkl'

# Load the saved scaled and selector
def load_scaler():
    # Use joblib to load the model if it was saved using joblib
    scaler_path = MODELS_DIR / "scaler.pkl"
    scaler = joblib.load(scaler_path)
    return scaler

def load_selector():
    # Use joblib to load the model if it was saved using joblib
    selector_path = BASE_DIR / 'models' / 'selector.pkl'
    selector = joblib.load(selector_path)
    return selector

def load_imputer():
    # Use joblib to load the model if it was saved using joblib
    imputer_path = BASE_DIR / 'models' / 'imputer.pkl'
    imputer = joblib.load(imputer_path)
    return imputer

# Pre-processing of categorical features for regression
def map_categorical(df, columns=None):
    """
    Convert categorical columns into integer codes.
    Returns transformed DataFrame and mapping dictionary.
    """

    import pandas as pd

    df = df.copy()
    mapping = {}

    # Auto-detect categorical columns
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in columns:
        # Extract unique categories
        categories = df[col].astype("category").cat.categories
        
        # Create mapping: category - integer
        mapping[col] = {cat: i for i, cat in enumerate(categories)}

        # Replace categories with integers
        df[col] = df[col].map(mapping[col]).fillna(-1).astype(int)

    return df, mapping

def process(df):
    imputer = load_imputer()
    scaler = load_scaler()
    selector = load_selector()

    df_input = df.copy()
    # Для Age используем KNN импутацию
    print("Обработка возраста (Patient Age): KNN импутация")
    df_input['patient_age'] = imputer.transform(df_input[['patient_age']])
    
    # Feature Engineering
    
    # Creating date features
    date_columns = ['procedure_date']
    df_input = ml._preprocess_date_columns(df_input, date_columns)
    
    print(df_input.head())
    
    # Creating Age Categories
    df_input['AgeGroup'] = pd.cut(df_input['patient_age'],
                                     bins=[0, 12, 18, 35, 60, 100],
                                     labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Coding categorical variables
    nominal_features = ['AgeGroup']
    df_input_encoded = pd.get_dummies(df_input, columns=nominal_features, drop_first=True)
    
    # Удаляем колонки, которые не будем использовать
    columns_to_drop = ['record_id', 'patient_id', 'provider_id', 'procedure_code','provider_state', 'provider_country']
    df_input_encoded.drop(columns_to_drop, axis=1, inplace=True)
    
    print(f"Размер датасета после кодирования: {df_input_encoded.shape}")
    print(f"Количество признаков: {len(df_input_encoded.columns) - 1}")  # -1 для целевой переменной
    
    # # Разделяем на признаки и целевую переменную
    X = df_input_encoded.drop('cost_billed', axis=1)
    y = df_input_encoded['cost_billed']
    
    # Scaling numerical features using the saved scaler corresponding to the trained and saved model
    numerical_features = ['patient_age', 'cost_paid', 'cost_patient_responsibility', 'cost_insurance_covered', 'provider_postal_code', 'patient_zip_code']
    X[numerical_features] = scaler.transform(X[numerical_features])

    cat_cols_for_cost_regression = ['procedure_description', 'patient_gender', 'patient_race_ethnicity', 'insurance_type', 'procedure_outcome', 'procedure_date_is_weekend', 'provider_name', 'provider_city']
    X, mapping = map_categorical(X, columns = cat_cols_for_cost_regression)
    
    # Remove columns that are not going to be used
    further_columns_to_drop = ['payment_date','procedure_date', 'procedure_date_month_cos', 'procedure_date_month_sin', 'procedure_date_weekday_sin', 'procedure_date_weekday_cos']
    X.drop(further_columns_to_drop, axis=1, inplace=True)
    
    X_test_selected = selector.transform(X)

    return X_test_selected
    
