# ml_module.py
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

import eda_module as eda
import decor_module as d

import warnings
from tqdm import TqdmWarning

# Ignore tqdm notebook warnings
warnings.filterwarnings("ignore", category=TqdmWarning)


def load_data(file_path):
    """
    Загрузка данных из CSV файла.
    :param file_path: Путь к CSV файлу.
    :return: DataFrame с загруженными данными.
    """
    file_path = str(file_path) # ensuring the file_path is a string
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found..')

        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_index:
                zip_files = zip_index.namelist()
                print(f'Files in zip {zip_files}')
                zip_file = zip_files[0]
                with zip_index.open(zip_file) as file:
                    return _load_by_extension(file_path, zip_file)
                    
        file_ext = os.path.splitext(file_path)[1].lower()

        return _load_by_extension(file_path, file_ext)

    except FileNotFoundError as e:
        print(f"File not found: {e}")

    except ValueError as e:
        print(f'Value error: {e}')

    except Exception as e:
        print(f'Undefined error: {e}')

    finally:
        print(f'Attempted data loading..')


def _load_by_extension(file_path, file_ext):
    
    if file_ext == '.csv':
        return pd.read_csv(file_path)

    elif file_ext == '.json':
        return pd.read_json(file_path)

    elif file_ext == '.txt':
        return pd.read_txt(file_path, delimiter = '\t') # tab separation is assumed

    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)

    elif file_ext == '.parquet':
        return pd.read_parquet(file_path)

    else:
        raise ValueError(f'Unsupported file {file_ext}')
    
def _preprocess_date_columns(df, date_columns):

    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors = 'coerce')

    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_weekday'] = df[col].dt.weekday
    df[f'{col}_day'] = df[col].dt.day
    df[f'{col}_quarter'] = df[col].dt.quarter
    df[f'{col}_day_of_year'] = df[col].dt.dayofyear

    # cyclic encoding for month and weekday
    df[f'{col}_month_sin'] = np.sin(2*np.pi*df[f'{col}_month']/12)
    df[f'{col}_month_cos'] = np.cos(2*np.pi*df[f'{col}_month']/12)
    df[f'{col}_weekday_sin'] = np.sin(2*np.pi*df[f'{col}_weekday']/7)
    df[f'{col}_weekday_cos'] = np.cos(2*np.pi*df[f'{col}_weekday']/7)

    df[f'{col}_is_weekend'] = df[f'{col}_weekday'].isin([5,6])

    return df

def preprocess_data(df, drop_columns, target_column):
    """
    Предобработка данных: разделение на признаки и целевую переменную, масштабирование признаков.
    :param df: DataFrame с данными.
    :param target_column: Имя столбца с целевой переменной.
    :return: Обработанные признаки, целевая переменная, препроцессор.
    """

    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    #df['date'] = pd.to_datetime(df['date'])

    date_columns = ['payment_date', 'procedure_date']
    df = _preprocess_date_columns(df, date_columns)
    
    df = df.set_index('procedure_description', inplace = False)
    df = df.drop(columns = drop_columns) 
    #df = df.dropna() # remove nan values
    df = df.drop_duplicates() # remove duplicates

    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp.fit(df)
    # imp = SimpleImputer(strategy = "most_frequent" )
    # imp.fit_transform(df)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Определение числовых и категориальных признаков
    # numeric_features = ['patient_age']
    # categorical_features = ['patient_gender', 'patient_race_ethnicity', 'patient_zip_code', 'insurance_type',\
    #     'provider_name', 'provider_city', 'provider_state', 'provider_postal_code', 'procedure_outcome']

    
    numeric_features = X.select_dtypes(include = ['number']).columns
    categorical_features = X.select_dtypes(include = ['object']).columns
    
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    for col in categorical_features:
        X[col] = X[col].fillna(X[col].mode()[0])
    
    # Создание препроцессора
    # numeric_transformer = StandardScaler()
    # categorical_transformer = OneHotEncoder(drop='first')
    # date_transformer = _preprocess_date_columns(df, date_columns)

    numeric_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('onehot', OneHotEncoder(drop = 'first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    #print('features: ', X.columns.tolist())
    
    # Применение препроцессора к данным
    X_processed = preprocessor.fit_transform(X)
    print(X_processed, y)
    print("Данные успешно предобработаны.")

    # Extract feature names after transformations
     # Extract the one-hot encoded feature names from the encoder
    categorical_feature_names = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
    
    # Combine numeric and categorical feature names
    feature_names = numeric_features.tolist() + categorical_feature_names.tolist()
    print('features',feature_names)
    
    return X_processed, y




import optuna
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def objective(trial, X_train, y_train):
    """
    Objective function for Optuna hyperparameter optimization.
    We define the hyperparameter space and optimize for MAE and R2.
    """
    
    # Hyperparameters for RandomForestRegressor
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 10, 50, step=5)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    # Correct the max_features parameter selection
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])  # Removing 'auto'
    
    # Initialize RandomForestRegressor with hyperparameters
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Predict and calculate metrics
    y_pred = rf_model.predict(X_train)
    
    # Calculate MAE and R^2 score
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    
    # Objective: minimize a weighted sum of MAE and the negative of R2
    weight_mae = 0.5
    weight_r2 = 0.5
    
    # We negate R^2 because Optuna minimizes the objective function
    return weight_mae * mae - weight_r2 * r2  # We subtract R^2 since we want to maximize it


def train_model(X, y):
    """
    Train a model using Optuna hyperparameter tuning for MAE and R².
    :param X: Features
    :param y: Target
    :return: Trained models and best hyperparameters.
    """
    
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 1: Train Linear Regression model (without hyperparameter tuning)
    # print("Training Linear Regression model...")
    # linear_model = LinearRegression()
    # linear_model.fit(X_train, y_train)
    # y_pred = linear_model.predict(X_test)
    # print("Linear Regression: MAE =", mean_absolute_error(y_test, y_pred), ", R^2 =", r2_score(y_test, y_pred))
    
    # Step 2: Hyperparameter tuning using Optuna for RandomForestRegressor
    print("Optimizing Random Forest Regressor with Optuna for MAE and R²...")
    
    # Optuna study setup
    study = optuna.create_study(direction='minimize')  # We minimize the objective
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)  # Run 50 trials
    
    # Get the best hyperparameters
    best_rf_params = study.best_params
    print(f"Best Random Forest hyperparameters: {best_rf_params}")
    
    # Train Random Forest model with the best parameters
    best_rf_model = RandomForestRegressor(**best_rf_params, random_state=42)
    best_rf_model.fit(X_train, y_train)
    
    # Evaluate the best model on the test set
    rf_y_pred = best_rf_model.predict(X_test)
    print("Random Forest: MAE =", mean_absolute_error(y_test, rf_y_pred), ", R^2 =", r2_score(y_test, rf_y_pred))

    #model = linear_model
    model = best_rf_model
    
    #return linear_model, best_rf_model, best_rf_params
    return model


# def train_model(X, y):
    # """
    # Обучение модели линейной регрессии.
    # :param X: Признаки.
    # :param y: Целевая переменная.
    # :return: Обученная модель.
    # """
    # """
    # Обучение модели на обучающих данных.
    # """
    # if X is None or y is None:
    #     print("Данные не загружены или не предобработаны.")
    #     return

    # try:
    #     model = LinearRegression()
    #     model.fit(X, y)
    #     print("Модель успешно обучена.")
    #     return model
    # except Exception as e:
    #     print(f"Ошибка при обучении модели: {e}")


def predict(model, X):
    """
    Предсказание на новых данных.
    :param model: Обученная модель.
    :param X: Признаки.
    :return: Предсказанные значения.
    """
    return model.predict(X)

def evaluate_model(y_true, y_pred):
    """
    Оценка модели с использованием метрик MSE и R^2.
    :param y_true: Истинные значения.
    :param y_pred: Предсказанные значения.
    :return: MSE, R^2.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2