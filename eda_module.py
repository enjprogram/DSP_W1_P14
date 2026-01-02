#EDA module

# Импорт стандартных модулей
import sys
import collections
from typing import List, Tuple

# Импорт сторонних библиотек
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

import decor_module as d

#----------------------------------Show Types and Stats------------------------------------------------
@d.data_descr_decorator
def view_data(df):

    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.dtypes)
    return df


#-----------------------------------Histograms---------------------------------------------------------

@d.hist_decorator
def plot_histogram(df):
    
    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    # Установка стиля Seaborn для красивых графиков
    sns.set(style="whitegrid")
    
    # Создание гистограмм для каждой числовой переменной
    df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
    
    # Добавление названий для каждого графика и осей
    for ax in plt.gcf().get_axes():
        ax.set_xlabel('Значение')
        ax.set_ylabel('Частота')
        ax.set_title(ax.get_title())
        #ax.set_title(ax.get_title().replace('wine_class', 'Класс вина'))
    
    # Регулировка макета для предотвращения наложения подписей
    plt.tight_layout()
    
    # Показать график
    plt.show()


#-------------------------------------Heatmaps----------------------------------------------------------
@d.heatmap_decorator
def plot_heatmap(df):
    
    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    # Установка стиля Seaborn
    sns.set(style="white")
    
    # Расчет корреляционной матрицы только для числовых данных
    numeric_df = df.select_dtypes(include=[np.number])  # Исключаем нечисловые столбцы
    corr = numeric_df.corr()
    
    # Маска для отображения только нижней треугольной части матрицы (опционально)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Настройка цветовой палитры
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Создание тепловой карты
    plt.figure(figsize=(30, 16))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    # Добавление заголовка
    plt.title('Тепловая карта корреляций', fontsize=20)
    
    # Показать график
    plt.show()


#------------------------------------WhiskerBox-----------------------------------------------------------
@d.whisker_box_decorator
def plot_whisker_box(df):

    if df is None:
        print("Данные не загружены. Вызовите метод load_data().")
        return
    # Установка стиля Seaborn
    sns.set(style="whitegrid")
    
    # Предполагаем, что df — это ваш DataFrame
    # Создаем ящики с усами для каждой колонки в DataFrame
    plt.figure(figsize=(12, 50))
    
    # Перебираем каждый числовой столбец и создаем для него ящик с усами
    for index, column in enumerate(df.select_dtypes(include=[np.number]).columns):
        plt.subplot((len(df.columns) // 3) + 1, 3, index + 1)
        sns.boxplot(y=df[column])
    
    plt.tight_layout()
    plt.show()


#------------------------------------FeatureImportancePlot--------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg') # Non-interactive backed for containers
import pandas as pd
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.sparse import issparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse


def show_or_save(filename = None):
    if filename:
        plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
    #plot.close()

def plot_feature_importance_rf(model, X_train, feature_names=None, max_features=50):
    """
    Function to plot the top N feature importance of a trained Random Forest model.
    
    :param model: The trained RandomForestRegressor model.
    :param X_train: The training dataset (features), can be a dense or sparse matrix.
    :param feature_names: List of feature names to match with the feature importance. 
                          If None, feature names are taken from X_train.columns if X_train is a DataFrame.
    :param max_features: Maximum number of top features to display.
    """
    # Get feature importance
    importance = model.feature_importances_

    # Generate feature names if not provided
    if feature_names is None:
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        elif hasattr(X_train, 'shape'):
            feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
        else:
            raise ValueError("Feature names must be provided for non-DataFrame input.")

    if len(importance) != len(feature_names):
        raise ValueError(f"Number of features in 'importance' ({len(importance)}) does not match number of feature names ({len(feature_names)}).")

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    # Sort features by importance (descending)
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Limit to top max_features
    feature_importance_df = feature_importance_df.head(max_features)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xticks(rotation=90, ha='center')
    plt.subplots_adjust(bottom=0.3)  # Adjust for longer labels
    plt.ylabel('Importance')
    plt.title(f'Top {max_features} Feature Importances (Random Forest)')
    plt.tight_layout()
    
    #plt.show()
    show_or_save('plot.png')
    plt.close()
    print("Top Feature Importances:")
    print(feature_importance_df)

    return feature_importance_df




def get_feature_names(X_train, feature_names=None):
    """
    Returns feature names based on the type of X_train.
    :param X_train: The input features (DataFrame, NumPy array, or sparse matrix).
    :param feature_names: If not provided, will try to extract feature names from X_train.
    :return: List of feature names.
    """
    # Case 1: If X_train is a pandas DataFrame
    if hasattr(X_train, 'columns'):
        return X_train.columns.tolist()
    
    # Case 2: If X_train is a NumPy array
    elif isinstance(X_train, np.ndarray):
        return [f"Feature {i}" for i in range(X_train.shape[1])]
    
    # Case 3: If X_train is a sparse matrix (e.g., from OneHotEncoder)
    elif issparse(X_train):
        if feature_names is None:
            raise ValueError("Sparse matrix provided, but feature names must be explicitly passed.")
        return feature_names  # Feature names must be explicitly passed for sparse matrices
    
    # Default case: unsupported type for X_train
    else:
        raise ValueError("Unsupported type for X_train.")

def plot_feature_importance(model, X_train, feature_names=None):
    """
    Function to plot the feature importance of a trained model with a vertical bar plot.
    This function works for both Linear Regression and Random Forest models.
    
    :param model: The trained model (RandomForestRegressor, LinearRegression, etc.).
    :param X_train: The training dataset (features), can be a dense or sparse matrix.
    :param feature_names: List of feature names to match with the feature importance. 
                          If None, feature names are taken from X_train.columns if X_train is a DataFrame.
    """
    # Check if the model has 'feature_importances_' (e.g., RandomForest) or 'coef_' (LinearRegression)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_  # RandomForestRegressor, GradientBoosting, etc.
        importance_type = 'Feature Importance'
    elif hasattr(model, 'coef_'):  # For models like LinearRegression
        importance = model.coef_
        importance_type = 'Coefficients'
    else:
        raise ValueError("Model does not have 'feature_importances_' or 'coef_' attribute.")
    
    # Get feature names using the get_feature_names function
    feature_names = get_feature_names(X_train, feature_names)

    if len(importance) != len(feature_names):
        raise ValueError(f"Number of features in 'importance' ({len(importance)}) does not match number of feature names ({len(feature_names)}).")


    # Create a DataFrame for feature importance or coefficients
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort the features by importance (descending order)
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Plotting the feature importance (vertical histogram)
    plt.figure(figsize=(12, 8))  # Increase figure width to fit labels
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    
    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=90, ha='center')
    
    # Compress the x-axis to give more space for labels
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin to fit labels
    
    plt.ylabel('Importance')
    plt.title(f'{importance_type} - {model.__class__.__name__}')
    plt.tight_layout()  # Automatically adjust the layout to prevent clipping
    plt.show()
    
    # Output the feature importance table
    print(f"{importance_type}:")
    print(feature_importance_df)

    return feature_importance_df

