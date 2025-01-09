import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle

# contant variables
DATASET_PATH = './data/tabela-fipe-historico-precos.csv'
MODEL_PATH = './random_forest_model_with_preprocessing.pkl'
TARGET_COLUMN = 'valor'
NUMERICAL_COLS = ['codigo_fipe', 'ano_modelo', 'ano_referencia', 'mes_referencia']
FREQ_COL = 'marca'
OHE_COL = 'classificacao_marca'

# utility functions
def snake_case(c):
    """
    Converts a string from camelCase or PascalCase to snake_case.
    
    Args:
        c (str): The string to convert.

    Returns:
        str: The converted string in snake_case.
    """
    return re.sub(r'(?<!^)(?=[A-Z])', '_', c).lower()

def preprocess_data(df):
    """
    Preprocess the input DataFrame by cleaning column names, formatting strings,
    handling specific columns, and adding new features.

    Args:
        df (pd.DataFrame): The input raw DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df.columns = [snake_case(c) for c in df.columns]

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_').str.replace('-', '_')

    df['codigo_fipe'] = df['codigo_fipe'].str.replace('-', '').astype('int64')

    if 'unnamed: 0' in df.columns:
        df.drop(columns=['unnamed: 0'], inplace=True)

    media_por_marca = df.groupby('marca')['valor'].mean()
    df['classificacao_marca'] = df['marca'].map(
        lambda marca: (
            'economical' if media_por_marca[marca] <= 50_000 else
            'affordable' if media_por_marca[marca] <= 100_000 else
            'mid_range' if media_por_marca[marca] <= 500_000 else
            'luxury' if media_por_marca[marca] <= 1_000_000 else
            'super_luxury' if media_por_marca[marca] <= 5_000_000 else
            'ultra_luxury'
        )
    )
    return df

def split_data(df, target_column, test_size=0.2, val_size=0.25, random_state=42):
    """
    Split the input DataFrame into training, validation, and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        test_size (float): Proportion of the test set.
        val_size (float): Proportion of the validation set within the remaining data.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Splits of features and target (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_full_train, test_size=val_size, random_state=random_state)

    X_train, y_train = df_train.drop(columns=[target_column]), df_train[target_column].values
    X_val, y_val = df_val.drop(columns=[target_column]), df_val[target_column].values
    X_test, y_test = df_test.drop(columns=[target_column]), df_test[target_column].values

    return X_train, X_val, X_test, y_train, y_val, y_test

def encode_features(X_train, X_val, X_test, freq_col, ohe_col):
    """
    Apply frequency encoding and one-hot encoding to specified columns.

    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.
        freq_col (str): Column to apply frequency encoding.
        ohe_col (str): Column to apply one-hot encoding.

    Returns:
        tuple: Encoded DataFrames (X_train_encoded, X_val_encoded, X_test_encoded).
    """
    freq_encoding_map = X_train[freq_col].value_counts() / len(X_train)
    X_train[f"{freq_col}_freq_encoded"] = X_train[freq_col].map(freq_encoding_map)
    X_val[f"{freq_col}_freq_encoded"] = X_val[freq_col].map(freq_encoding_map)
    X_test[f"{freq_col}_freq_encoded"] = X_test[freq_col].map(freq_encoding_map)

    X_train_encoded = pd.get_dummies(X_train, columns=[ohe_col], drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, columns=[ohe_col], drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=[ohe_col], drop_first=True)

    return X_train_encoded, X_val_encoded, X_test_encoded

def build_pipeline(numerical_cols, categorical_cols):
    """
    Build a machine learning pipeline with preprocessing and a regressor.

    Args:
        numerical_cols (list): List of numerical columns to pass through.
        categorical_cols (list): List of categorical columns to pass through.

    Returns:
        Pipeline: The machine learning pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', 'passthrough', categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=500,
                                            min_samples_leaf=1,
                                            min_samples_split=2,
                                            random_state=42))
    ])
    return pipeline

def train_and_save_model(pipeline, X_train, y_train, model_path):
    """
    Train the machine learning model and save it to a file.

    Args:
        pipeline (Pipeline): The machine learning pipeline.
        X_train (pd.DataFrame): Training features.
        y_train (np.array): Training target values.
        model_path (str): Path to save the trained model.
    """
    pipeline.fit(X_train, y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved at: {model_path}")

# main function
def main():
    """
    Main function to execute the entire data processing and model training workflow.
    """
    df_raw = pd.read_csv(DATASET_PATH)
    df = preprocess_data(df_raw.copy())

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, TARGET_COLUMN)
    X_train, X_val, X_test = encode_features(X_train, X_val, X_test, FREQ_COL, OHE_COL)

    categorical_cols = [f"{FREQ_COL}_freq_encoded"] + [col for col in X_train.columns if col.startswith(OHE_COL)]
    pipeline_rf = build_pipeline(NUMERICAL_COLS, categorical_cols)

    train_and_save_model(pipeline_rf, X_train, y_train, MODEL_PATH)

if __name__ == '__main__':
    main()