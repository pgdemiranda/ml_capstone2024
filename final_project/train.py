import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import pickle

FILEPATH = './data/tabela-fipe-historico-precos.csv'
FILENAME = './final_model.pkl'

def load_and_clean_data(filepath):
    df_raw = pd.read_csv(filepath)
    df = df_raw.copy()

    def snake_case(c):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', c).lower()
    df.columns = [snake_case(c) for c in df.columns]

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_').str.replace('-', '_')
    df['codigo_fipe'] = df['codigo_fipe'].str.replace('-', '').astype('int64')
    df.drop(columns=['unnamed: 0'], inplace=True, errors='ignore')
    return df

def feature_engineering(df):
    media_por_marca = df.groupby('marca')['valor'].mean()
    df['classificacao_marca'] = df['marca'].map(
        lambda marca: (
            'economical' if media_por_marca[marca] <= 50_000 else
            'affordable' if media_por_marca[marca] <= 100_000 else
            'mid_range' if media_por_marca[marca] <= 500_000 else
            'luxury' if media_por_marca[marca] <= 1_000_000 else
            'super_luxury'
        )
    )
    return df

def split_data(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    X_train = df_train.reset_index(drop=True)
    X_val = df_val.reset_index(drop=True)
    X_test = df_test.reset_index(drop=True)

    y_train = X_train.pop('valor').values
    y_val = X_val.pop('valor').values
    y_test = X_test.pop('valor').values

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_preprocessor(X_train, X_val, X_test):
    freq_encoding_map = X_train['marca'].value_counts() / len(X_train)
    X_train['marca_freq_encoded'] = X_train['marca'].map(freq_encoding_map)
    X_val['marca_freq_encoded'] = X_val['marca'].map(freq_encoding_map)
    X_test['marca_freq_encoded'] = X_test['marca'].map(freq_encoding_map)

    X_train = pd.get_dummies(X_train, columns=['classificacao_marca'])
    X_val = pd.get_dummies(X_val, columns=['classificacao_marca'])
    X_test = pd.get_dummies(X_test, columns=['classificacao_marca'])

    categorical_cols = ['marca_freq_encoded'] + [col for col in X_train.columns if col.startswith('classificacao_marca')]
    numerical_cols = ['codigo_fipe', 'ano_modelo', 'ano_referencia', 'mes_referencia']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', 'passthrough', categorical_cols)
        ]
    )

    X_train.drop(columns=['marca', 'modelo'], inplace=True) 
    X_val.drop(columns=['marca', 'modelo'], inplace=True) 
    X_test.drop(columns=['marca', 'modelo'], inplace=True)

    return preprocessor, X_train, X_val, X_test

def train_model(X_train, y_train, preprocessor):
    pipeline_lgb_final = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(
            random_state=42, 
            metric='rmse', 
            force_row_wise=True,
            bagging_fraction=1.0,
            bagging_freq=6,
            feature_fraction=0.55,
            lambda_l1=5.41,
            lambda_l2=8.62,
            learning_rate=0.045,
            max_depth=9,
            min_data_in_leaf=35,
            n_estimators=824,
            num_leaves=67,
            subsample=0.79
        ))
    ])

    pipeline_lgb_final.fit(X_train, y_train)
    return pipeline_lgb_final

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def main():
    df = load_and_clean_data(FILEPATH)
    df = feature_engineering(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    preprocessor, X_train, X_val, X_test = get_preprocessor(X_train, X_val, X_test)
    pipeline_lgb_final = train_model(X_train, y_train, preprocessor)
    save_model(pipeline_lgb_final, FILENAME)

if __name__ == '__main__':
    main()
