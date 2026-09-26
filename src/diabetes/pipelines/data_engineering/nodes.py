"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.6.0
"""

import pandas as pd
from typing import Any
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

def clean_data(
        raw_diabetes_data: pd.DataFrame,
        columns: dict[str, Any]
    ) -> pd.DataFrame:

    all_columns = [
        item
        for v in columns.values()
        for item in (v if isinstance(v, list) else [v])
        if item in raw_diabetes_data.columns
    ]

    df_flt = raw_diabetes_data[all_columns].copy() # dataframe filtered

    for c in columns['numerical']:
        df_flt[c] = pd.to_numeric(df_flt[c], errors='coerce')

    #Não há colunas catégoricas na base original
    #for c in columns['categorical']:
    #    df_flt[c] = df_flt[c].astype('str').str.strip().fillna('Unknown')

    logger.info(f"Cleaned data: {df_flt.shape[0]} rows, {df_flt.shape[1]}")

    return df_flt

def add_split_column(
    df_in: pd.DataFrame,
    split: dict[str, Any]
    ) -> pd.DataFrame:

    total = split['train'] + split['test'] + split['validate']
    probs = [split['train'], split['test'], split['validate']]

    if not np.isclose(total, 1.0):
        raise ValueError(f'Split proportions must sum to 1.0, got {total}')

    rng = np.random.default_rng(split['seed']) # Random Number Generator

    labels = rng.choice(
        a = ['train','test','validate'],
        size = len(df_in),
        p = probs,
        replace = True
    )

    df_out = df_in.assign(split=labels)

    return df_out

def feature_extraction_columns(
    df_in: pd.DataFrame
    ) -> pd.DataFrame:

    df_out = df_in.copy()

    # Criação de 'NEW_AGE_CAT'
    df_out.loc[(df_out['Age'] >= 21) & (df_out['Age'] < 50), 'NEW_AGE_CAT'] = 'mature'
    df_out.loc[df_out['Age'] >= 50, 'NEW_AGE_CAT'] = 'senior'

    # Criação de 'NEW_BMI'
    df_out['NEW_BMI'] = pd.cut(
        x=df_out['BMI'],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=['Underweight', 'Healthy', 'Overweight', 'Obese']
    )

    # Criação de 'NEW_GLUCOSE'
    df_out['NEW_GLUCOSE'] = pd.cut(
        x=df_out['Glucose'],
        bins=[0, 140, 200, 300],
        labels=['Normal', 'Prediabetes', 'Diabetes']
    )

    # Criação de NEW_AGE_BMI_NOM
    df_out.loc[(df_out['BMI'] < 18.5) & ((df_out['Age'] >= 21) & (df_out['Age'] < 50)), 'NEW_AGE_BMI_NOM'] = 'underweightmature'
    df_out.loc[(df_out['BMI'] < 18.5) & (df_out['Age'] >= 50), 'NEW_AGE_BMI_NOM'] = 'underweightsenior'
    df_out.loc[((df_out['BMI'] >= 18.5) & (df_out['BMI'] < 25)) & ((df_out['Age'] >= 21) & (df_out['Age'] < 50)), 'NEW_AGE_BMI_NOM'] = 'healthymature'
    df_out.loc[((df_out['BMI'] >= 18.5) & (df_out['BMI'] < 25)) & (df_out['Age'] >= 50), 'NEW_AGE_BMI_NOM'] = 'healthysenior'
    df_out.loc[((df_out['BMI'] >= 25) & (df_out['BMI'] < 30)) & ((df_out['Age'] >= 21) & (df_out['Age'] < 50)), 'NEW_AGE_BMI_NOM'] = 'overweightmature'
    df_out.loc[((df_out['BMI'] >= 25) & (df_out['BMI'] < 30)) & (df_out['Age'] >= 50), 'NEW_AGE_BMI_NOM'] = 'overweightsenior'
    df_out.loc[(df_out['BMI'] > 18.5) & ((df_out['Age'] >= 21) & (df_out['Age'] < 50)), 'NEW_AGE_BMI_NOM'] = 'obesemature'
    df_out.loc[(df_out['BMI'] > 18.5) & (df_out['Age'] >= 50), 'NEW_AGE_BMI_NOM'] = 'obesesenior'

    # Criação de NEW_AGE_GLUCOSE_NOM
    df_out.loc[(df_out['Glucose'] < 70) & ((df_out['Age'] >= 21) & (df_out['Age'] < 50)), 'NEW_AGE_GLUCOSE_NOM'] = 'lowmature'
    df_out.loc[(df_out['Glucose'] < 70) & (df_out['Age'] >= 50), 'NEW_AGE_GLUCOSE_NOM'] = 'lowsenior'
    df_out.loc[((df_out['Glucose'] >= 70) & (df_out['Glucose'] < 100)) & ((df_out['Age'] >= 21) & (df_out['Age'] < 50)), 'NEW_AGE_GLUCOSE_NOM'] = 'normalmature'
    df_out.loc[((df_out['Glucose'] >= 70) & (df_out['Glucose'] < 100)) & (df_out['Age'] >= 50), 'NEW_AGE_GLUCOSE_NOM'] = 'normalsenior'
    df_out.loc[((df_out['Glucose'] >= 100) & (df_out['Glucose'] <= 125)) & ((df_out['Age'] >= 21) & (df_out['Age'] < 50)), 'NEW_AGE_GLUCOSE_NOM'] = 'hiddenmature'
    df_out.loc[((df_out['Glucose'] >= 100) & (df_out['Glucose'] <= 125)) & (df_out['Age'] >= 50), 'NEW_AGE_GLUCOSE_NOM'] = 'hiddensenior'
    df_out.loc[(df_out['Glucose'] > 125) & ((df_out['Age'] >= 21) & (df_out['Age'] < 50)), 'NEW_AGE_GLUCOSE_NOM'] = 'highmature'
    df_out.loc[(df_out['Glucose'] > 125) & (df_out['Age'] >= 50), 'NEW_AGE_GLUCOSE_NOM'] = 'highsenior'

    # Criação de NEW_INSULIN_SCORE
    df_out['NEW_INSULIN_SCORE'] = df_out['Insulin'].between(16, 166).map({True: np.nan, False: 'Abnormal'})

    # Criação de NEW_GLUCOSE_X_INSULIN
    df_out['NEW_GLUCOSE_X_INSULIN'] = df_out['Glucose'] * df_out['Insulin']

    # Criação de NEW_GLUCOSE_X_PREGNANCIES
    df_out['NEW_GLUCOSE_X_PREGNANCIES'] = df_out['Glucose'] * df_out['Pregnancies']

    return df_out

def fit_encoders(
        df_in: pd.DataFrame,
        columns: dict[str, Any],
        params: dict[str, Any]
    ) -> dict[str, LabelEncoder]:

    encoders: dict[str, LabelEncoder] = {}

    cols_to_encode = [
        item
        for p in params['columns']
        for item in (columns[p] if isinstance(columns[p], list) else [columns[p]])
    ]

    for col in cols_to_encode:
        le = LabelEncoder()
        le.fit(df_in.loc[df_in['split'].isin(params['split_to_fit']), col].astype(str))
        encoders[col] = le

    logger.info(f'Fitted label encoders for {len(encoders)} columns')

    return encoders

def transform_encoders(
    df_in: pd.DataFrame,
    encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:

    df_out = df_in.copy()

    for col, en in encoders.items():
        df_out[col] = en.transform(df_out[col].astype(str))

    return df_out

def fit_scalers(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any]
) -> dict[str, StandardScaler]:

    scalers: dict[str, StandardScaler] = {}

    cols_to_scale = [
        item
        for p in params['columns']
        for item in (columns[p] if isinstance(columns[p], list) else [columns[p]])
    ]

    for col in cols_to_scale:
        sc = StandardScaler()
        sc.fit(df_in.loc[df_in['split'].isin(params['split_to_fit']), [col]])
        scalers[col] = sc

    logger.info(f'Fitted scalers for {len(scalers)} columns')

    return scalers

def transform_scalers(
    df_in: pd.DataFrame,
    scalers: dict[str, StandardScaler]
) -> pd.DataFrame:

    df_out = df_in.copy()

    ## Adicionar remocao hardcoded de valores estranhos.
    #df_out = df_out[df_out['Glucose']>0]       # Glucose não pode ser 0
    #df_out = df_out[df_out['Insulin']>0]       # Insulin não pode ser 0
    #df_out = df_out[df_out['BloodPressure']>0] # BloodPressure não pode ser 0
    #df_out = df_out[df_out['SkinThickness']>0] # SkinThickness não pode ser 0
    #df_out = df_out[df_out['BMI']>0]           # BMI não pode ser 0

    # Analisando pelo Debug Console, as operações acima reduzem o número de linhas do
    # df_out de 652 para 451.

    # Faço o scaling
    for col, sc in scalers.items():
        df_out[col] = sc.transform(df_out[[col]])

    return df_out