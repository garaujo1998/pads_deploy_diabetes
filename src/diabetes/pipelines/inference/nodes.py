"""
This is a boilerplate pipeline 'inference'
generated using Kedro 1.6.0
"""

import pandas as pd
from typing import Any
from sklearn.preprocessing import LabelEncoder

def transform_inference_encoders(
    df_in: pd.DataFrame,
    encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:

    df_out = df_in.copy()

    for col, en in encoders.items():
        if col in df_out.columns:
            df_out[col] = en.transform(df_out[col].astype(str))

    return df_out

def predict(
        model_artifact: dict[str, Any],
        inference_data: pd.DataFrame
) -> list[dict[str, Any]]:

    estimator = model_artifact['estimator']
    feature_cols = model_artifact['feature_columns']

    X = inference_data[feature_cols]

    predictions = estimator.predict(X)
    proba = estimator.predict_proba(X)[:, 1]

    result = [
        {
            'index': idx,
            'prediction': int(p),
            'probability': float(prob)
        }
        for idx, (p, prob) in enumerate(zip(predictions, proba))
    ]

    return result
