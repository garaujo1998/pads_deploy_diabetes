"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 1.6.0
"""

import pandas as pd
from typing import Any
from importlib import import_module

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

def _load_class(class_path: str) -> type:

    module_name, class_name = class_path.rsplit('.', 1)

    return getattr(import_module(module_name), class_name)

def train_model(
        master_table: pd.DataFrame,
        columns: dict[str, Any],
        params: dict[str, Any]
) -> dict[str, Any]:

    estimators = {}

    target = params['target_column']

    for class_path in params['class_path']:

        df_train = master_table[master_table['split'].isin(params['train_splits'])]

        feature_cols = columns['numerical']# + columns['categorical'] # Nosso setup nao tem categoricas

        X_train = df_train[feature_cols]
        y_train = df_train[target]

        cls = _load_class(class_path)

        estimator = cls(**params.get('init_args', {}).get(class_path, {}))

        estimator.fit(X_train, y_train)

        estimators[class_path] = estimator 

    return {
        'estimators': estimators,
        'target_column': target,
        'feature_columns': feature_cols,
        'eval_splits': params.get('eval_splits', [])
    }

def evaluate_model(
        model_artifact: dict[str, Any],
        master_table: pd.DataFrame
    ) -> dict[str, Any]:

    estimators = model_artifact['estimators']
    target = model_artifact['target_column']
    feature_cols = model_artifact['feature_columns']
    eval_splits = model_artifact['eval_splits']

    all_metrics: dict[str, Any] = {}

    for class_path, estimator in estimators.items():

        model_metrics: dict[str, Any] = {}

        for split_name in eval_splits:
            split_df = master_table[master_table['split'] == split_name]
            X_split = split_df[feature_cols]
            y_split = split_df[target]

            y_pred = estimator.predict(X_split)
            y_proba = estimator.predict_proba(X_split)[:, 1]

            split_metrics = {
                'accuracy'  : float(accuracy_score(y_split, y_pred)),
                'recall'    : float(recall_score(y_split, y_pred)),
                'precision' : float(precision_score(y_split, y_pred)),
                'f1_macro'  : float(f1_score(y_split, y_pred, average = 'macro')),
                'roc_auc'   : float(roc_auc_score(y_split, y_proba)),
                'n_samples' : int(len(y_split))
            }

            model_metrics[split_name] = split_metrics

        all_metrics[class_path] = model_metrics

    return all_metrics

def select_best_model(
        model_artifact: dict[str, Any],
        metrics: dict[str, Any],
        params: dict[str, Any]
    ) -> dict[str, Any]:

    split = params['selection_split']
    metric = params['selection_metric']

    best_class_path = max(
        metrics,
        key = lambda class_path: metrics[class_path][split][metric]
    )

    return {
        'estimator': model_artifact['estimators'][best_class_path],
        'class_path': best_class_path,
        'target_column': model_artifact['target_column'],
        'feature_columns': model_artifact['feature_columns'],
        'selection_split': split,
        'selection_metric': metric,
        'selection_score': metrics[best_class_path][split][metric]
    }