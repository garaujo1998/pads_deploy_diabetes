"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 1.6.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import train_model, evaluate_model, select_best_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func = train_model,
            inputs = ['master_table', 'params:columns', 'params:modelling_baseline'],
            outputs = 'baseline_model',
            name = 'train_baseline_model'
        ),
        Node(
            func = evaluate_model,
            inputs = ['baseline_model', 'master_table'],
            outputs = 'baseline_metrics',
            name = 'evaluate_baseline_model'
        ),
        Node(
            func = select_best_model,
            inputs = ['baseline_model', 'baseline_metrics', 'params:model_selection'],
            outputs = 'best_model',
            name = 'select_best_model'
        )
    ])