"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.6.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import clean_data, feature_extraction_columns, add_split_column, fit_encoders, transform_encoders, fit_scalers, transform_scalers

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func = clean_data,
            inputs = ['raw_diabetes_data', 'params:original_columns'],
            outputs = 'cleaned_diabetes_data',
            name = 'clean_data'
        ),        
        Node(
            func = feature_extraction_columns,
            inputs = ['cleaned_diabetes_data'],
            outputs = 'enriched_diabetes_data',
            name = 'feature_extraction_columns'
        ),
        Node(
            func = add_split_column,
            inputs = ['enriched_diabetes_data', 'params:split'],
            outputs = 'split_diabetes_data',
            name = 'add_split_column'
        ),
        Node(
            func = fit_encoders,
            inputs = ['split_diabetes_data', 'params:columns', 'params:modelling_encoders'],
            outputs = 'modelling_encoders',
            name = 'fit_encoders'
        ),
        Node(
            func = transform_encoders,
            inputs = ['split_diabetes_data', 'modelling_encoders'],
            outputs = 'encoded_diabetes_data',
            name = 'transform_encoders'
        ),
        Node(
            func = fit_scalers,
            inputs = ['encoded_diabetes_data', 'params:columns', 'params:modelling_scalers'],
            outputs = 'modelling_scalers',
            name = 'fit_scalers'
        )
        ,
        Node(
            func = transform_scalers,
            inputs = ['encoded_diabetes_data', 'modelling_scalers'],
            outputs = 'master_table',
            name = 'transform_scalers'
        )
    ])