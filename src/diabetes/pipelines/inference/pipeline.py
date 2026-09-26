"""
This is a boilerplate pipeline 'inference'
generated using Kedro 1.6.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from diabetes.pipelines.data_engineering.nodes import clean_data, feature_extraction_columns, transform_scalers
from .nodes import transform_inference_encoders, predict

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func = clean_data,
            inputs = ['raw_inference_data', 'params:inference_columns'],
            outputs = 'cleaned_inference_data',
            name = 'clean_inference_data'
        ),
        Node(
            func = feature_extraction_columns,
            inputs = ['cleaned_inference_data'],
            outputs = 'enriched_inference_data',
            name = 'enrich_inference_data'
        ),
        Node(
            func = transform_inference_encoders,
            inputs = ['enriched_inference_data', 'modelling_encoders'],
            outputs = 'encoded_inference_data',
            name = 'encode_inference_data'
        ),
        Node(
            func = transform_scalers,
            inputs = ['encoded_inference_data', 'modelling_scalers'],
            outputs = 'scaled_inference_data',
            name = 'scale_inference_data'
        ),
        Node(
            func = predict,
            inputs = ['best_model', 'scaled_inference_data'],
            outputs = 'inference_predictions',
            name = 'predict'
        )
    ])
