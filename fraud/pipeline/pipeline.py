"""TFX fraud pipeline definition."""

from typing import List, Optional, Text

import configs
import tensorflow_model_analysis as tfma
from google.protobuf.wrappers_pb2 import BoolValue
from ml_metadata.proto import metadata_store_pb2
from tensorflow_model_analysis import Options
from tfx import v1 as tfx


def create_pipeline(
        pipeline_name: Text,
        pipeline_root: Text,
        data_path: Text,
        preprocessing_fn: Text,
        run_fn: Text,
        train_args: tfx.proto.TrainArgs,
        eval_args: tfx.proto.EvalArgs,
        eval_accuracy_threshold: float,
        serving_model_dir: Text,
        metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
        beam_pipeline_args: Optional[List[Text]] = None,
) -> tfx.dsl.Pipeline:
    """Implements the chicago taxi pipeline with TFX."""

    components = []

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = tfx.components.CsvExampleGen(input_base=data_path)
    components.append(example_gen)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
    components.append(schema_gen)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(  # pylint: disable=unused-variable
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    components.append(example_validator)

    # Performs transformations and feature engineering in training and serving.
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        preprocessing_fn=preprocessing_fn)
    components.append(transform)

    # Uses user-provided Python function that implements a model using TF-Learn.
    trainer_args = {
        'run_fn': run_fn,
        'examples': transform.outputs['transformed_examples'],
        'schema': schema_gen.outputs['schema'],
        'transform_graph': transform.outputs['transform_graph'],
        'train_args': train_args,
        'eval_args': eval_args,
    }
    trainer = tfx.components.Trainer(**trainer_args)
    components.append(trainer)

    # Uses TFMA to compute a evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    eval_config = tfma.EvalConfig(
        options=Options(include_default_metrics=BoolValue(value=True)),
        model_specs=[
            tfma.ModelSpec(
                signature_name='serving_default',
                label_key=f'{configs.LABEL_KEY}_xf',
                preprocessing_function_names=['transform_features'])
        ],
        slicing_specs=[tfma.SlicingSpec(feature_keys=[spec]) for spec in configs.SLICE_BY],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': eval_accuracy_threshold}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ])
        ])
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=None,
        eval_config=eval_config)
    components.append(evaluator)

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher_args = {'model': trainer.outputs['model'], 'model_blessing': evaluator.outputs['blessing'],
                   'push_destination': tfx.proto.PushDestination(
                       filesystem=tfx.proto.PushDestination.Filesystem(base_directory=serving_model_dir)
                   )}
    pusher = tfx.components.Pusher(**pusher_args)  # pylint: disable=unused-variable
    components.append(pusher)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )
