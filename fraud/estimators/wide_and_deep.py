"""
A DNN keras model
"""
from typing import List, Union, Dict, Any

from keras import Model
from tensorflow_transform import TFTransformOutput
from tfx.components.trainer.fn_args_utils import DataAccessor

import configs
from absl import logging
import tensorflow as tf
import tensorflow_transform as tft

import features
from tfx_bsl.public import tfxio


def _get_tf_examples_serving_signature(model: Model, tf_transform_output: TFTransformOutput):
    """Returns a serving signature that accepts `tensorflow.Example`."""
    # We need to track the layers in the model in order to save it.
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(configs.LABEL_KEY)
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)
        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model: Model, tf_transform_output: TFTransformOutput):
    """Returns a serving signature that applies tf.Transform to features."""
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn


def _input_fn(
        file_pattern: List[str],
        data_accessor: DataAccessor,
        tf_transform_output: TFTransformOutput,
        batch_size: int = 200
):
    """Generates features and label for tuning/training.
      Args:
        file_pattern: List of paths or patterns of input tfrecord files.
        data_accessor: DataAccessor for converting input to RecordBatch.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of returned dataset to combine in a single batch
      Returns:
        A dataset that contains (features, indices) tuple where features is a
          dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=features.transformed_name(configs.LABEL_KEY)
        ),
        tf_transform_output.transformed_metadata.schema
    ).repeat()


def _build_model(hidden_units: Union[int, List[int]], learning_rate: float):
    """Creates a DNN Keras model for classifying taxi data.
      Args:
        hidden_units: the layer sizes of the DNN (input layer first).
        learning_rate: [float], learning rate of the Adam optimizer.
      Returns:
        A keras Model.
    """
    return _wide_and_deep_classifier(
        wide_columns=features.indicator_columns,
        deep_columns=features.real_valued_columns + features.embedding_columns,
        dnn_hidden_units=hidden_units,
        learning_rate=learning_rate
    )


def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units, learning_rate):
    """Build a simple keras wide and deep model.
      Args:
        wide_columns: Feature columns wrapped in indicator_column for wide (linear)
          part of the model.
        deep_columns: Feature columns for deep part of the model.
        dnn_hidden_units: [int], the layer sizes of the hidden DNN.
        learning_rate: [float], learning rate of the Adam optimizer.

      Returns:
        A Wide and Deep Keras model
    """
    # Keras needs the feature definitions at compile time.
    float_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
        for colname in features.transformed_names(configs.DENSE_FLOAT_FEATURE_KEYS)
    }
    int_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in
        features.transformed_names(configs.EMBED_FEATURE_KEYS) +
        features.transformed_names(configs.BUCKET_FEATURE_KEYS) +
        features.transformed_names(configs.OHE_FEATURE_KEYS)
    }
    input_layers = {**float_layers, **int_layers}

    # Keras preprocessing layers.
    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
    for num_nodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(num_nodes)(deep)
    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(tf.keras.layers.concatenate([deep, wide]))
    output = tf.squeeze(output, -1)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
        ]
    )
    model.summary(print_fn=logging.info)
    return model


# TFX Trainer will call this function.
def run_fn(fn_args):
    """Train the model based on given args.
    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, tf_transform_output, configs.TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, tf_transform_output, configs.EVAL_BATCH_SIZE)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_model(hidden_units=configs.HIDDEN_UNITS, learning_rate=configs.LEARNING_RATE)

    # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=configs.TENSORBOARD_LOG_DIR,
        histogram_freq=1,
        write_images=False,
        write_graph=False,
        update_freq='batch',

    )

    model.fit(
        train_dataset,
        epochs=configs.MODEL_EPOCHS,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
    )

    signatures = {
        'serving_default': _get_tf_examples_serving_signature(model, tf_transform_output),
        'transform_features': _get_transform_features_signature(model, tf_transform_output),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
