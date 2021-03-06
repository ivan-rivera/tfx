"""TFX taxi preprocessing.

This file defines a template for TFX Transform component.
"""

import tensorflow as tf
import tensorflow_transform as tft

import features
import configs


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.

      Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

      Args:
        x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
          in the second dimension.

      Returns:
        A rank 1 tensor where missing values of `x` have been filled in.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value
        ),
        axis=1
    )


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

      Args:
        inputs: map from feature keys to raw not-yet-transformed features.

      Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}
    for key in configs.DENSE_FLOAT_FEATURE_KEYS:
        # Preserve this feature as a dense float, setting nan's to the mean.
        outputs[features.transformed_name(key)] = tft.scale_to_z_score(_fill_in_missing(inputs[key]))

    for key, num_buckets in zip(configs.EMBED_FEATURE_KEYS, configs.EMBED_COL_CARDINALITY):
        # Build a vocabulary for this feature
        # Note that it is important to use the default value of zero because otherwise the
        # embedding layer in Keras is not going to be able to deal with out of vocabulary values
        outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]), top_k=num_buckets, num_oov_buckets=0, default_value=0)

    for key, num_buckets in zip(configs.BUCKET_FEATURE_KEYS, configs.BUCKET_FEATURE_BUCKET_COUNT):
        outputs[features.transformed_name(key)] = tft.bucketize(_fill_in_missing(inputs[key]), num_buckets)

    for key, num_buckets in zip(configs.OHE_FEATURE_KEYS, configs.OHE_FEATURE_MAX_VALUES):
        if inputs[key].dtype == tf.string:
            outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
                _fill_in_missing(inputs[key]), top_k=num_buckets, num_oov_buckets=0)
        else:
            outputs[features.transformed_name(key)] = _fill_in_missing(inputs[key])

    outputs[features.transformed_name(configs.LABEL_KEY)] = _fill_in_missing(inputs[configs.LABEL_KEY])

    return outputs
