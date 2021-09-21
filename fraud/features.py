"""TFX taxi model features.

Define constants here that are common across all estimators
including features names, label and size of vocabulary.
"""

import configs
from typing import Text, List
import tensorflow as tf


def transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + '_xf'


def vocabulary_name(key: Text) -> Text:
    """Generate the name of the vocabulary feature from original name."""
    return key + '_vocab'


def transformed_names(keys: List[Text]) -> List[Text]:
    """Transform multiple feature names at once."""
    return [transformed_name(key) for key in keys]


# TODO: switch to keras preprocessing

real_valued_columns = [
    tf.feature_column.numeric_column(key, shape=())
    for key in transformed_names(configs.DENSE_FLOAT_FEATURE_KEYS)
]

embedded_columns = [
    tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
        key,
        num_buckets=num_buckets,
        default_value=0
    )
    for key, num_buckets in zip(transformed_names(configs.EMBED_FEATURE_KEYS), configs.EMBED_COL_CARDINALITY)
]

bucketed_columns = [
    tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
        key,
        num_buckets=num_buckets,
        default_value=0
    )
    for key, num_buckets in zip(
        transformed_names(configs.BUCKET_FEATURE_KEYS),
        configs.BUCKET_FEATURE_BUCKET_COUNT
    )
]

categorical_columns = [
    tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
        key,
        num_buckets=num_buckets,
        default_value=0
    )
    for key, num_buckets in zip(
        transformed_names(configs.OHE_FEATURE_KEYS),
        configs.OHE_FEATURE_MAX_VALUES
    )
]

indicator_columns = [
    tf.feature_column.indicator_column(categorical_column)
    for categorical_column in categorical_columns + bucketed_columns
]

embedding_columns = [
    tf.feature_column.embedding_column(embedding_column, dim)
    for embedding_column, dim in zip(embedded_columns, configs.EMBEDDING_DIMS)
]
