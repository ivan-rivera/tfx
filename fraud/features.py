"""TFX taxi model features.

Define constants here that are common across all estimators
including features names, label and size of vocabulary.
"""
from typing import Text, List

from keras.layers import Input, Reshape, Embedding, CategoryEncoding

import configs


def transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + '_xf'


def vocabulary_name(key: Text) -> Text:
    """Generate the name of the vocabulary feature from original name."""
    return key + '_vocab'


def transformed_names(keys: List[Text]) -> List[Text]:
    """Transform multiple feature names at once."""
    return [transformed_name(key) for key in keys]


# note that the features have to be wrapped into functions because some of them must be initialised
# within a strategy scope as opposed to on import
def get_real_valued_columns():
    real_valued_columns = {"inputs": {}, "processed": []}
    for key in transformed_names(configs.DENSE_FLOAT_FEATURE_KEYS):
        in_col = Input(shape=(1,), dtype="float32")
        real_valued_columns["inputs"][key] = in_col
        real_valued_columns["processed"].append(in_col)
    return real_valued_columns


def get_indicator_columns():
    indicator_columns = {"inputs": {}, "processed": []}
    for key, buckets in zip(
            transformed_names(configs.BUCKET_FEATURE_KEYS) +
            transformed_names(configs.OHE_FEATURE_KEYS),
            configs.BUCKET_FEATURE_BUCKET_COUNT +
            configs.OHE_FEATURE_MAX_VALUES):
        in_col = Input(shape=(1,), dtype="int64")
        indicator_columns["inputs"][key] = in_col
        indicator_columns["processed"].append(CategoryEncoding(buckets, output_mode="one_hot")(in_col))
    return indicator_columns


def get_embedded_columns():
    embedding_columns = {"inputs": {}, "processed": []}
    for key, in_dim, out_dim in zip(
            transformed_names(configs.EMBED_FEATURE_KEYS),
            configs.EMBED_COL_CARDINALITY,
            configs.EMBEDDING_DIMS):
        in_col = Input(shape=(1,), dtype="int64")
        embedded = Embedding(in_dim + 1, out_dim)(in_col)  # +1 is needed here to account for out of vocab values
        embedded = Reshape((out_dim,))(embedded)
        embedding_columns["inputs"][key] = in_col
        embedding_columns["processed"].append(embedded)
    return embedding_columns
