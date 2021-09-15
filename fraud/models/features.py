"""TFX taxi model features.

Define constants here that are common across all models
including features names, label and size of vocabulary.
"""

from typing import Text, List

# At least one feature is needed.

# Name of features which have continuous float values. These features will be
# used as their own values.
DENSE_FLOAT_FEATURE_KEYS = ['AMT_CREDIT', 'CNT_CHILDREN']

# Name of features which have continuous float values. These features will be
# bucketized using `tft.bucketize`, and will be used as categorical features.
BUCKET_FEATURE_KEYS = ['AMT_INCOME_TOTAL']
# Number of buckets used by tf.transform for encoding each feature. The length
# of this list should be the same with BUCKET_FEATURE_KEYS.
BUCKET_FEATURE_BUCKET_COUNT = [10]

# Name of features which have categorical values which are mapped to integers.
# These features will be used as categorical features.
CATEGORICAL_FEATURE_KEYS = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
# Number of buckets to use integer numbers as categorical features. The length
# of this list should be the same with CATEGORICAL_FEATURE_KEYS.
CATEGORICAL_FEATURE_MAX_VALUES = [3, 2, 2]

# Name of features which have string values and are mapped to integers.
VOCAB_FEATURE_KEYS = ['HOUSETYPE_MODE']

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

# Keys
LABEL_KEY = 'TARGET'


def transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + '_xf'


def vocabulary_name(key: Text) -> Text:
    """Generate the name of the vocabulary feature from original name."""
    return key + '_vocab'


def transformed_names(keys: List[Text]) -> List[Text]:
    """Transform multiple feature names at once."""
    return [transformed_name(key) for key in keys]