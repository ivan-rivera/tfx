"""TFX taxi template configurations.

This file defines environments for a TFX taxi pipeline.
"""

import yaml
import os  # pylint: disable=unused-import

# paths
PROJECT_DIR = '/Users/ivan/Documents/dev/learn/tfx'
OUTPUT_DIR = f'{PROJECT_DIR}/pipeline_outputs'

with open(f'{PROJECT_DIR}/configs.yaml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

# pipeline: main parameters
PIPELINE_NAME = conf['meta']['pipeline_name']
PREPROCESSING_FN = conf['meta']['preprocessing_fn']
RUN_FN = conf['meta']['run_fn']

# model training
TRAIN_NUM_STEPS = conf['model']['steps']['train']
EVAL_NUM_STEPS = conf['model']['steps']['eval']

# tensorboard logs directory
TENSORBOARD_LOG_DIR = OUTPUT_DIR + '/tensorboard'

# features

# Name of features which have continuous float values
DENSE_FLOAT_FEATURE_KEYS = conf['features']['dense']
# Name of features which have continuous float values. These features will be
# bucketized using `tft.bucketize`, and will be used as categorical features.
BUCKET_FEATURE_KEYS = list(conf['features']['bucketed'].keys())
# Number of buckets used by tf.transform for encoding each feature. The length
# of this list should be the same with BUCKET_FEATURE_KEYS.
BUCKET_FEATURE_BUCKET_COUNT = list(conf['features']['bucketed'].values())
# Name of features which have categorical values which are mapped to integers.
# These features will be used as categorical features.
OHE_FEATURE_KEYS = list(conf['features']['one_hot_encoded'].keys())
# Number of buckets to use integer numbers as categorical features.
OHE_FEATURE_MAX_VALUES = list(conf['features']['one_hot_encoded'].values())
# Name of features which have string values and are mapped to integers.
EMBED_FEATURE_KEYS = list(conf['features']['embedded'].keys())
# cardinality of the embeddable columns
EMBED_COL_CARDINALITY = [k['cardinality'] for k in conf['features']['embedded'].values()]
# dimensionality of the embedding space per column
EMBEDDING_DIMS = [k['embedding'] for k in conf['features']['embedded'].values()]
# Keys
LABEL_KEY = conf['features']['label']
# Slice by keys for evaluation
SLICE_BY = conf['features']['slice_by']

# training
HIDDEN_UNITS = conf['model']['hidden_units']
LEARNING_RATE = conf['model']['learning_rate']
TRAIN_BATCH_SIZE = conf['model']['batch_size']['train']
EVAL_BATCH_SIZE = conf['model']['batch_size']['eval']

# lowest evaluation threshold
EVAL_ACCURACY_THRESHOLD = conf['model']['eval_accuracy_threshold']