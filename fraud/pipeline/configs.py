"""TFX taxi template configurations.

This file defines environments for a TFX taxi pipeline.
"""

import os  # pylint: disable=unused-import

# Where pipeline temp outputs get stored
OUTPUT_DIR = '/Documents/dev/learn/tfx/pipeline_outputs'

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = 'fraud'
PREPROCESSING_FN = 'models.preprocessing.preprocessing_fn'
RUN_FN = 'models.keras_model.model.run_fn'

TRAIN_NUM_STEPS = 1000
EVAL_NUM_STEPS = 150

# Change this value according to your use cases.
EVAL_ACCURACY_THRESHOLD = 0.6
