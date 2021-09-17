"""TFX taxi model features.

Define constants here that are common across all models
including features names, label and size of vocabulary.
"""

from typing import Text, List


def transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + '_xf'


def vocabulary_name(key: Text) -> Text:
    """Generate the name of the vocabulary feature from original name."""
    return key + '_vocab'


def transformed_names(keys: List[Text]) -> List[Text]:
    """Transform multiple feature names at once."""
    return [transformed_name(key) for key in keys]
