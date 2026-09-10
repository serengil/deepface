# built-in dependencies
from typing import Any, List, cast

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

# this module is framework agnostic on purpose. the labels and the post processing below
# do not depend on a backend engine, so the tensorflow and the pytorch models share them.

# Labels for the emotions that can be detected by the emotion model.
EMOTION_LABELS: List[str] = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Labels for the genders that can be detected by the gender model.
GENDER_LABELS: List[str] = ["Woman", "Man"]

# Labels for the ethnic phenotypes that can be detected by the race model.
RACE_LABELS: List[str] = [
    "asian",
    "indian",
    "black",
    "white",
    "middle eastern",
    "latino hispanic",
]


def find_apparent_age(age_predictions: NDArray[Any]) -> np.float64:
    """
    Find apparent age prediction from a given probas of ages
    Args:
        age_predictions (age_classes,)
    Returns:
        apparent_age (float)
    """
    assert (
        len(age_predictions.shape) == 1
    ), f"Input should be a list of predictions, not batched. Got shape: {age_predictions.shape}"
    output_indexes = np.arange(0, 101)
    apparent_age = cast(np.float64, np.sum(age_predictions * output_indexes))
    return apparent_age
