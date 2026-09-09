# built-in dependencies
from typing import Any, List, Tuple, Union, cast

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.commons import weight_utils
from deepface.models.spoofing import FasNetBackbone
from deepface.models.spoofing.FasNetUtils import crop
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long, too-few-public-methods, nested-min-max
FIRST_WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/2.7_80x80_MiniFASNetV2.h5"
SECOND_WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/4_0_0_80x80_MiniFASNetV1SE.h5"


class Fasnet:
    """
    Mini Face Anti Spoofing Net Library from repo: github.com/minivision-ai/Silent-Face-Anti-Spoofing

    Minivision's Silent-Face-Anti-Spoofing Repo licensed under Apache License 2.0
    Ref: github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/src/model_lib/MiniFASNet.py
    """

    def __init__(self) -> None:
        # download pre-trained models if not installed yet
        first_model_weight_file = weight_utils.download_weights_if_necessary(
            file_name="2.7_80x80_MiniFASNetV2.h5",
            source_url=FIRST_WEIGHTS_URL,
        )

        second_model_weight_file = weight_utils.download_weights_if_necessary(
            file_name="4_0_0_80x80_MiniFASNetV1SE.h5",
            source_url=SECOND_WEIGHTS_URL,
        )

        # Fasnet will use 2 distinct models to predict, then it will find the sum of predictions
        # to make a final prediction

        first_model = FasNetBackbone.MiniFASNetV2(conv6_kernel=(5, 5))
        second_model = FasNetBackbone.MiniFASNetV1SE(conv6_kernel=(5, 5))

        first_model = weight_utils.load_model_weights(
            model=first_model, weight_file=first_model_weight_file
        )
        second_model = weight_utils.load_model_weights(
            model=second_model, weight_file=second_model_weight_file
        )

        self.first_model = first_model
        self.second_model = second_model

    def analyze(
        self,
        img: NDArray[Any],
        facial_area: Union[List[Union[int, float]], Tuple[Union[int, float], ...]],
    ) -> Tuple[bool, float]:
        """
        Analyze a given image spoofed or not
        Args:
            img (np.ndarray): pre loaded image
            facial_area (list or tuple): facial rectangle area coordinates with x, y, w, h respectively
        Returns:
            result (tuple): a result tuple consisting of is_real and score
        """
        x, y, w, h = facial_area
        first_img = crop(img, (x, y, w, h), 2.7, 80, 80)
        second_img = crop(img, (x, y, w, h), 4, 80, 80)

        # pixels are fed in their [0, 255] scale, they are not normalized
        first_img = np.expand_dims(first_img.astype(np.float32), axis=0)
        second_img = np.expand_dims(second_img.astype(np.float32), axis=0)

        first_result = softmax(self.first_model(first_img, training=False).numpy())
        second_result = softmax(self.second_model(second_img, training=False).numpy())

        prediction = np.zeros((1, 3))
        prediction += first_result
        prediction += second_result

        label = np.argmax(prediction)
        is_real = True if label == 1 else False  # pylint: disable=simplifiable-if-expression
        score = prediction[0][label] / 2

        return is_real, score


# subsdiary functions


def softmax(logits: NDArray[Any]) -> NDArray[Any]:
    """
    Find the class probabilities of given logits
    Args:
        logits (np.ndarray): (n, classes) shaped model output
    Returns:
        probabilities (np.ndarray): (n, classes) shaped probabilities
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return cast(NDArray[Any], exps / np.sum(exps, axis=-1, keepdims=True))
