# built-in dependencies
from typing import Any, List, Tuple, Union

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.commons.onnx_utils import build_session, run_session
from deepface.models.spoofing.FasNetUtils import crop
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long, too-few-public-methods
FIRST_WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/2.7_80x80_MiniFASNetV2.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/2.7_80x80_MiniFASNetV2.onnx",
]
SECOND_WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/4_0_0_80x80_MiniFASNetV1SE.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/4_0_0_80x80_MiniFASNetV1SE.onnx",
]


class Fasnet:
    """
    Mini Face Anti Spoofing Net Library from repo: github.com/minivision-ai/Silent-Face-Anti-Spoofing

    Minivision's Silent-Face-Anti-Spoofing Repo licensed under Apache License 2.0
    Ref: github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/src/model_lib/MiniFASNet.py
    """

    def __init__(self) -> None:
        # Fasnet will use 2 distinct models to predict, then it will find the sum of predictions
        # to make a final prediction. softmax was baked into both graphs while exporting.
        self.first_model = build_session(
            file_name="2.7_80x80_MiniFASNetV2.onnx", source_url=FIRST_WEIGHTS_URL
        )
        self.second_model = build_session(
            file_name="4_0_0_80x80_MiniFASNetV1SE.onnx", source_url=SECOND_WEIGHTS_URL
        )

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
        first_result = run_session(self.first_model, first_img)
        second_result = run_session(self.second_model, second_img)

        prediction = np.zeros((1, 3))
        prediction += first_result
        prediction += second_result

        label = np.argmax(prediction)
        is_real = True if label == 1 else False  # pylint: disable=simplifiable-if-expression
        score = prediction[0][label] / 2

        return is_real, score
