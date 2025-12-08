# standard library imports
from abc import ABC
from typing import Any, Union, List, Tuple, cast

# third party imports
import numpy as np
from numpy.typing import NDArray

# project imports
from deepface.commons import package_utils

tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    from tensorflow.keras.models import Model
else:
    from keras.models import Model

# Notice that all facial recognition models must be inherited from this class


# pylint: disable=too-few-public-methods
class FacialRecognition(ABC):
    model: Union[Model, Any]
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int

    def forward(self, img: NDArray[Any]) -> Union[List[float], List[List[float]]]:
        if not isinstance(self.model, Model):
            raise ValueError(
                "You must overwrite forward method if it is not a keras model,"
                f"but {self.model_name} not overwritten!"
            )

        # predict expexts e.g. (1, 224, 224, 3) shaped inputs
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        if img.ndim == 4 and img.shape[0] == 1:
            # model.predict causes memory issue when it is called in a for loop
            # embedding = model.predict(img, verbose=0)[0].tolist()
            embeddings = self.model(img, training=False).numpy()
        elif img.ndim == 4 and img.shape[0] > 1:
            embeddings = self.model.predict_on_batch(img)
        else:
            raise ValueError(f"Input image must be (1, X, X, 3) shaped but it is {img.shape}")

        assert isinstance(
            embeddings, np.ndarray
        ), f"Embeddings must be numpy array but it is {type(embeddings)}"

        if embeddings.shape[0] == 1:
            return cast(List[float], embeddings[0].tolist())
        return cast(List[List[float]], embeddings.tolist())
