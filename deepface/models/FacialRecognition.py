from abc import ABC
from typing import Any, Union, List, Tuple
import numpy as np
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

    def forward(self, img: np.ndarray) -> Union[List[float], List[List[float]]]:
        if not isinstance(self.model, Model):
            raise ValueError(
                "You must overwrite forward method if it is not a keras model,"
                f"but {self.model_name} not overwritten!"
            )
        # model.predict causes memory issue when it is called in a for loop
        # embedding = model.predict(img, verbose=0)[0].tolist()
        if img.shape == 4 and img.shape[0] == 1:
            img = img[0]
        embeddings = self.model(img, training=False).numpy()
        if embeddings.shape[0] == 1:
            return embeddings[0].tolist()
        return embeddings.tolist()
