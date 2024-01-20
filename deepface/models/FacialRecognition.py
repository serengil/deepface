from abc import ABC
from typing import Any, Union
import numpy as np
import tensorflow as tf

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
if tf_version == 2:
    from tensorflow.keras.models import Model
else:
    from keras.models import Model

# Notice that all facial recognition models must be inherited from this class

# pylint: disable=too-few-public-methods
class FacialRecognition(ABC):
    model: Union[Model, Any]
    model_name: str

    def find_embeddings(self, img: np.ndarray) -> list:
        if not isinstance(self.model, Model):
            raise ValueError(
                "If a facial recognition model is not type of (tf.)keras.models.Model,"
                "Then its find_embeddings method must be implemented its own module."
                f"However {self.model_name}'s model type is {type(self.model)}"
            )
        # model.predict causes memory issue when it is called in a for loop
        # embedding = model.predict(img, verbose=0)[0].tolist()
        return self.model(img, training=False).numpy()[0].tolist()
