from typing import Union
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model
else:
    from tensorflow.keras.models import Model

# Notice that all facial attribute analysis models must be inherited from this class


# pylint: disable=too-few-public-methods
class Demography(ABC):
    model: Model
    model_name: str

    @abstractmethod
    def predict(self, img: np.ndarray) -> Union[np.ndarray, np.float64]:
        pass
