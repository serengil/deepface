from abc import ABC, abstractmethod
from typing import Any, Union, List
import numpy as np
from deepface.commons import functions

tf_version = functions.get_tf_major_version()
if tf_version == 2:
    from tensorflow.keras.models import Model
else:
    from keras.models import Model

# Notice that all facial recognition models must be inherited from this class

# pylint: disable=too-few-public-methods
class FacialRecognition(ABC):
    model: Union[Model, Any]
    model_name: str

    @abstractmethod
    def find_embeddings(self, img: np.ndarray) -> List[float]:
        pass
