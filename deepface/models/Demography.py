from typing import Union, List
from abc import ABC, abstractmethod
import numpy as np
from deepface.commons import package_utils

tf_version = package_utils.get_tf_major_version()
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
    def predict(self, img: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, np.float64]:
        pass
