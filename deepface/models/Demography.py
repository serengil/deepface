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

    def _preprocess_batch_or_single_input(self, img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:

        """
        Preprocess single or batch of images, return as 4-D numpy array.
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                List of images as List[np.ndarray] or
                Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            Four-dimensional numpy array (n, 224, 224, 3)
        """
        if isinstance(img, list): # Convert from list to image batch.
            image_batch = np.array(img)
        else:
            image_batch = img

        # Remove batch dimension in advance if exists
        image_batch = image_batch.squeeze()

        # Check input dimension
        if len(image_batch.shape) == 3:
            # Single image - add batch dimension
            imgs = np.expand_dims(image_batch, axis=0)

        return image_batch
