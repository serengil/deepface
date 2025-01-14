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

    def _predict_internal(self, img_batch: np.ndarray) -> np.ndarray:
        """
        Predict for single image or batched images.
        This method uses legacy method while receiving single image as input. 
        And switch to batch prediction if receives batched images.

        Args:
            img_batch:
                Batch of images as np.ndarray (n, x, y, c)
                    with n >= 1, x = image width, y = image height, c = channel
                Or Single image as np.ndarray (1, x, y, c)
                    with x = image width, y = image height and c = channel
                The channel dimension will be 1 if input is grayscale. (For emotion model)
        """
        if not self.model_name: # Check if called from derived class
            raise NotImplementedError("no model selected")
        assert img_batch.ndim == 4, "expected 4-dimensional tensor input"

        if img_batch.shape[0] == 1: # Single image
            # Predict with legacy method.
            return self.model(img_batch, training=False).numpy()[0, :]

        # Batch of images
        # Predict with batch prediction
        return self.model.predict_on_batch(img_batch)

    def _preprocess_batch_or_single_input(
        self,
        img: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:

        """
        Preprocess single or batch of images, return as 4-D numpy array.
        Args:
            img: Single image as np.ndarray (224, 224, 3) or
                 List of images as List[np.ndarray] or
                 Batch of images as np.ndarray (n, 224, 224, 3)
        Returns:
            Four-dimensional numpy array (n, 224, 224, 3)
        """
        image_batch = np.array(img)
        # Remove batch dimension in advance if exists
        image_batch = image_batch.squeeze()
        # Check input dimension
        if len(image_batch.shape) == 3:
            # Single image - add batch dimension
            image_batch = np.expand_dims(image_batch, axis=0)
        return image_batch
