# built-in dependencies
from typing import Union, List, Any, cast
from abc import ABC, abstractmethod

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
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
    def predict(
        self, img: Union[NDArray[Any], List[NDArray[Any]]]
    ) -> Union[NDArray[Any], np.float64]:
        pass

    def _predict_internal(self, img_batch: NDArray[Any]) -> NDArray[Any]:
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
        if not self.model_name:  # Check if called from derived class
            raise NotImplementedError("no model selected")
        assert img_batch.ndim == 4, "expected 4-dimensional tensor input"

        if img_batch.shape[0] == 1:  # Single image
            # Predict with legacy method.
            return cast(NDArray[Any], self.model(img_batch, training=False).numpy()[0, :])

        # Batch of images
        # Predict with batch prediction
        return cast(NDArray[Any], self.model.predict_on_batch(img_batch))

    def _preprocess_batch_or_single_input(
        self, img: Union[NDArray[Any], List[NDArray[Any]]]
    ) -> NDArray[Any]:
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

        # Check input dimension
        if len(image_batch.shape) == 3:
            # Single image - add batch dimension
            image_batch = np.expand_dims(image_batch, axis=0)
        return image_batch
