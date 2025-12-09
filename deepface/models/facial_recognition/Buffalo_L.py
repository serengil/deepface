# built-in dependencies
import os
from typing import List, Union, Any

# third-party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.commons import weight_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.models.FacialRecognition import FacialRecognition

logger = Logger()


class Buffalo_L(FacialRecognition):
    def __init__(self) -> None:
        self.model = None
        self.input_shape = (112, 112)
        self.output_shape = 512
        self.load_model()

    def load_model(self) -> None:
        """Load the InsightFace Buffalo_L recognition model."""
        try:
            from insightface.model_zoo import get_model
        except Exception as err:
            raise ModuleNotFoundError(
                "InsightFace and its dependencies are optional for the Buffalo_L model. "
                "Please install them with: "
                "pip install insightface>=0.7.3 onnxruntime>=1.9.0 typing-extensions pydantic"
                "albumentations"
            ) from err

        sub_dir = "buffalo_l"
        model_file = "webface_r50.onnx"
        model_rel_path = os.path.join(sub_dir, model_file)
        home = folder_utils.get_deepface_home()
        weights_dir = os.path.join(home, ".deepface", "weights")
        buffalo_l_dir = os.path.join(weights_dir, sub_dir)

        if not os.path.exists(buffalo_l_dir):
            os.makedirs(buffalo_l_dir, exist_ok=True)
            logger.info(f"Created directory: {buffalo_l_dir}")

        weights_path = weight_utils.download_weights_if_necessary(
            file_name=model_rel_path,
            source_url="https://drive.google.com/uc?export=download&confirm=pbef&id=1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg",  # pylint: disable=line-too-long
        )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model file not found at: {weights_path}")
        logger.debug(f"Model file found at: {weights_path}")

        self.model = get_model(weights_path)
        self.model.prepare(ctx_id=-1, input_size=self.input_shape)

    def preprocess(self, img: NDArray[Any]) -> NDArray[Any]:
        """
        Preprocess the input image or batch of images.

        Args:
            img: Input image or batch with shape (112, 112, 3)
            or (batch_size, 112, 112, 3).

        Returns:
            Preprocessed image(s) with RGB converted to BGR.
        """
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)  # Convert single image to batch of 1
        elif len(img.shape) != 4:
            raise ValueError(f"Input must be (112, 112, 3) or (X, 112, 112, 3). Got {img.shape}")
        # Convert RGB to BGR for the entire batch
        img = img[:, :, :, ::-1]
        return img

    def forward(self, img: NDArray[Any]) -> Union[List[float], List[List[float]]]:
        """
        Extract facial embedding(s) from the input image or batch of images.

        Args:
            img: Input image or batch with shape (112, 112, 3)
            or (batch_size, 112, 112, 3).

        Returns:
            Embedding as a list of floats (single image)
            or list of lists of floats (batch).
        """
        # Preprocess the input (single image or batch)
        img = self.preprocess(img)
        batch_size = img.shape[0]

        # Handle both single images and batches
        embeddings = []
        for i in range(batch_size):
            embedding = self.model.get_feat(img[i])
            embeddings.append(embedding.flatten().tolist())

        # Return single embedding if batch_size is 1, otherwise return list of embeddings
        return embeddings[0] if batch_size == 1 else embeddings
