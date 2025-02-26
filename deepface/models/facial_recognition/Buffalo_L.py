import os
import numpy as np
from typing import List

try:
    from insightface.model_zoo import get_model
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "InsightFace is an optional dependency for the Buffalo_L model."
        "You can install it with: pip install insightface>=0.7.3"
    ) from err

from deepface.commons import weight_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.models.FacialRecognition import FacialRecognition

logger = Logger()

class Buffalo_L(FacialRecognition):
    def __init__(self):
        self.model = None
        self.input_shape = (112, 112)
        self.output_shape = 512
        self.load_model()

    def load_model(self):
        """
        Load the InsightFace Buffalo_L recognition model.
        """
        # Define the relative model path
        model_rel_path = os.path.join("buffalo_l", "w600k_r50.onnx")

        # Get the DeepFace weights directory
        home = folder_utils.get_deepface_home()
        weights_dir = os.path.join(home, ".deepface", "weights")
        buffalo_l_dir = os.path.join(weights_dir, "buffalo_l")

        # Ensure the buffalo_l subdirectory exists
        if not os.path.exists(buffalo_l_dir):
            os.makedirs(buffalo_l_dir, exist_ok=True)
            logger.info(f"Created directory: {buffalo_l_dir}")

        # Download the model weights
        weights_path = weight_utils.download_weights_if_necessary(
            file_name=model_rel_path,
            source_url="https://drive.google.com/uc?export=download&confirm=pbef&id=1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg"
        )

        # Verify the model file exists
        if os.path.exists(weights_path):
            logger.debug(f"Model file found at: {weights_path}")
        else:
            logger.debug(f"Model file NOT found at: {weights_path}")

        # Load the model using the full absolute path
        self.model = get_model(weights_path)
        self.model.prepare(ctx_id=-1, input_size=self.input_shape)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the image to match InsightFace recognition model expectations.
        Args:
            img: Image in shape (1, 112, 112, 3) or (112, 112, 3)
        Returns:
            Preprocessed image as numpy array
        """
        if len(img.shape) == 4:
            img = img[0]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        img = img[:, :, ::-1]
        return img

    def forward(self, img: np.ndarray) -> List[float]:
        """
        Extract face embedding from a pre-cropped face image.
        Args:
            img: Preprocessed face image with shape (1, 112, 112, 3)
        Returns:
            Face embedding as a list of floats
        """
        img = self.preprocess(img)
        embedding = self.model.get_feat(img)
        
        if isinstance(embedding, np.ndarray) and len(embedding.shape) > 1:
            embedding = embedding.flatten()
        elif isinstance(embedding, list):
            embedding = np.array(embedding).flatten()
        else:
            raise ValueError(f"Unexpected embedding type: {type(embedding)}")
        
        return embedding.tolist()