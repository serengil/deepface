import os
import numpy as np
from typing import List

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
        try:
            from insightface.model_zoo import get_model
        except Exception as err:
            raise ModuleNotFoundError(
                "InsightFace and its dependencies are optional for the Buffalo_L model. "
                "Please install them with: "
                "pip install insightface>=0.7.3 onnxruntime>=1.9.0 typing-extensions pydantic"
            ) from err

        # Define the model filename and subdirectory
        sub_dir = "buffalo_l"
        model_file = "w600k_r50.onnx"
        model_rel_path = os.path.join(sub_dir, model_file)

        # Get the DeepFace home directory and construct weights path
        home = folder_utils.get_deepface_home()
        weights_dir = os.path.join(home, ".deepface", "weights")
        buffalo_l_dir = os.path.join(weights_dir, sub_dir)

        # Ensure the buffalo_l subdirectory exists
        if not os.path.exists(buffalo_l_dir):
            os.makedirs(buffalo_l_dir, exist_ok=True)
            logger.info(f"Created directory: {buffalo_l_dir}")

        # Download the model weights if not already present
        weights_path = weight_utils.download_weights_if_necessary(
            file_name=model_rel_path,
            source_url="https://drive.google.com/uc?export=download&confirm=pbef&id=1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg"
        )

        # Verify the model file exists
        if os.path.exists(weights_path):
            logger.debug(f"Model file found at: {weights_path}")
        else:
            raise FileNotFoundError(f"Model file not found at: {weights_path}")

        # Load the model
        self.model = get_model(model_file, root=buffalo_l_dir)
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
        if len(img.shape) != 3:
            raise ValueError(f"Expected image to be 3D after preprocessing, but got shape: {img.shape}")
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        img = img[:, :, ::-1]  # Convert RGB to BGR
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