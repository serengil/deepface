import os
from typing import List, Union
import numpy as np

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
                "pip install insightface>=0.7.3 onnxruntime>=1.9.0 typing-extensions pydantic albumentations"
            ) from err

        # Define the model filename and subdirectory
        sub_dir = "buffalo_l"
        model_file = "webface_r50.onnx"
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
            source_url="https://drive.google.com/uc?export=download&confirm=pbef&id=1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg"  # pylint: disable=line-too-long
        )

        # Verify the model file exists
        if os.path.exists(weights_path):
            logger.debug(f"Model file found at: {weights_path}")
        else:
            raise FileNotFoundError(f"Model file not found at: {weights_path}")

        # Load the model using the full path
        self.model = get_model(weights_path)
        self.model.prepare(ctx_id=-1, input_size=self.input_shape)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the image or batch of images to match InsightFace recognition model expectations.
        Args:
            img: Image in shape (1, 112, 112, 3) or (112, 112, 3) or batch (batch_size, 112, 112, 3)
        Returns:
            Preprocessed image or batch as numpy array
        """
        if len(img.shape) == 4:  # Batch of images
            preprocessed_imgs = []
            for i in range(img.shape[0]):
                single_img = img[i]
                if single_img.max() <= 1.0:
                    single_img = (single_img * 255).astype(np.uint8)
                single_img = single_img[:, :, ::-1]  # Convert RGB to BGR
                preprocessed_imgs.append(single_img)
            return np.array(preprocessed_imgs)
        if len(img.shape) != 3:
            raise ValueError(
                f"Expected image to be 3D after preprocessing, but got shape: {img.shape}")
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        img = img[:, :, ::-1]  # Convert RGB to BGR
        return img

    def forward(self, img: np.ndarray) -> Union[List[float], List[List[float]]]:
        """
        Extract face embedding from a pre-cropped face image or batch of images.
        Args:
            img: Preprocessed face image with shape (1, 112, 112, 3) or batch (batch_size, 112, 112, 3)
        Returns:
            Face embedding as a list of floats (single image) or list of lists of floats (batch)
        """
        img = self.preprocess(img)
        if len(img.shape) == 4:  # Batch
            embeddings = self.model.get_feat(img)
            return [embedding.tolist() for embedding in embeddings]
        elif len(img.shape) == 3:  # Single image
            embedding = self.model.get_feat(np.expand_dims(img, axis=0))[0]
            return embedding.tolist()
        else:
            raise ValueError(f"Unexpected embedding type after preprocessing: {img.shape}")