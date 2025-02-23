import os
import numpy as np
from deepface.commons import weight_utils
from deepface.commons.logger import Logger
from deepface.models.FacialRecognition import FacialRecognition

logger = Logger()

try:
    from insightface.model_zoo import get_model
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "InsightFace is an optional dependency for the Buffalo_L model. "
        "You can install it with: pip install insightface>=0.7.3"
    ) from err

class Buffalo_L(FacialRecognition):
    def __init__(self):
        self.model = None
        self.input_shape = (112, 112)  # Buffalo_L expects 112x112
        self.output_shape = 512        # Embedding size
        self.load_model()

    def load_model(self):
        """
        Load the InsightFace Buffalo_L recognition model.
        """
        # Define the model filename and subdirectory
        sub_dir = "buffalo_l"
        model_file = "w600k_r50.onnx"
        model_rel_path = os.path.join(sub_dir, model_file)

        # Define the weights directory and ensure the buffalo_l subdirectory exists
        weights_dir = os.path.join(os.path.expanduser("~"), ".deepface", "weights")
        buffalo_l_dir = os.path.join(weights_dir, sub_dir)
        if not os.path.exists(buffalo_l_dir):
            os.makedirs(buffalo_l_dir, exist_ok=True)
            print("Created directory:", buffalo_l_dir)

        # Download the model weights if not already present
        weights_path = weight_utils.download_weights_if_necessary(
            file_name=model_rel_path,
            source_url="https://drive.google.com/uc?export=download&confirm=pbef&id=1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg"
        )
        print("Downloaded model path:", weights_path)

        # Verify that the model file exists at the expected location
        expected_model_path = os.path.join(buffalo_l_dir, model_file)
        if os.path.exists(expected_model_path):
            print("Model file found at expected location:", expected_model_path)
        else:
            print("Model file NOT found at expected location:", expected_model_path)

        # Use the full absolute path for loading the model
        full_model_path = os.path.join(buffalo_l_dir, model_file)
        print("Full model path:", full_model_path)
        self.model = get_model(full_model_path)
        self.model.prepare(ctx_id=-1, input_size=self.input_shape)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the image to match InsightFace recognition model expectations.
        Args:
            img: Image in shape (1, 112, 112, 3) or (112, 112, 3)
        Returns:
            Preprocessed image as numpy array
        """
        if len(img.shape) == 4:  # (1, 112, 112, 3)
            img = img[0]  # Remove batch dimension
        if img.max() <= 1.0:  # If normalized to [0, 1]
            img = (img * 255).astype(np.uint8)
        # Convert RGB to BGR (DeepFace outputs RGB, InsightFace expects BGR)
        img = img[:, :, ::-1]
        return img

    def forward(self, img: np.ndarray) -> list[float]:
        """
        Extract face embedding from a pre-cropped face image.
        Args:
            img: Preprocessed face image with shape (1, 112, 112, 3)
        Returns:
            Face embedding as a list of floats
        """
        img = self.preprocess(img)
        embedding = self.model.get_feat(img)
        
        # Handle different embedding formats
        if isinstance(embedding, np.ndarray) and len(embedding.shape) > 1:
            embedding = embedding.flatten()
        elif isinstance(embedding, list):
            embedding = np.array(embedding).flatten()
        else:
            raise ValueError(f"Unexpected embedding type: {type(embedding)}")
        
        return embedding.tolist()  # Convert to list per FacialRecognition spec