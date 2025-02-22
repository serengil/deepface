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
        "InsightFace is an optional dependency for the Buffalo_L model."
        "You can install it with: pip install insightface>=0.7.3"
    ) from err

class Buffalo_L(FacialRecognition):
    def __init__(self):
        self.model = None
        self.input_shape = (112, 112)  # Buffalo_L expects 112x112
        self.output_shape = 512  # Embedding size
        self.load_model()

    def load_model(self):
        """
        Load the InsightFace Buffalo_L recognition model.
        """
        # Use DeepFace's utility to download weights if necessary
        model_rel_path = os.path.join("insightface", "buffalo_l", "w600k_r50.onnx")
        weights_path = weight_utils.download_weights_if_necessary(
            file_name="webface_r50.onnx",
            source_url="https://drive.google.com/uc?export=download&confirm=pbef&id=1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg"
        )
        # Load model from weights folder
        self.model = get_model("buffalo_l/w600k_r50.onnx", root=os.path.dirname(weights_path))
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
        # Always convert RGB to BGR (DeepFace outputs RGB, InsightFace expects BGR)
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