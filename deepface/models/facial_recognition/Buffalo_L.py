import cv2
import numpy as np
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# Check for insightface dependency
try:
    from insightface.model_zoo import get_model
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "InsightFace is an optional dependency for the Buffalo_L model."
        "You can install it with: pip install insightface>=0.7.3"
    )

class Buffalo_L(FacialRecognition):
    def __init__(self):
        self.model = None
        self.input_shape = (112, 112)  # Buffalo_L recognition model expects 112x112
        self.output_shape = 512  # Embedding size for Buffalo_L
        self.load_model()

    def load_model(self):
        """
        Load the InsightFace Buffalo_L recognition model.
        """
        self.model = get_model('buffalo_l/w600k_r50.onnx', download=True)
        self.model.prepare(ctx_id=-1, input_size=self.input_shape)  # ctx_id=-1 for CPU

    def preprocess(self, img):
        """
        Preprocess the image to match InsightFace recognition model expectations.
        Args:
            img (numpy array): Image in shape (1, 112, 112, 3) or (112, 112, 3)
        Returns:
            numpy array: Preprocessed image
        """
        if len(img.shape) == 4:  # (1, 112, 112, 3)
            img = img[0]  # Remove batch dimension
        if img.max() <= 1.0:  # If normalized to [0, 1]
            img = (img * 255).astype(np.uint8)
        # Always convert RGB to BGR (InsightFace expects BGR, DeepFace provides RGB)
        img = img[:, :, ::-1]
        return img

    def forward(self, img):
        """
        Extract face embedding from a pre-cropped face image.
        Args:
            img (numpy array): Preprocessed face image with shape (1, 112, 112, 3)
        Returns:
            numpy array: Face embedding
        """
        img = self.preprocess(img)
        embedding = self.model.get_feat(img)
        
        # Handle different embedding formats
        if isinstance(embedding, np.ndarray):
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()
        elif isinstance(embedding, list):
            embedding = np.array(embedding).flatten()
            
        return embedding