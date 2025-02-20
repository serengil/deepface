import cv2
import numpy as np
from insightface.model_zoo import get_model
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

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
        # Load the recognition model directly (e.g., w600k_r50 from buffalo_l)
        # The buffalo_l package includes recognition model weights
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
        # Ensure image is in uint8 format (0-255 range)
        if img.max() <= 1.0:  # If normalized to [0, 1]
            img = (img * 255).astype(np.uint8)
        # Convert to BGR if in RGB (InsightFace expects BGR)
        if img.shape[2] == 3:
            img = img[:, :, ::-1]  # RGB to BGR
        return img

    def forward(self, img):
        """
        Extract face embedding from a pre-cropped face image.
        Args:
            img (numpy array): Preprocessed face image with shape (1, 112, 112, 3)
        Returns:
            numpy array: Face embedding
        """
        # Preprocess the input image
        img = self.preprocess(img)
        
        # Extract embedding directly (no detection needed)
        embedding = self.model.get_feat(img)
        
        # InsightFace recognition models return a list or array; ensure 1D output
        if isinstance(embedding, (list, np.ndarray)) and len(embedding.shape) > 1:
            embedding = embedding.flatten()
            
        return embedding

    def verify(self, img1, img2, threshold=0.65):
        """
        Verify if two images contain the same person using cosine similarity.
        Args:
            img1, img2 (numpy arrays): Preprocessed images
            threshold (float): Cosine similarity threshold
        Returns:
            tuple: (similarity_score, is_same_person)
        """
        emb1 = self.forward(img1)
        emb2 = self.forward(img2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity, similarity > threshold