# built-in dependencies
from typing import List, Union, Optional, IO, cast

# 3rd party dependencies
import cv2
import numpy as np

# project dependencies
from deepface.modules import modeling, detection, preprocessing
from deepface.models.FacialRecognition import FacialRecognition
from deepface.models.Face import Face


def represent(
    img_path: Union[str, np.ndarray, IO[bytes], List[str], List[np.ndarray], List[IO[bytes]]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> List[Face]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (str, np.ndarray, IO[bytes], or Sequence[Union[str, np.ndarray, IO[bytes]]]):
            The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If the source image contains multiple faces,
            the result will include information for each detected face. If a sequence is provided,
            each element should be a string or numpy array representing an image, and the function
            will process images in batch.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet
            (default is VGG-Face.).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        max_faces (int): Set a limit on the number of faces to be processed and returned (default is None).

    Returns:
        List[Face.Face]: A list of Face objects, each containing embedding results for detected faces.
    """

    # batch input
    if (isinstance(img_path, np.ndarray) and img_path.ndim == 4 and img_path.shape[0] > 1) or isinstance(img_path, list):
        batch_faces: List[Face] = []
        # Execute analysis for each image in the batch.
        for single_img in img_path:
            # Call the analyze function for each image in the batch.
            faces = represent(
                img_path=single_img,
                model_name=model_name,
                detector_backend=detector_backend,
                anti_spoofing=anti_spoofing,
                max_faces=max_faces,
            )
            # Append the response to the batch response list.
            batch_faces.extend(faces)
        return batch_faces

    # load model
    model: FacialRecognition = modeling.build_model(task="facial_recognition", model_name=model_name)
    target_size = model.input_shape # expected input size (width, height)

    # extract faces
    img, faces = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces,
    )

    # prepare input for model
    face_imgs: List[np.ndarray] = []
    for face in faces:
        face_img = face.get_face_crop(img, target_size=target_size, aligned=True)
        face_img = face_img[:, :, ::-1] # RGB to BGR
        face_img = preprocessing.convert_to_4D(img=face_img) # convert to 4D for ML models
        face_img = preprocessing.normalize_input(face_img, model_name) # normalize input for the requested model
        face_imgs.append(face_img)

    # forward pass through the model for the entire batch
    embeddings = model.forward(np.concatenate(face_imgs, axis=0)) if len(face_imgs) > 0 else []

    # shady fix since the design ofr the model forward function is awful
    if len(faces) == 1:
        embeddings = [embeddings]

    # add embeddings to face objects
    for idx, face in enumerate(faces):
        embedding = embeddings[idx]
        face.embedding = cast(List[float], embedding)

    return faces
