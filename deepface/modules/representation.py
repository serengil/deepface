# built-in dependencies
from typing import Any, Dict, List, Union

# 3rd party dependencies
import numpy as np
import cv2

# project dependencies
from deepface.modules import modeling
from deepface.commons import functions
from deepface.models.FacialRecognition import FacialRecognition


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    normalization: str = "base",
) -> List[Dict[str, Any]]:
    """
    This function represents facial images as vectors. The function uses convolutional neural
    networks models to generate vector embeddings.

    Parameters:
            img_path (string): exact image path. Alternatively, numpy array (BGR) or based64
            encoded images could be passed. Source image can have many faces. Then, result will
            be the size of number of faces appearing in the source image.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
            ArcFace, SFace

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib, mediapipe or yolov8. A special value `skip` could be used to skip face-detection
            and only encode the given image.

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Represent function returns a list of object, each object has fields as follows:
            {
                // Multidimensional vector
                // The number of dimensions is changing based on the reference model.
                // E.g. FaceNet returns 128 dimensional vector;
                //      VGG-Face returns 2622 dimensional vector.
                "embedding": np.array,

                // Detected Facial-Area by Face detection in dict format.
                // (x, y) is left-corner point, and (w, h) is the width and height
                // If `detector_backend` == `skip`, it is the full image area and nonsense.
                "facial_area": dict{"x": int, "y": int, "w": int, "h": int},

                // Face detection confidence.
                // If `detector_backend` == `skip`, will be 0 and nonsense.
                "face_confidence": float
            }
    """
    resp_objs = []

    model: FacialRecognition = modeling.build_model(model_name)

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = functions.find_target_size(model_name=model_name)
    if detector_backend != "skip":
        img_objs = functions.extract_faces(
            img=img_path,
            target_size=(target_size[1], target_size[0]),
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )
    else:  # skip
        # Try load. If load error, will raise exception internal
        img, _ = functions.load_image(img_path)
        # --------------------------------
        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
            # when called from verify, this is already normalized. But needed when user given.
            if img.max() > 1:
                img = (img.astype(np.float32) / 255.0).astype(np.float32)
        # --------------------------------
        # make dummy region and confidence to keep compatibility with `extract_faces`
        img_region = {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[2]}
        img_objs = [(img, img_region, 0)]
    # ---------------------------------

    for img, region, confidence in img_objs:
        # custom normalization
        img = functions.normalize_input(img=img, normalization=normalization)

        embedding = model.find_embeddings(img)

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs
