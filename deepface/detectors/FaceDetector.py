from PIL import Image
import numpy as np
from deepface.detectors import (
    OpenCvWrapper,
    SsdWrapper,
    DlibWrapper,
    MtcnnWrapper,
    RetinaFaceWrapper,
    MediapipeWrapper,
    YoloWrapper,
    YunetWrapper,
)


def build_model(detector_backend):
    global face_detector_obj  # singleton design pattern

    backends = {
        "opencv": OpenCvWrapper.build_model,
        "ssd": SsdWrapper.build_model,
        "dlib": DlibWrapper.build_model,
        "mtcnn": MtcnnWrapper.build_model,
        "retinaface": RetinaFaceWrapper.build_model,
        "mediapipe": MediapipeWrapper.build_model,
        "yolov8": YoloWrapper.build_model,
        "yunet": YunetWrapper.build_model,
    }

    if not "face_detector_obj" in globals():
        face_detector_obj = {}

    built_models = list(face_detector_obj.keys())
    if detector_backend not in built_models:
        face_detector = backends.get(detector_backend)

        if face_detector:
            face_detector = face_detector()
            face_detector_obj[detector_backend] = face_detector
        else:
            raise ValueError("invalid detector_backend passed - " + detector_backend)

    return face_detector_obj[detector_backend]


def detect_face(face_detector, detector_backend, img, align=True):
    obj = detect_faces(face_detector, detector_backend, img, align)

    if len(obj) > 0:
        face, region, confidence = obj[0]  # discard multiple faces

    # If no face is detected, set face to None,
    # image region to full image, and confidence to 0.
    else:  # len(obj) == 0
        face = None
        region = [0, 0, img.shape[1], img.shape[0]]
        confidence = 0

    return face, region, confidence


def detect_faces(face_detector, detector_backend, img, align=True):
    backends = {
        "opencv": OpenCvWrapper.detect_face,
        "ssd": SsdWrapper.detect_face,
        "dlib": DlibWrapper.detect_face,
        "mtcnn": MtcnnWrapper.detect_face,
        "retinaface": RetinaFaceWrapper.detect_face,
        "mediapipe": MediapipeWrapper.detect_face,
        "yolov8": YoloWrapper.detect_face,
        "yunet": YunetWrapper.detect_face,
    }

    detect_face_fn = backends.get(detector_backend)

    if detect_face_fn:  # pylint: disable=no-else-return
        obj = detect_face_fn(face_detector, img, align)
        # obj stores list of (detected_face, region, confidence)
        return obj
    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)


def get_alignment_angle_arctan2(left_eye, right_eye):
    """
    The left_eye is the eye to the left of the viewer,
    i.e., right eye of the person in the image.
    The top-left point of the frame is (0, 0).
    """
    return float(np.degrees(
        np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    ))


def alignment_procedure(img, left_eye, right_eye):
    angle = get_alignment_angle_arctan2(left_eye, right_eye)
    img = Image.fromarray(img)
    img = np.array(img.rotate(angle))
    return img
