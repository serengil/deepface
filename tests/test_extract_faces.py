# built-in dependencies
import base64

# 3rd party dependencies
import cv2
import numpy as np
import pytest

# project dependencies
from deepface import DeepFace
from deepface.commons import image_utils
from deepface.commons.logger import Logger

logger = Logger()

detectors = ["opencv", "mtcnn", "ssd"]


def test_different_detectors():
    img_path = "dataset/img11.jpg"
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    for detector in detectors:
        img_objs = DeepFace.extract_faces(img_path=img_path, detector_backend=detector)

        # Check return type for non-batch input
        assert isinstance(img_objs, list) and all(isinstance(obj, dict) for obj in img_objs)

        for img_obj in img_objs:
            assert "face" in img_obj.keys()
            assert "facial_area" in img_obj.keys()
            assert isinstance(img_obj["facial_area"], dict)
            assert "x" in img_obj["facial_area"].keys()
            assert "y" in img_obj["facial_area"].keys()
            assert "w" in img_obj["facial_area"].keys()
            assert "h" in img_obj["facial_area"].keys()
            # is left eye set with respect to the person instead of observer
            assert "left_eye" in img_obj["facial_area"].keys()
            assert "right_eye" in img_obj["facial_area"].keys()
            right_eye = img_obj["facial_area"]["right_eye"]
            left_eye = img_obj["facial_area"]["left_eye"]

            # left eye and right eye must be tuple
            assert isinstance(left_eye, tuple)
            assert isinstance(right_eye, tuple)

            # right eye should be the right eye of the person
            assert left_eye[0] > right_eye[0]

            # left eye and right eye must be int not to have problem in api
            assert isinstance(left_eye[0], int)
            assert isinstance(left_eye[1], int)
            assert isinstance(right_eye[0], int)
            assert isinstance(right_eye[1], int)

            # confidence must be float, not numpy not to have problem in api
            assert "confidence" in img_obj.keys()
            type_conf = type(img_obj["confidence"])
            assert isinstance(
                img_obj["confidence"], float
            ), f"confidence type must be float but it is {type_conf}"

            # we added black pixeled borders to image because if faces are close to border,
            # then alignment moves them to outside of the image. adding this borders may
            # cause to miscalculate the facial area. check it is restored correctly.
            x = img_obj["facial_area"]["x"]
            y = img_obj["facial_area"]["y"]
            w = img_obj["facial_area"]["w"]
            h = img_obj["facial_area"]["h"]

            assert x < width
            assert x + w < width
            assert y < height
            assert y + h < height
            assert left_eye[0] < height
            assert right_eye[0] < height
            assert left_eye[1] < width
            assert right_eye[1] < width

            img = img_obj["face"]
            assert img.shape[0] > 0 and img.shape[1] > 0
        logger.info(f"✅ extract_faces for {detector} backend test is done")


@pytest.mark.parametrize("detector_backend", [
    "opencv",
    "ssd",
    "mtcnn",
    "retinaface",
    "yunet",
    "centerface",
    # optional
    # "yolov11s",
    # "mediapipe",
    # "dlib",
])
def test_batch_extract_faces(detector_backend):
    # Relative tolerance for comparing floating-point values
    rtol = 0.03
    img_paths = [
        "dataset/img2.jpg",
        "dataset/img3.jpg",
        "dataset/img11.jpg",
        "dataset/couple.jpg"
    ]
    expected_num_faces = [1, 1, 1, 2]
    
    # Extract faces one by one
    imgs_objs_individual = [
        DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            align=True,
        ) for img_path in img_paths
    ]

    # Check that individual extraction returns a list of faces
    for img_objs_individual in imgs_objs_individual:
        assert isinstance(img_objs_individual, list)
        assert all(isinstance(face, dict) for face in img_objs_individual)

    # Check that the individual extraction results match the expected number of faces
    for img_objs_individual, expected_faces in zip(imgs_objs_individual, expected_num_faces):
        assert len(img_objs_individual) == expected_faces
    
    # Extract faces in batch
    imgs_objs_batch = DeepFace.extract_faces(
        img_path=img_paths,
        detector_backend=detector_backend,
        align=True,
    )
    
    # Check that the batch extraction returned the expected number of face lists
    assert len(imgs_objs_batch) == len(img_paths)
    
    # Check that each face list has the expected number of faces
    for i, expected_faces in enumerate(expected_num_faces):
        assert len(imgs_objs_batch[i]) == expected_faces

    # Check that the individual extraction results match the batch extraction results
    for img_objs_individual, img_objs_batch in zip(imgs_objs_individual, imgs_objs_batch):
        assert len(img_objs_batch) == len(img_objs_individual), (
            "Batch and individual extraction results should have the same number of detected faces"
        )
        for img_obj_individual, img_obj_batch in zip(img_objs_individual, img_objs_batch):
            for key in img_obj_individual["facial_area"]:
                if isinstance(img_obj_individual["facial_area"][key], tuple):
                    for ind_val, batch_val in zip(
                        img_obj_individual["facial_area"][key], 
                        img_obj_batch["facial_area"][key]
                    ):
                        # Ensure the difference between individual and batch values 
                        # is within rtol% of the individual value
                        assert abs(ind_val - batch_val) <= rtol * ind_val
                elif (
                    isinstance(img_obj_individual["facial_area"][key], int) or 
                    isinstance(img_obj_individual["facial_area"][key], float)
                ):
                    # Ensure the difference between individual and batch values 
                    # is within rtol% of the individual value
                    assert abs(
                        img_obj_individual["facial_area"][key] - 
                        img_obj_batch["facial_area"][key]
                    ) <= rtol * img_obj_individual["facial_area"][key]
            # Ensure the confidence difference is within rtol% of the individual confidence
            assert abs(
                img_obj_individual["confidence"] - 
                img_obj_batch["confidence"]
            ) <= rtol * img_obj_individual["confidence"]
    

@pytest.mark.parametrize("detector_backend", [
    "opencv",
    "ssd",
    "mtcnn",
    "retinaface",
    "yunet",
    # "centerface",
    # optional
    # "yolov11s",
    # "mediapipe",
    # "dlib",
])
def test_batch_extract_faces_with_nparray(detector_backend):
    img_paths = [
        "dataset/img2.jpg",
        "dataset/img3.jpg",
        "dataset/img11.jpg",
        "dataset/couple.jpg"
    ]
    imgs = [
        cv2.resize(image_utils.load_image(img_path)[0], (1920, 1080))
        for img_path in img_paths
    ]
    expected_num_faces = [1, 1, 1, 2]

    # load images as numpy arrays
    imgs_batch = np.stack(imgs, axis=0)

    # extract faces in batch of numpy arrays
    imgs_objs_batch = DeepFace.extract_faces(
        img_path=imgs_batch,
        detector_backend=detector_backend,
        align=True,
        enforce_detection=False,
    )

    # Check return type for batch input
    assert (
        isinstance(imgs_objs_batch, list) and 
        all(
            isinstance(obj, list) and 
            all(isinstance(face, dict) for face in obj) 
            for obj in imgs_objs_batch
        )
    )

    # Check that the batch extraction returned the expected number of face lists
    assert len(imgs_objs_batch) == 4
    for img_objs_batch, img_expected_num_faces in zip(imgs_objs_batch, expected_num_faces):
        assert len(img_objs_batch) == img_expected_num_faces

    # extract faces in batch of paths
    imgs_objs_batch_paths = DeepFace.extract_faces(
        img_path=imgs,
        detector_backend=detector_backend,
        align=True,
        enforce_detection=False,
    )

    # compare results
    for img_objs_batch, img_objs_batch_paths in zip(imgs_objs_batch, imgs_objs_batch_paths):
        assert len(img_objs_batch) == len(img_objs_batch_paths), (
            "Batch and individual extraction results should have the same number of detected faces"
        )


def test_batch_extract_faces_single_image():
    img_path = "dataset/couple.jpg"
    imgs_objs_batch = DeepFace.extract_faces(
        img_path=[img_path],
        align=True,
    )
    assert len(imgs_objs_batch) == 2
    assert [isinstance(obj, dict) for obj in imgs_objs_batch]


def test_backends_for_enforced_detection_with_non_facial_inputs():
    black_img = np.zeros([224, 224, 3])
    for detector in detectors:
        with pytest.raises(ValueError):
            _ = DeepFace.extract_faces(img_path=black_img, detector_backend=detector)
    logger.info("✅ extract_faces for enforced detection and non-facial image test is done")


def test_backends_for_not_enforced_detection_with_non_facial_inputs():
    black_img = np.zeros([224, 224, 3])
    for detector in detectors:
        objs = DeepFace.extract_faces(
            img_path=black_img, detector_backend=detector, enforce_detection=False
        )
        assert objs[0]["face"].shape == (224, 224, 3)
    logger.info("✅ extract_faces for not enforced detection and non-facial image test is done")


def test_file_types_while_loading_base64():
    img1_path = "dataset/img47.jpg"
    img1_base64 = image_to_base64(image_path=img1_path)

    with pytest.raises(ValueError, match="Input image can be jpg or png, but it is"):
        _ = image_utils.load_image_from_base64(uri=img1_base64)

    img2_path = "dataset/img1.jpg"
    img2_base64 = image_to_base64(image_path=img2_path)

    img2 = image_utils.load_image_from_base64(uri=img2_base64)
    # 3 dimensional image should be loaded
    assert len(img2.shape) == 3


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return "data:image/jpeg," + encoded_string


def test_facial_coordinates_are_in_borders():
    detectors = ["retinaface", "mtcnn"]
    expected_faces = [7, 6]

    img_path = "dataset/selfie-many-people.jpg"
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    for i, detector_backend in enumerate(detectors):
        results = DeepFace.extract_faces(img_path=img_path, detector_backend=detector_backend)

        # this is a hard example, mtcnn can detect 6 and retinaface can detect 7 faces
        # be sure all those faces detected. any change in detection module can break this.
        assert len(results) == expected_faces[i]

        for result in results:
            facial_area = result["facial_area"]

            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]

            assert x >= 0
            assert y >= 0
            assert x + w < width
            assert y + h < height

        logger.info(f"✅ facial area coordinates are all in image borders for {detector_backend}")
