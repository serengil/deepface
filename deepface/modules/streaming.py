# built-in dependencies
import os
import time
from typing import List, Tuple, Optional

# 3rd party dependencies
import numpy as np
import pandas as pd
import cv2

# project dependencies
from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger(module="commons.realtime")

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


IDENTIFIED_IMG_SIZE = 112
TEXT_COLOR = (255, 255, 255)


def analysis(
    db_path: str,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=5,
    frame_threshold=5,
):
    """
    Run real time face recognition and facial attribute analysis

    Args:
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        enable_face_analysis (bool): Flag to enable face analysis (default is True).

        source (Any): The source for the video stream (default is 0, which represents the
            default camera).

        time_threshold (int): The time threshold (in seconds) for face recognition (default is 5).

        frame_threshold (int): The frame threshold for face recognition (default is 5).
    Returns:
        None
    """
    # initialize models
    build_demography_models(enable_face_analysis=enable_face_analysis)
    target_size = build_facial_recognition_model(model_name=model_name)
    # call a dummy find function for db_path once to create embeddings before starting webcam
    _ = search_identity(
        detected_face=np.zeros([224, 224, 3]),
        db_path=db_path,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        model_name=model_name,
    )

    freezed_img = None
    freeze = False
    num_frames_with_faces = 0
    tic = time.time()

    cap = cv2.VideoCapture(source)  # webcam
    while True:
        has_frame, img = cap.read()
        if not has_frame:
            break

        # we are adding some figures into img such as identified facial image, age, gender
        # that is why, we need raw image itself to make analysis
        raw_img = img.copy()

        faces_coordinates = []
        if freeze is False:
            faces_coordinates = grab_facial_areas(
                img=img, detector_backend=detector_backend, target_size=target_size
            )

            # we will pass img to analyze modules (identity, demography) and add some illustrations
            # that is why, we will not be able to extract detected face from img clearly
            detected_faces = extract_facial_areas(img=img, faces_coordinates=faces_coordinates)

            img = highlight_facial_areas(img=img, faces_coordinates=faces_coordinates)
            img = countdown_to_freeze(
                img=img,
                faces_coordinates=faces_coordinates,
                frame_threshold=frame_threshold,
                num_frames_with_faces=num_frames_with_faces,
            )

            num_frames_with_faces = num_frames_with_faces + 1 if len(faces_coordinates) else 0

            freeze = num_frames_with_faces > 0 and num_frames_with_faces % frame_threshold == 0
            if freeze:
                # add analyze results into img - derive from raw_img
                img = highlight_facial_areas(img=raw_img, faces_coordinates=faces_coordinates)

                # age, gender and emotion analysis
                img = perform_demography_analysis(
                    enable_face_analysis=enable_face_analysis,
                    img=raw_img,
                    faces_coordinates=faces_coordinates,
                    detected_faces=detected_faces,
                )
                # facial recogntion analysis
                img = perform_facial_recognition(
                    img=img,
                    faces_coordinates=faces_coordinates,
                    detected_faces=detected_faces,
                    db_path=db_path,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    model_name=model_name,
                )

                # freeze the img after analysis
                freezed_img = img.copy()

                # start counter for freezing
                tic = time.time()
                logger.info("freezed")

        elif freeze is True and time.time() - tic > time_threshold:
            freeze = False
            freezed_img = None
            # reset counter for freezing
            tic = time.time()
            logger.info("freeze released")

        freezed_img = countdown_to_release(img=freezed_img, tic=tic, time_threshold=time_threshold)

        cv2.imshow("img", img if freezed_img is None else freezed_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()


def build_facial_recognition_model(model_name: str) -> tuple:
    """
    Build facial recognition model
    Args:
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).
    Returns
        input_shape (tuple): input shape of given facial recognitio n model.
    """
    model: FacialRecognition = DeepFace.build_model(model_name=model_name)
    logger.info(f"{model_name} is built")
    return model.input_shape


def search_identity(
    detected_face: np.ndarray,
    db_path: str,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Search an identity in facial database.
    Args:
        detected_face (np.ndarray): extracted individual facial image
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).
        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).
    Returns:
        result (tuple): result consisting of following objects
            identified image path (str)
            identified image itself (np.ndarray)
    """
    target_path = None
    try:
        dfs = DeepFace.find(
            img_path=detected_face,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=False,
            silent=True,
        )
    except ValueError as err:
        if f"No item found in {db_path}" in str(err):
            logger.warn(
                f"No item is found in {db_path}."
                "So, no facial recognition analysis will be performed."
            )
            dfs = []
        else:
            raise err
    if len(dfs) == 0:
        # you may consider to return unknown person's image here
        return None, None

    # detected face is coming from parent, safe to access 1st index
    df = dfs[0]

    if df.shape[0] == 0:
        return None, None

    candidate = df.iloc[0]
    target_path = candidate["identity"]
    logger.info(f"Hello, {target_path}")

    # load found identity image - extracted if possible
    target_objs = DeepFace.extract_faces(
        img_path=target_path,
        target_size=(IDENTIFIED_IMG_SIZE, IDENTIFIED_IMG_SIZE),
        detector_backend=detector_backend,
        enforce_detection=False,
        align=True,
    )

    # extract facial area of the identified image if and only if it has one face
    # otherwise, show image as is
    if len(target_objs) == 1:
        # extract 1st item directly
        target_obj = target_objs[0]
        target_img = target_obj["face"]
        target_img *= 255
        target_img = target_img[:, :, ::-1]
    else:
        target_img = cv2.imread(target_path)

    return target_path.split("/")[-1], target_img


def build_demography_models(enable_face_analysis: bool) -> None:
    """
    Build demography analysis models
    Args:
        enable_face_analysis (bool): Flag to enable face analysis (default is True).
    Returns:
        None
    """
    if enable_face_analysis is False:
        return
    DeepFace.build_model(model_name="Age")
    logger.info("Age model is just built")
    DeepFace.build_model(model_name="Gender")
    logger.info("Gender model is just built")
    DeepFace.build_model(model_name="Emotion")
    logger.info("Emotion model is just built")


def highlight_facial_areas(
    img: np.ndarray, faces_coordinates: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Highlight detected faces with rectangles in the given image
    Args:
        img (np.ndarray): image itself
        faces_coordinates (list): list of face coordinates as tuple with x, y, w and h
    Returns:
        img (np.ndarray): image with highlighted facial areas
    """
    for x, y, w, h in faces_coordinates:
        # highlight facial area with rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (67, 67, 67), 1)
    return img


def countdown_to_freeze(
    img: np.ndarray,
    faces_coordinates: List[Tuple[int, int, int, int]],
    frame_threshold: int,
    num_frames_with_faces: int,
) -> np.ndarray:
    """
    Highlight time to freeze in the image's facial areas
    Args:
        img (np.ndarray): image itself
        faces_coordinates (list): list of face coordinates as tuple with x, y, w and h
        frame_threshold (int): how many sequantial frames required with face(s) to freeze
        num_frames_with_faces (int): how many sequantial frames do we have with face(s)
    Returns:
        img (np.ndarray): image with counter values
    """
    for x, y, w, h in faces_coordinates:
        cv2.putText(
            img,
            str(frame_threshold - (num_frames_with_faces % frame_threshold)),
            (int(x + w / 4), int(y + h / 1.5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (255, 255, 255),
            2,
        )
    return img


def countdown_to_release(
    img: Optional[np.ndarray], tic: float, time_threshold: int
) -> Optional[np.ndarray]:
    """
    Highlight time to release the freezing in the image top left area
    Args:
        img (np.ndarray): image itself
        tic (float): time specifying when freezing started
        time_threshold (int): freeze time threshold
    Returns:
        img (np.ndarray): image with time to release the freezing
    """
    # do not take any action if it is not frozen yet
    if img is None:
        return img
    toc = time.time()
    time_left = int(time_threshold - (toc - tic) + 1)
    cv2.rectangle(img, (10, 10), (90, 50), (67, 67, 67), -10)
    cv2.putText(
        img,
        str(time_left),
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        1,
    )
    return img


def grab_facial_areas(
    img: np.ndarray, detector_backend: str, target_size: Tuple[int, int], threshold: int = 130
) -> List[Tuple[int, int, int, int]]:
    """
    Find facial area coordinates in the given image
    Args:
        img (np.ndarray): image itself
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).
        target_size (tuple): input shape of the facial recognition model.
        threshold (int): threshold for facial area, discard smaller ones
    Returns
        result (list): list of tuple with x, y, w and h coordinates
    """
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img,
            detector_backend=detector_backend,
            target_size=target_size,
            # you may consider to extract with larger expanding value
            expand_percentage=0,
        )
        faces = [
            (
                face_obj["facial_area"]["x"],
                face_obj["facial_area"]["y"],
                face_obj["facial_area"]["w"],
                face_obj["facial_area"]["h"],
            )
            for face_obj in face_objs
            if face_obj["facial_area"]["w"] > threshold
        ]
        return faces
    except:  # to avoid exception if no face detected
        return []


def extract_facial_areas(
    img: np.ndarray, faces_coordinates: List[Tuple[int, int, int, int]]
) -> List[np.ndarray]:
    """
    Extract facial areas as numpy array from given image
    Args:
        img (np.ndarray): image itself
        faces_coordinates (list): list of facial area coordinates as tuple with
            x, y, w and h values
    Returns:
        detected_faces (list): list of detected facial area images
    """
    detected_faces = []
    for x, y, w, h in faces_coordinates:
        detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
        detected_faces.append(detected_face)
    return detected_faces


def perform_facial_recognition(
    img: np.ndarray,
    detected_faces: List[np.ndarray],
    faces_coordinates: List[Tuple[int, int, int, int]],
    db_path: str,
    detector_backend: str,
    distance_metric: str,
    model_name: str,
) -> np.ndarray:
    """
    Perform facial recognition
    Args:
        img (np.ndarray): image itself
        detected_faces (list): list of extracted detected face images as numpy
        faces_coordinates (list): list of facial area coordinates as tuple with
            x, y, w and h values
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8' (default is opencv).
        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace (default is VGG-Face).
    Returns:
        img (np.ndarray): image with identified face informations
    """
    for idx, (x, y, w, h) in enumerate(faces_coordinates):
        detected_face = detected_faces[idx]
        target_label, target_img = search_identity(
            detected_face=detected_face,
            db_path=db_path,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            model_name=model_name,
        )
        if target_label is None:
            continue

        img = overlay_identified_face(
            img=img,
            target_img=target_img,
            label=target_label,
            x=x,
            y=y,
            w=w,
            h=h,
        )

    return img


def perform_demography_analysis(
    enable_face_analysis: bool,
    img: np.ndarray,
    faces_coordinates: List[Tuple[int, int, int, int]],
    detected_faces: List[np.ndarray],
) -> np.ndarray:
    """
    Perform demography analysis on given image
    Args:
        enable_face_analysis (bool): Flag to enable face analysis.
        img (np.ndarray): image itself
        faces_coordinates (list): list of face coordinates as tuple with
            x, y, w and h values
        detected_faces (list): list of extracted detected face images as numpy
    Returns:
        img (np.ndarray): image with analyzed demography information
    """
    if enable_face_analysis is False:
        return img
    for idx, (x, y, w, h) in enumerate(faces_coordinates):
        detected_face = detected_faces[idx]
        demographies = DeepFace.analyze(
            img_path=detected_face,
            actions=("age", "gender", "emotion"),
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )

        if len(demographies) == 0:
            continue

        # safe to access 1st index because detector backend is skip
        demography = demographies[0]

        img = overlay_emotion(img=img, emotion_probas=demography["emotion"], x=x, y=y, w=w, h=h)
        img = overlay_age_gender(
            img=img,
            apparent_age=demography["age"],
            gender=demography["dominant_gender"][0:1],  # M or W
            x=x,
            y=y,
            w=w,
            h=h,
        )
    return img


def overlay_identified_face(
    img: np.ndarray,
    target_img: np.ndarray,
    label: str,
    x: int,
    y: int,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Overlay the identified face onto image itself
    Args:
        img (np.ndarray): image itself
        target_img (np.ndarray): identified face's image
        label (str): name of the identified face
        x (int): x coordinate of the face on the given image
        y (int): y coordinate of the face on the given image
        w (int): w coordinate of the face on the given image
        h (int): h coordinate of the face on the given image
    Returns:
        img (np.ndarray): image with overlayed identity
    """
    try:
        if y - IDENTIFIED_IMG_SIZE > 0 and x + w + IDENTIFIED_IMG_SIZE < img.shape[1]:
            # top right
            img[
                y - IDENTIFIED_IMG_SIZE : y,
                x + w : x + w + IDENTIFIED_IMG_SIZE,
            ] = target_img

            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(
                img,
                (x + w, y),
                (x + w + IDENTIFIED_IMG_SIZE, y + 20),
                (46, 200, 255),
                cv2.FILLED,
            )
            cv2.addWeighted(
                overlay,
                opacity,
                img,
                1 - opacity,
                0,
                img,
            )

            cv2.putText(
                img,
                label,
                (x + w, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
            )

            # connect face and text
            cv2.line(
                img,
                (x + int(w / 2), y),
                (x + 3 * int(w / 4), y - int(IDENTIFIED_IMG_SIZE / 2)),
                (67, 67, 67),
                1,
            )
            cv2.line(
                img,
                (x + 3 * int(w / 4), y - int(IDENTIFIED_IMG_SIZE / 2)),
                (x + w, y - int(IDENTIFIED_IMG_SIZE / 2)),
                (67, 67, 67),
                1,
            )

        elif y + h + IDENTIFIED_IMG_SIZE < img.shape[0] and x - IDENTIFIED_IMG_SIZE > 0:
            # bottom left
            img[
                y + h : y + h + IDENTIFIED_IMG_SIZE,
                x - IDENTIFIED_IMG_SIZE : x,
            ] = target_img

            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(
                img,
                (x - IDENTIFIED_IMG_SIZE, y + h - 20),
                (x, y + h),
                (46, 200, 255),
                cv2.FILLED,
            )
            cv2.addWeighted(
                overlay,
                opacity,
                img,
                1 - opacity,
                0,
                img,
            )

            cv2.putText(
                img,
                label,
                (x - IDENTIFIED_IMG_SIZE, y + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
            )

            # connect face and text
            cv2.line(
                img,
                (x + int(w / 2), y + h),
                (
                    x + int(w / 2) - int(w / 4),
                    y + h + int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (67, 67, 67),
                1,
            )
            cv2.line(
                img,
                (
                    x + int(w / 2) - int(w / 4),
                    y + h + int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (x, y + h + int(IDENTIFIED_IMG_SIZE / 2)),
                (67, 67, 67),
                1,
            )

        elif y - IDENTIFIED_IMG_SIZE > 0 and x - IDENTIFIED_IMG_SIZE > 0:
            # top left
            img[y - IDENTIFIED_IMG_SIZE : y, x - IDENTIFIED_IMG_SIZE : x] = target_img

            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(
                img,
                (x - IDENTIFIED_IMG_SIZE, y),
                (x, y + 20),
                (46, 200, 255),
                cv2.FILLED,
            )
            cv2.addWeighted(
                overlay,
                opacity,
                img,
                1 - opacity,
                0,
                img,
            )

            cv2.putText(
                img,
                label,
                (x - IDENTIFIED_IMG_SIZE, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
            )

            # connect face and text
            cv2.line(
                img,
                (x + int(w / 2), y),
                (
                    x + int(w / 2) - int(w / 4),
                    y - int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (67, 67, 67),
                1,
            )
            cv2.line(
                img,
                (
                    x + int(w / 2) - int(w / 4),
                    y - int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (x, y - int(IDENTIFIED_IMG_SIZE / 2)),
                (67, 67, 67),
                1,
            )

        elif (
            x + w + IDENTIFIED_IMG_SIZE < img.shape[1]
            and y + h + IDENTIFIED_IMG_SIZE < img.shape[0]
        ):
            # bottom righ
            img[
                y + h : y + h + IDENTIFIED_IMG_SIZE,
                x + w : x + w + IDENTIFIED_IMG_SIZE,
            ] = target_img

            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(
                img,
                (x + w, y + h - 20),
                (x + w + IDENTIFIED_IMG_SIZE, y + h),
                (46, 200, 255),
                cv2.FILLED,
            )
            cv2.addWeighted(
                overlay,
                opacity,
                img,
                1 - opacity,
                0,
                img,
            )

            cv2.putText(
                img,
                label,
                (x + w, y + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
            )

            # connect face and text
            cv2.line(
                img,
                (x + int(w / 2), y + h),
                (
                    x + int(w / 2) + int(w / 4),
                    y + h + int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (67, 67, 67),
                1,
            )
            cv2.line(
                img,
                (
                    x + int(w / 2) + int(w / 4),
                    y + h + int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (x + w, y + h + int(IDENTIFIED_IMG_SIZE / 2)),
                (67, 67, 67),
                1,
            )
        else:
            logger.info("cannot put facial recognition info on the image")
    except Exception as err:  # pylint: disable=broad-except
        logger.error(str(err))
    return img


def overlay_emotion(
    img: np.ndarray, emotion_probas: dict, x: int, y: int, w: int, h: int
) -> np.ndarray:
    """
    Overlay the analyzed emotion of face onto image itself
    Args:
        img (np.ndarray): image itself
        emotion_probas (dict): probability of different emotionas dictionary
        x (int): x coordinate of the face on the given image
        y (int): y coordinate of the face on the given image
        w (int): w coordinate of the face on the given image
        h (int): h coordinate of the face on the given image
    Returns:
        img (np.ndarray): image with overlay emotion analsis results
    """
    emotion_df = pd.DataFrame(emotion_probas.items(), columns=["emotion", "score"])
    emotion_df = emotion_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)

    # background of mood box

    # transparency
    overlay = img.copy()
    opacity = 0.4

    # put gray background to the right of the detected image
    if x + w + IDENTIFIED_IMG_SIZE < img.shape[1]:
        cv2.rectangle(
            img,
            (x + w, y),
            (x + w + IDENTIFIED_IMG_SIZE, y + h),
            (64, 64, 64),
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # put gray background to the left of the detected image
    elif x - IDENTIFIED_IMG_SIZE > 0:
        cv2.rectangle(
            img,
            (x - IDENTIFIED_IMG_SIZE, y),
            (x, y + h),
            (64, 64, 64),
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    for index, instance in emotion_df.iterrows():
        current_emotion = instance["emotion"]
        emotion_label = f"{current_emotion} "
        emotion_score = instance["score"] / 100

        filled_bar_x = 35  # this is the size if an emotion is 100%
        bar_x = int(filled_bar_x * emotion_score)

        if x + w + IDENTIFIED_IMG_SIZE < img.shape[1]:

            text_location_y = y + 20 + (index + 1) * 20
            text_location_x = x + w

            if text_location_y < y + h:
                cv2.putText(
                    img,
                    emotion_label,
                    (text_location_x, text_location_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.rectangle(
                    img,
                    (x + w + 70, y + 13 + (index + 1) * 20),
                    (
                        x + w + 70 + bar_x,
                        y + 13 + (index + 1) * 20 + 5,
                    ),
                    (255, 255, 255),
                    cv2.FILLED,
                )

        elif x - IDENTIFIED_IMG_SIZE > 0:

            text_location_y = y + 20 + (index + 1) * 20
            text_location_x = x - IDENTIFIED_IMG_SIZE

            if text_location_y <= y + h:
                cv2.putText(
                    img,
                    emotion_label,
                    (text_location_x, text_location_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.rectangle(
                    img,
                    (
                        x - IDENTIFIED_IMG_SIZE + 70,
                        y + 13 + (index + 1) * 20,
                    ),
                    (
                        x - IDENTIFIED_IMG_SIZE + 70 + bar_x,
                        y + 13 + (index + 1) * 20 + 5,
                    ),
                    (255, 255, 255),
                    cv2.FILLED,
                )

    return img


def overlay_age_gender(
    img: np.ndarray, apparent_age: float, gender: str, x: int, y: int, w: int, h: int
) -> np.ndarray:
    """
    Overlay the analyzed age and gender of face onto image itself
    Args:
        img (np.ndarray): image itself
        apparent_age (float): analyzed apparent age
        gender (str): analyzed gender
        x (int): x coordinate of the face on the given image
        y (int): y coordinate of the face on the given image
        w (int): w coordinate of the face on the given image
        h (int): h coordinate of the face on the given image
    Returns:
        img (np.ndarray): image with overlay age and gender analsis results
    """
    logger.debug(f"{apparent_age} years old {gender}")
    analysis_report = f"{int(apparent_age)} {gender}"

    info_box_color = (46, 200, 255)

    # show its age and gender on the top of the image
    if y - IDENTIFIED_IMG_SIZE + int(IDENTIFIED_IMG_SIZE / 5) > 0:

        triangle_coordinates = np.array(
            [
                (x + int(w / 2), y),
                (
                    x + int(w / 2) - int(w / 10),
                    y - int(IDENTIFIED_IMG_SIZE / 3),
                ),
                (
                    x + int(w / 2) + int(w / 10),
                    y - int(IDENTIFIED_IMG_SIZE / 3),
                ),
            ]
        )

        cv2.drawContours(
            img,
            [triangle_coordinates],
            0,
            info_box_color,
            -1,
        )

        cv2.rectangle(
            img,
            (
                x + int(w / 5),
                y - IDENTIFIED_IMG_SIZE + int(IDENTIFIED_IMG_SIZE / 5),
            ),
            (x + w - int(w / 5), y - int(IDENTIFIED_IMG_SIZE / 3)),
            info_box_color,
            cv2.FILLED,
        )

        cv2.putText(
            img,
            analysis_report,
            (x + int(w / 3.5), y - int(IDENTIFIED_IMG_SIZE / 2.1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 111, 255),
            2,
        )

    # show its age and gender on the top of the image
    elif y + h + IDENTIFIED_IMG_SIZE - int(IDENTIFIED_IMG_SIZE / 5) < img.shape[0]:

        triangle_coordinates = np.array(
            [
                (x + int(w / 2), y + h),
                (
                    x + int(w / 2) - int(w / 10),
                    y + h + int(IDENTIFIED_IMG_SIZE / 3),
                ),
                (
                    x + int(w / 2) + int(w / 10),
                    y + h + int(IDENTIFIED_IMG_SIZE / 3),
                ),
            ]
        )

        cv2.drawContours(
            img,
            [triangle_coordinates],
            0,
            info_box_color,
            -1,
        )

        cv2.rectangle(
            img,
            (x + int(w / 5), y + h + int(IDENTIFIED_IMG_SIZE / 3)),
            (
                x + w - int(w / 5),
                y + h + IDENTIFIED_IMG_SIZE - int(IDENTIFIED_IMG_SIZE / 5),
            ),
            info_box_color,
            cv2.FILLED,
        )

        cv2.putText(
            img,
            analysis_report,
            (x + int(w / 3.5), y + h + int(IDENTIFIED_IMG_SIZE / 1.5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 111, 255),
            2,
        )

    return img
