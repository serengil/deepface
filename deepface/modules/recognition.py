# built-in dependencies
import os
import pickle
from typing import List, Union, Optional, Dict, Any
import time

# 3rd party dependencies
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# project dependencies
from deepface.commons import package_utils
from deepface.modules import representation, detection, verification
from deepface.commons import logger as log

logger = log.get_singletonish_logger()


def find(
    img_path: Union[str, np.ndarray],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
    silent: bool = False,
) -> List[pd.DataFrame]:
    """
    Identify individuals in a database

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2'.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'.

        align (boolean): Perform alignment based on the eye positions.

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

        silent (boolean): Suppress or allow some log messages for a quieter analysis process.

    Returns:
        results (List[pd.DataFrame]): A list of pandas dataframes. Each dataframe corresponds
            to the identity information for an individual detected in the source image.
            The DataFrame columns include:

            - 'identity': Identity label of the detected individual.

            - 'target_x', 'target_y', 'target_w', 'target_h': Bounding box coordinates of the
                    target face in the database.

            - 'source_x', 'source_y', 'source_w', 'source_h': Bounding box coordinates of the
                    detected face in the source image.

            - 'threshold': threshold to determine a pair whether same person or different persons

            - 'distance': Similarity score between the faces based on the
                    specified model and distance metric
    """

    tic = time.time()

    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    file_parts = [
        "ds",
        "model",
        model_name,
        "detector",
        detector_backend,
        "aligned" if align else "unaligned",
        "normalization",
        normalization,
        "expand",
        str(expand_percentage),
    ]

    file_name = "_".join(file_parts) + ".pkl"
    file_name = file_name.replace("-", "").lower()

    datastore_path = os.path.join(db_path, file_name)
    representations = []

    # required columns for representations
    df_cols = [
        "identity",
        "hash",
        "embedding",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
    ]

    # Ensure the proper pickle file exists
    if not os.path.exists(datastore_path):
        with open(datastore_path, "wb") as f:
            pickle.dump([], f)

    # Load the representations from the pickle file
    with open(datastore_path, "rb") as f:
        representations = pickle.load(f)

    # check each item of representations list has required keys
    for i, current_representation in enumerate(representations):
        missing_keys = list(set(df_cols) - set(current_representation.keys()))
        if len(missing_keys) > 0:
            raise ValueError(
                f"{i}-th item does not have some required keys - {missing_keys}."
                f"Consider to delete {datastore_path}"
            )

    # embedded images
    pickled_images = [representation["identity"] for representation in representations]

    # Get the list of images on storage
    storage_images = __list_images(path=db_path)

    if len(storage_images) == 0:
        raise ValueError(f"No item found in {db_path}")

    # Enforce data consistency amongst on disk images and pickle file
    must_save_pickle = False
    new_images = list(set(storage_images) - set(pickled_images))  # images added to storage
    old_images = list(set(pickled_images) - set(storage_images))  # images removed from storage

    # detect replaced images
    replaced_images = []
    for current_representation in representations:
        identity = current_representation["identity"]
        if identity in old_images:
            continue
        alpha_hash = current_representation["hash"]
        beta_hash = package_utils.find_hash_of_file(identity)
        if alpha_hash != beta_hash:
            logger.debug(f"Even though {identity} represented before, it's replaced later.")
            replaced_images.append(identity)

    if not silent and (len(new_images) > 0 or len(old_images) > 0 or len(replaced_images) > 0):
        logger.info(
            f"Found {len(new_images)} newly added image(s)"
            f", {len(old_images)} removed image(s)"
            f", {len(replaced_images)} replaced image(s)."
        )

    # append replaced images into both old and new images. these will be dropped and re-added.
    new_images = new_images + replaced_images
    old_images = old_images + replaced_images

    # remove old images first
    if len(old_images) > 0:
        representations = [rep for rep in representations if rep["identity"] not in old_images]
        must_save_pickle = True

    # find representations for new images
    if len(new_images) > 0:
        representations += __find_bulk_embeddings(
            employees=new_images,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            normalization=normalization,
            silent=silent,
        )  # add new images
        must_save_pickle = True

    if must_save_pickle:
        with open(datastore_path, "wb") as f:
            pickle.dump(representations, f)
        if not silent:
            logger.info(f"There are now {len(representations)} representations in {file_name}")

    # Should we have no representations bailout
    if len(representations) == 0:
        if not silent:
            toc = time.time()
            logger.info(f"find function duration {toc - tic} seconds")
        return []

    # ----------------------------
    # now, we got representations for facial database
    df = pd.DataFrame(representations)

    if silent is False:
        logger.info(f"Searching {img_path} in {df.shape[0]} length datastore")

    # img path might have more than once face
    source_objs = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
    )

    resp_obj = []

    for source_obj in source_objs:
        source_img = source_obj["face"]
        source_region = source_obj["facial_area"]
        target_embedding_obj = representation.represent(
            img_path=source_img,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = source_region["x"]
        result_df["source_y"] = source_region["y"]
        result_df["source_w"] = source_region["w"]
        result_df["source_h"] = source_region["h"]

        distances = []
        for _, instance in df.iterrows():
            source_representation = instance["embedding"]
            if source_representation is None:
                distances.append(float("inf"))  # no representation for this image
                continue

            target_dims = len(list(target_representation))
            source_dims = len(list(source_representation))
            if target_dims != source_dims:
                raise ValueError(
                    "Source and target embeddings must have same dimensions but "
                    + f"{target_dims}:{source_dims}. Model structure may change"
                    + " after pickle created. Delete the {file_name} and re-run."
                )

            distance = verification.find_distance(
                source_representation, target_representation, distance_metric
            )

            distances.append(distance)

            # ---------------------------
        target_threshold = threshold or verification.find_threshold(model_name, distance_metric)

        result_df["threshold"] = target_threshold
        result_df["distance"] = distances

        result_df = result_df.drop(columns=["embedding"])
        # pylint: disable=unsubscriptable-object
        result_df = result_df[result_df["distance"] <= target_threshold]
        result_df = result_df.sort_values(by=["distance"], ascending=True).reset_index(drop=True)

        resp_obj.append(result_df)

    # -----------------------------------

    if not silent:
        toc = time.time()
        logger.info(f"find function duration {toc - tic} seconds")

    return resp_obj


def __list_images(path: str) -> List[str]:
    """
    List images in a given path
    Args:
        path (str): path's location
    Returns:
        images (list): list of exact image paths
    """
    images = []
    for r, _, f in os.walk(path):
        for file in f:
            exact_path = os.path.join(r, file)

            _, ext = os.path.splitext(exact_path)
            ext_lower = ext.lower()

            if ext_lower not in {".jpg", ".jpeg", ".png"}:
                continue

            with Image.open(exact_path) as img:  # lazy
                if img.format.lower() in ["jpeg", "png"]:
                    images.append(exact_path)
    return images


def __find_bulk_embeddings(
    employees: List[str],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
) -> List[Dict["str", Any]]:
    """
    Find embeddings of a list of images

    Args:
        employees (list): list of exact image paths

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (str): face detector model name

        enforce_detection (bool): set this to False if you
            want to proceed when you cannot detect any face

        align (bool): enable or disable alignment of image
            before feeding to facial recognition model

        expand_percentage (int): expand detected facial area with a
            percentage (default is 0).

        normalization (bool): normalization technique

        silent (bool): enable or disable informative logging
    Returns:
        representations (list): pivot list of dict with
            image name, hash, embedding and detected face area's coordinates
    """
    representations = []
    for employee in tqdm(
        employees,
        desc="Finding representations",
        disable=silent,
    ):
        file_hash = package_utils.find_hash_of_file(employee)

        try:
            img_objs = detection.extract_faces(
                img_path=employee,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
            )

        except ValueError as err:
            logger.error(f"Exception while extracting faces from {employee}: {str(err)}")
            img_objs = []

        if len(img_objs) == 0:
            representations.append(
                {
                    "identity": employee,
                    "hash": file_hash,
                    "embedding": None,
                    "target_x": 0,
                    "target_y": 0,
                    "target_w": 0,
                    "target_h": 0,
                }
            )
        else:
            for img_obj in img_objs:
                img_content = img_obj["face"]
                img_region = img_obj["facial_area"]
                embedding_obj = representation.represent(
                    img_path=img_content,
                    model_name=model_name,
                    enforce_detection=enforce_detection,
                    detector_backend="skip",
                    align=align,
                    normalization=normalization,
                )

                img_representation = embedding_obj[0]["embedding"]
                representations.append(
                    {
                        "identity": employee,
                        "hash": file_hash,
                        "embedding": img_representation,
                        "target_x": img_region["x"],
                        "target_y": img_region["y"],
                        "target_w": img_region["w"],
                        "target_h": img_region["h"],
                    }
                )

    return representations
