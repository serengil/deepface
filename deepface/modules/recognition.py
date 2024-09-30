# built-in dependencies
import os
import pickle
from typing import List, Union, Optional, Dict, Any, Set
import time

# 3rd party dependencies
import numpy as np
import pandas as pd
from tqdm import tqdm

# project dependencies
from deepface.commons import image_utils
from deepface.modules import representation, detection, verification
from deepface.commons.logger import Logger

logger = Logger()


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
    refresh_database: bool = True,
    anti_spoofing: bool = False,
    batched: bool = False,
) -> Union[List[pd.DataFrame], List[List[Dict[str, Any]]]]:
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

        refresh_database (boolean): Synchronizes the images representation (pkl) file with the
            directory/db files, if set to false, it will ignore any file changes inside the db_path
            directory (default is True).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).


    Returns:
        results (List[pd.DataFrame] or List[List[Dict[str, Any]]]): 
            A list of pandas dataframes (if `batched=False`) or
            a list of dicts (if `batched=True`).
            Each dataframe or dict corresponds to the identity information for
            an individual detected in the source image.

            Note: If you have a large database and/or a source photo with many faces,
            use `batched=True`, as it is optimized for large batch processing. 
            Please pay attention that when using `batched=True`, the function returns 
            a list of dicts (not a list of DataFrames),
            but with the same keys as the columns in the DataFrame.
            
            The DataFrame columns or dict keys include:

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

    if not os.path.isdir(db_path):
        raise ValueError(f"Passed path {db_path} does not exist!")

    img, _ = image_utils.load_image(img_path)
    if img is None:
        raise ValueError(f"Passed image path {img_path} does not exist!")

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
        missing_keys = set(df_cols) - set(current_representation.keys())
        if len(missing_keys) > 0:
            raise ValueError(
                f"{i}-th item does not have some required keys - {missing_keys}."
                f"Consider to delete {datastore_path}"
            )

    # embedded images
    pickled_images = [representation["identity"] for representation in representations]

    # Get the list of images on storage
    storage_images = image_utils.list_images(path=db_path)

    if len(storage_images) == 0 and refresh_database is True:
        raise ValueError(f"No item found in {db_path}")
    if len(representations) == 0 and refresh_database is False:
        raise ValueError(f"Nothing is found in {datastore_path}")

    must_save_pickle = False
    new_images, old_images, replaced_images = set(), set(), set()

    if not refresh_database:
        logger.info(
            f"Could be some changes in {db_path} not tracked."
            "Set refresh_database to true to assure that any changes will be tracked."
        )

    # Enforce data consistency amongst on disk images and pickle file
    if refresh_database:
        new_images = set(storage_images) - set(pickled_images)  # images added to storage
        old_images = set(pickled_images) - set(storage_images)  # images removed from storage

        # detect replaced images
        for current_representation in representations:
            identity = current_representation["identity"]
            if identity in old_images:
                continue
            alpha_hash = current_representation["hash"]
            beta_hash = image_utils.find_image_hash(identity)
            if alpha_hash != beta_hash:
                logger.debug(f"Even though {identity} represented before, it's replaced later.")
                replaced_images.add(identity)

    if not silent and (len(new_images) > 0 or len(old_images) > 0 or len(replaced_images) > 0):
        logger.info(
            f"Found {len(new_images)} newly added image(s)"
            f", {len(old_images)} removed image(s)"
            f", {len(replaced_images)} replaced image(s)."
        )

    # append replaced images into both old and new images. these will be dropped and re-added.
    new_images.update(replaced_images)
    old_images.update(replaced_images)

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

    # img path might have more than once face
    source_objs = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )

    if batched:
        return find_batched(
            representations,
            source_objs,
            model_name,
            distance_metric,
            enforce_detection,
            align,
            threshold,
            normalization,
            anti_spoofing
        )

    df = pd.DataFrame(representations)

    if silent is False:
        logger.info(f"Searching {img_path} in {df.shape[0]} length datastore")

    resp_obj = []

    for source_obj in source_objs:
        if anti_spoofing is True and source_obj.get("is_real", True) is False:
            raise ValueError("Spoof detected in the given image.")
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


def __find_bulk_embeddings(
    employees: Set[str],
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
        file_hash = image_utils.find_image_hash(employee)

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

def find_batched(
    representations: List[Dict[str, Any]],
    source_objs: List[Dict[str, Any]],
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    threshold: Optional[float] = None,
    normalization: str = "base",
    anti_spoofing: bool = False,
) -> List[List[Dict[str, Any]]]:
    """
    Perform batched face recognition by comparing source face embeddings with a set of
    target embeddings. It calculates pairwise distances between the source and target
    embeddings using the specified distance metric.
    The function uses batch processing for efficient computation of distances.

    Args:
        representations (List[Dict[str, Any]]): 
            A list of dictionaries containing precomputed target embeddings and associated metadata. 
            Each dictionary should have at least the key `embedding`.
        
        source_objs (List[Dict[str, Any]]): 
            A list of dictionaries representing the source images to compare against
            the target embeddings. Each dictionary should contain:
                - `face`: The image data or path to the source face image.
                - `facial_area`: A dictionary with keys `x`, `y`, `w`, `h`
                   indicating the facial region.
                - Optionally, `is_real`: A boolean indicating if the face is real
                  (used for anti-spoofing).
        
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2'.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'.

        align (boolean): Perform alignment based on the eye positions.

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

        silent (boolean): Suppress or allow some log messages for a quieter analysis process.

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        List[List[Dict[str, Any]]]: 
            A list where each element corresponds to a source face and
            contains a list of dictionaries with matching faces.
    """
    embeddings_list = []
    valid_mask = []
    other_keys = set()

    for item in representations:
        emb = item.get('embedding')
        if emb is not None:
            embeddings_list.append(emb)
            valid_mask.append(True)
        else:
            embeddings_list.append(np.zeros_like(representations[0]['embedding']))
            valid_mask.append(False)

        other_keys.update(item.keys())

    # remove embedding key from other keys
    other_keys.discard('embedding')
    other_keys = list(other_keys)

    embeddings = np.array(embeddings_list) # (N, D)
    valid_mask = np.array(valid_mask) # (N,)

    data = {
        key: np.array([item.get(key, None) for item in representations])
        for key in other_keys
    }

    target_embeddings = []
    source_regions = []
    target_thresholds = []

    for source_obj in source_objs:
        if anti_spoofing and not source_obj.get("is_real", True):
            raise ValueError("Spoof detected in the given image.")

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
        # it is safe to access 0 index because we already fed detected face to represent function
        target_representation = target_embedding_obj[0]["embedding"]

        target_embeddings.append(target_representation)
        source_regions.append(source_region)

        target_threshold = threshold or verification.find_threshold(model_name, distance_metric)
        target_thresholds.append(target_threshold)

    target_embeddings = np.array(target_embeddings) # (M, D)
    target_thresholds = np.array(target_thresholds) # (M,)
    source_regions_arr = {
        'source_x': np.array([region['x'] for region in source_regions]),
        'source_y': np.array([region['y'] for region in source_regions]),
        'source_w': np.array([region['w'] for region in source_regions]),
        'source_h': np.array([region['h'] for region in source_regions]),
    }

    def find_cosine_distance_batch(
        embeddings: np.ndarray, target_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Find the cosine distances between batches of embeddings
        Args:
            embeddings (np.ndarray): array of shape (N, D)
            target_embeddings (np.ndarray): array of shape (M, D)
        Returns:
            np.ndarray: distance matrix of shape (M, N)
        """
        embeddings_norm = verification.l2_normalize(embeddings, axis=1)
        target_embeddings_norm = verification.l2_normalize(target_embeddings, axis=1)
        cosine_similarities = np.dot(target_embeddings_norm, embeddings_norm.T)
        cosine_distances = 1 - cosine_similarities
        return cosine_distances

    def find_euclidean_distance_batch(
        embeddings: np.ndarray, target_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Find the Euclidean distances between batches of embeddings
        Args:
            embeddings (np.ndarray): array of shape (N, D)
            target_embeddings (np.ndarray): array of shape (M, D)
        Returns:
            np.ndarray: distance matrix of shape (M, N)
        """
        diff = embeddings[None, :, :] - target_embeddings[:, None, :] # (M, N, D)
        distances = np.linalg.norm(diff, axis=2) # (M, N)
        return distances

    def find_distance_batch(
        embeddings: np.ndarray, target_embeddings: np.ndarray, distance_metric: str,
    ) -> np.ndarray:
        """
        Find pairwise distances between batches of embeddings using the specified distance metric
        Args:
            embeddings (np.ndarray): array of shape (N, D)
            target_embeddings (np.ndarray): array of shape (M, D)
            distance_metric (str): distance metric ('cosine', 'euclidean', 'euclidean_l2')
        Returns:
            np.ndarray: distance matrix of shape (M, N)
        """
        if distance_metric == "cosine":
            distances = find_cosine_distance_batch(embeddings, target_embeddings)
        elif distance_metric == "euclidean":
            distances = find_euclidean_distance_batch(embeddings, target_embeddings)
        elif distance_metric == "euclidean_l2":
            embeddings_norm = verification.l2_normalize(embeddings, axis=1)
            target_embeddings_norm = verification.l2_normalize(target_embeddings, axis=1)
            distances = find_euclidean_distance_batch(embeddings_norm, target_embeddings_norm)
        else:
            raise ValueError("Invalid distance_metric passed - ", distance_metric)
        return np.round(distances, 6)

    distances = find_distance_batch(embeddings, target_embeddings, distance_metric) # (M, N)
    distances[:, ~valid_mask] = np.inf

    resp_obj = []

    for i in range(len(target_embeddings)):
        target_distances = distances[i] # (N,)
        target_threshold = target_thresholds[i]

        N = embeddings.shape[0]
        result_data = dict(data)
        result_data.update({
            'source_x': np.full(N, source_regions_arr['source_x'][i]),
            'source_y': np.full(N, source_regions_arr['source_y'][i]),
            'source_w': np.full(N, source_regions_arr['source_w'][i]),
            'source_h': np.full(N, source_regions_arr['source_h'][i]),
            'threshold': np.full(N, target_threshold),
            'distance': target_distances,
        })

        mask = target_distances <= target_threshold
        filtered_data = {key: value[mask] for key, value in result_data.items()}

        sorted_indices = np.argsort(filtered_data['distance'])
        sorted_data = {key: value[sorted_indices] for key, value in filtered_data.items()}

        num_results = len(sorted_data['distance'])
        result_dicts = [
            {key: sorted_data[key][i] for key in sorted_data}
            for i in range(num_results)
        ]
        resp_obj.append(result_dicts)
    return resp_obj
