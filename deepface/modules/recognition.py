# built-in dependencies
from typing import List, Union, Optional, Dict, Any, Set, IO, cast, Callable
import time

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
from pyarrow import feather
from pyarrow import parquet

# project dependencies
from deepface.commons import image_utils
from deepface.modules import representation, detection, verification, modeling
from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules.filestore import FileStore, build_file_store

from deepface.modules.exceptions import (
    ImgNotFound,
    EmptyDatasource,
    SpoofDetected,
    DimensionMismatchError,
)
from deepface.commons.logger import Logger

logger = Logger()

# embeddings are stored as float64 to keep them bit-exact with the in-memory python floats
DATASTORE_SCHEMA = pa.schema(
    [
        ("identity", pa.string()),
        ("hash", pa.string()),
        ("embedding", pa.list_(pa.float64())),
        ("target_x", pa.int64()),
        ("target_y", pa.int64()),
        ("target_w", pa.int64()),
        ("target_h", pa.int64()),
    ]
)

DATASTORE_FORMATS = {"feather", "parquet"}


# pylint: disable=too-many-arguments, too-many-positional-arguments
def find(
    img_path: Union[str, NDArray[Any], IO[bytes]],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: Union[str, Callable] = "cosine", # type: ignore[type-arg]
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    similarity_search: bool = False,
    k: Optional[int] = None,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
    silent: bool = False,
    refresh_database: bool = True,
    anti_spoofing: bool = False,
    batched: bool = False,
    datastore_format: str = "feather",
) -> Union[List[pd.DataFrame], List[List[Dict[str, Any]]]]:
    """
    Identify individuals in a database

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process. Besides a local
            folder, it can be an S3 location (s3://bucket/prefix) or an FTP location
            (ftp://user:password@host:port/path). The representations datastore is stored
            in the same location. S3 credentials and endpoint are resolved by boto3, e.g. with
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_ENDPOINT_URL environment variables.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        distance_metric (string or callable): Metric for measuring similarity. Options:
            'cosine', 'euclidean', 'euclidean_l2', 'angular' (default is cosine).
            With batched=False, a custom callable receives two 1D NumPy arrays and
            returns a scalar distance. With batched=True, it receives gallery embeddings
            of shape (N, D) first and query embeddings of shape (M, D) second, and must
            return a distance matrix of shape (M, N). Rows correspond to query faces;
            columns correspond to gallery entries. Scalar-only callables require
            batched=False. An explicit threshold is required for custom callables.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov11n',
            'yolov11s', 'yolov11m', 'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m', 'yolov12l',
            'centerface' or 'skip'.

        align (boolean): Perform alignment based on the eye positions.

        similarity_search (boolean): If False, performs identity verification and returns images of
            the same person. If True, performs similarity search and returns visually similar faces
            (e.g., celebrity or parental look-alikes). Default is False.

        k (int): Number of top similar faces to retrieve from the database for each detected face.
            If not specified, all faces within the threshold will be returned (default is None).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

        silent (boolean): Suppress or allow some log messages for a quieter analysis process.

        refresh_database (boolean): Synchronizes the images representation datastore file with the
            directory/db files, if set to false, it will ignore any file changes inside the db_path
            directory (default is True).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        datastore_format (str): File format of the representations datastore stored in
            db_path. Options: feather, parquet (default is feather).

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

            - 'confidence': Confidence score indicating the likelihood that the faces belong to
                    the same individual. This is calculated based on the distance and the threshold.
    """
    if not isinstance(distance_metric, str) and threshold is None:
        raise ValueError("Threshold must be specified when using custom distance metrics")

    if datastore_format not in DATASTORE_FORMATS:
        raise ValueError(
            f"Unsupported datastore_format {datastore_format}. "
            f"Options: {', '.join(sorted(DATASTORE_FORMATS))}"
        )

    tic = time.time()

    store = build_file_store(db_path)

    img, _ = image_utils.load_image(img_path)
    if img is None:
        raise ImgNotFound(f"Passed image path {img_path} does not exist!")

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

    file_name = "_".join(file_parts)
    file_name = file_name.replace("-", "").lower() + f".{datastore_format}"

    datastore_path = store.join(file_name)
    representations = []

    # required columns for representations
    df_cols = {
        "identity",
        "hash",
        "embedding",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
    }

    # Ensure the proper datastore file exists
    if not store.exists(datastore_path):
        __save_representations(
            store=store, datastore_path=datastore_path, datastore_format=datastore_format
        )

    # Load the representations from the existing datastore
    representations = __load_representations(
        store=store, datastore_path=datastore_path, datastore_format=datastore_format
    )

    # check each item of representations list has required keys
    for i, current_representation in enumerate(representations):
        missing_keys = df_cols - set(current_representation.keys())
        if len(missing_keys) > 0:
            raise ValueError(
                f"{i}-th item does not have some required keys - {missing_keys}."
                f"Consider to delete {datastore_path}"
            )

    # Get the list of images on storage
    storage_images = set(store.list_images())

    if len(storage_images) == 0 and refresh_database is True:
        raise EmptyDatasource(f"No item found in {db_path}")
    if len(representations) == 0 and refresh_database is False:
        raise EmptyDatasource(f"Nothing is found in {datastore_path}")

    must_save_datastore = False
    new_images, old_images, replaced_images = set(), set(), set()

    if not refresh_database:
        logger.info(
            f"Could be some changes in {db_path} not tracked."
            "Set refresh_database to true to assure that any changes will be tracked."
        )

    # Enforce data consistency amongst on disk images and datastore file
    if refresh_database:
        # embedded images
        stored_images = {representation["identity"] for representation in representations}

        new_images = storage_images - stored_images  # images added to storage
        old_images = stored_images - storage_images  # images removed from storage

        # detect replaced images
        for current_representation in representations:
            identity = current_representation["identity"]
            if identity in old_images:
                continue
            alpha_hash = current_representation["hash"]
            beta_hash = store.find_hash(identity)
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
        must_save_datastore = True

    # find representations for new images
    if len(new_images) > 0:
        representations += __find_bulk_embeddings(
            store=store,
            employees=new_images,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            normalization=normalization,
            silent=silent,
        )  # add new images
        must_save_datastore = True

    if must_save_datastore:
        __save_representations(
            store=store,
            datastore_path=datastore_path,
            representations=representations,
            datastore_format=datastore_format,
        )
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
    source_objs: List[Dict[str, Any]] = cast(
        List[Dict[str, Any]],
        detection.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            anti_spoofing=anti_spoofing,
            color_face="bgr",  # `represent` expects images in bgr format.
        ),
    )

    pretuned_threshold = verification.find_threshold(model_name, distance_metric)
    target_threshold = pretuned_threshold if threshold is None else threshold

    if batched:
        return find_batched(
            representations=representations,
            source_objs=source_objs,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align,
            threshold=target_threshold,
            normalization=normalization,
            anti_spoofing=anti_spoofing,
            similarity_search=similarity_search,
            k=k,
        )

    df = pd.DataFrame(representations)

    if silent is False:
        logger.info(f"Searching {img_path} in {df.shape[0]} length datastore")

    resp_obj = []

    for source_obj in source_objs:
        if anti_spoofing is True and source_obj.get("is_real", True) is False:
            raise SpoofDetected("Spoof detected in the given image.")
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
        target_embedding_obj = cast(List[Dict[str, Any]], target_embedding_obj)
        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img

        result_df["threshold"] = target_threshold
        result_df["source_x"] = source_region["x"]
        result_df["source_y"] = source_region["y"]
        result_df["source_w"] = source_region["w"]
        result_df["source_h"] = source_region["h"]

        distances: List[float] = []
        confidences: List[float] = []
        for _, instance in df.iterrows():
            source_representation = instance["embedding"]
            if source_representation is None:
                # no representation for this image
                distances.append(float("inf"))
                confidences.append(0.0)
                continue

            target_dims = len(list(target_representation))
            source_dims = len(list(source_representation))
            if target_dims != source_dims:
                raise DimensionMismatchError(
                    "Source and target embeddings must have same dimensions but "
                    + f"{target_dims}:{source_dims}. Model structure may change"
                    + f" after datastore created. Delete the {file_name} and re-run."
                )

            distance: float = float(
                cast(
                    np.float64,
                    verification.find_distance(
                        source_representation, target_representation, distance_metric
                    ),
                )
            )

            confidence = verification.find_confidence(
                distance=distance,
                model_name=model_name,
                distance_metric=distance_metric,
                verified=bool(distance <= target_threshold),
            )

            distances.append(distance)
            confidences.append(confidence)

            # ---------------------------

        result_df["distance"] = distances
        result_df["confidence"] = confidences

        result_df = result_df.drop(columns=["embedding"])
        # pylint: disable=unsubscriptable-object

        if similarity_search is False:
            result_df = result_df[result_df["distance"] <= result_df["threshold"]]

        result_df = result_df.sort_values(by=["distance"], ascending=True).reset_index(drop=True)

        if k is not None and len(result_df) > k:
            result_df = result_df.head(k)

        resp_obj.append(result_df)

    # -----------------------------------

    if not silent:
        toc = time.time()
        logger.info(f"find function duration {toc - tic} seconds")

    return resp_obj


def __find_bulk_embeddings(
    store: FileStore,
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
        store (FileStore): file store where images live

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
        file_hash = store.find_hash(employee)

        try:
            img_objs: List[Dict[str, Any]] = cast(
                List[Dict[str, Any]],
                detection.extract_faces(
                    img_path=store.load_image(employee),
                    detector_backend=detector_backend,
                    grayscale=False,
                    enforce_detection=enforce_detection,
                    align=align,
                    expand_percentage=expand_percentage,
                    color_face="bgr",  # `represent` expects images in bgr format.
                ),
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
                embedding_obj = cast(List[Dict[str, Any]], embedding_obj)
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
    distance_metric: Union[str, Callable] = "cosine", # type: ignore[type-arg]
    enforce_detection: bool = True,
    align: bool = True,
    threshold: Optional[float] = None,
    normalization: str = "base",
    anti_spoofing: bool = False,
    similarity_search: bool = False,
    k: Optional[int] = None,
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

        distance_metric (string or callable): Metric for measuring similarity. Options:
            'cosine', 'euclidean', 'euclidean_l2', 'angular'. A custom callable receives
            gallery embeddings of shape (N, D) first and query embeddings of shape (M, D)
            second. It must return a distance matrix of shape (M, N), with one row per
            query face and one column per gallery entry. The callable is invoked once
            with both complete matrices, not once per embedding pair.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s',
            'yolov11m', 'centerface' or 'skip'.

        align (boolean): Perform alignment based on the eye positions.

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

        silent (boolean): Suppress or allow some log messages for a quieter analysis process.

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        similarity_search (boolean): If False, performs identity verification and returns images of
            the same person. If True, performs similarity search and returns visually similar faces
            (e.g., celebrity or parental look-alikes). Default is False.

        k (int): Number of top similar faces to retrieve from the database for each detected face.
            If not specified, all faces within the threshold will be returned (default is None).

    Returns:
        List[List[Dict[str, Any]]]:
            A list where each element corresponds to a source face and
            contains a list of dictionaries with matching faces.
    """

    # build_model caches models as singletons, so this returns the already built instance
    # (represent calls build_model for the same model) without any extra cost
    model: FacialRecognition = modeling.build_model(
        task="facial_recognition", model_name=model_name
    )
    dims = model.output_shape

    embeddings_list = []
    valid_mask_lst = []
    metadata: Set[str] = set()

    for item in representations:
        emb = item.get("embedding")
        if emb is not None:
            embeddings_list.append(emb)
            valid_mask_lst.append(True)
        else:
            embeddings_list.append(np.zeros(dims))
            valid_mask_lst.append(False)

        metadata.update(item.keys())

    # remove embedding key from other keys
    metadata.discard("embedding")
    metadata_lst = list(metadata)

    embeddings = np.array(embeddings_list)  # (N, D)
    valid_mask = np.array(valid_mask_lst)  # (N,)

    data = {
        key: np.array([item.get(key, None) for item in representations]) for key in metadata_lst
    }

    target_embeddings = []
    source_regions = []
    target_thresholds = []

    target_threshold = threshold if similarity_search is False else np.inf

    for source_obj in source_objs:
        if anti_spoofing and not source_obj.get("is_real", True):
            raise SpoofDetected("Spoof detected in the given image.")

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
        target_embedding_obj = cast(List[Dict[str, Any]], target_embedding_obj)
        target_representation = target_embedding_obj[0]["embedding"]

        target_embeddings.append(target_representation)
        source_regions.append(source_region)
        target_thresholds.append(target_threshold)

    target_embeddings_np = np.array(target_embeddings)  # (M, D)
    target_thresholds_np = np.array(target_thresholds)  # (M,)
    source_regions_arr = {
        "source_x": np.array([region["x"] for region in source_regions]),
        "source_y": np.array([region["y"] for region in source_regions]),
        "source_w": np.array([region["w"] for region in source_regions]),
        "source_h": np.array([region["h"] for region in source_regions]),
    }

    distances: NDArray[Any] = cast(
        NDArray[Any],
        verification.find_distance(embeddings, target_embeddings_np, distance_metric),
    )  # (M, N)
    distances[:, ~valid_mask] = np.inf

    resp_obj = []
    for i in range(len(target_embeddings_np)):
        target_distances = distances[i]  # (N,)
        target_threshold = target_thresholds_np[i]

        N = embeddings.shape[0]
        result_data = dict(data)
        result_data.update(
            {
                "source_x": np.full(N, source_regions_arr["source_x"][i]),
                "source_y": np.full(N, source_regions_arr["source_y"][i]),
                "source_w": np.full(N, source_regions_arr["source_w"][i]),
                "source_h": np.full(N, source_regions_arr["source_h"][i]),
                "threshold": np.full(N, target_threshold),
                "distance": target_distances,
            }
        )

        mask = target_distances <= target_threshold

        filtered_data = {key: value[mask] for key, value in result_data.items()}

        sorted_indices = np.argsort(filtered_data["distance"])
        sorted_data = {key: value[sorted_indices] for key, value in filtered_data.items()}

        num_results = len(sorted_data["distance"])
        result_dicts = [
            {key: sorted_data[key][i] for key in sorted_data} for i in range(num_results)
        ]

        if k is not None and len(result_dicts) > k:
            result_dicts = result_dicts[:k]

        resp_obj.append(result_dicts)
    return resp_obj


def __save_representations(
    store: FileStore,
    datastore_path: str,
    datastore_format: str,
    representations: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Save representations to a feather or parquet file

    Args:
        store (FileStore): file store where the datastore file lives
        datastore_path (str): path to the datastore file
        datastore_format (str): feather or parquet
        representations (list): list of representations to be saved
    """
    representations = representations or []
    table = pa.Table.from_pydict(
        {
            name: [representation[name] for representation in representations]
            for name in DATASTORE_SCHEMA.names
        },
        schema=DATASTORE_SCHEMA,
    )

    sink = pa.BufferOutputStream()
    if datastore_format == "feather":
        # size is not a concern, uncompressed is the fastest to read and write
        feather.write_feather(table, sink, compression="uncompressed")
    else:
        parquet.write_table(table, sink)

    store.write_bytes(datastore_path, sink.getvalue().to_pybytes())


def __load_representations(
    store: FileStore,
    datastore_path: str,
    datastore_format: str,
) -> List[Dict[str, Any]]:
    """
    Load representations from a feather or parquet file

    Args:
        store (FileStore): file store where the datastore file lives
        datastore_path (str): path to the datastore file
        datastore_format (str): feather or parquet
    Returns:
        representations (list): list of loaded representations
    """
    source = pa.BufferReader(store.read_bytes(datastore_path))
    if datastore_format == "feather":
        table = feather.read_table(source)
    else:
        table = parquet.read_table(source)

    missing_columns = set(DATASTORE_SCHEMA.names) - set(table.column_names)
    if len(missing_columns) > 0:
        raise ValueError(
            f"Datastore {datastore_path} does not have some required columns - "
            f"{missing_columns}. Consider to delete it."
        )
    table = table.select(DATASTORE_SCHEMA.names).cast(DATASTORE_SCHEMA)

    columns = {
        name: table.column(name).to_pylist()
        for name in DATASTORE_SCHEMA.names
        if name != "embedding"
    }
    columns["embedding"] = __embeddings_to_pylist(table.column("embedding"))

    return [dict(zip(columns.keys(), row)) for row in zip(*columns.values())]


def __embeddings_to_pylist(column: pa.ChunkedArray) -> List[Optional[List[float]]]:
    """
    Convert an arrow list<float64> column into python lists of floats, or None for missing ones.
        Converting the flat values once and slicing them is much faster than to_pylist on lists.
    Args:
        column (pa.ChunkedArray): embedding column of the datastore
    Returns:
        embeddings (list): list of embeddings, None if no face was found in the image
    """
    embeddings: List[Optional[List[float]]] = []
    for chunk in column.chunks:
        values = chunk.values.to_numpy(zero_copy_only=False).tolist()
        offsets = chunk.offsets.to_numpy(zero_copy_only=False).tolist()
        is_valid = chunk.is_valid().to_numpy(zero_copy_only=False).tolist()
        embeddings.extend(
            values[offsets[i] : offsets[i + 1]] if is_valid[i] else None for i in range(len(chunk))
        )
    return embeddings
