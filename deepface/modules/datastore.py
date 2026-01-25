# built-in dependencies
import os
from typing import Any, Dict, IO, List, Union, Optional, cast
import uuid
import time
import math
import tempfile

# 3rd party dependencies
import pandas as pd
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.modules.database.types import Database
from deepface.modules.database.inventory import database_inventory

from deepface.modules.representation import represent
from deepface.modules.verification import (
    find_angular_distance,
    find_cosine_distance,
    find_euclidean_distance,
    l2_normalize as find_l2_normalize,
    find_threshold,
    find_confidence,
)
from deepface.commons.logger import Logger


logger = Logger()


# pylint: disable=too-many-positional-arguments, no-else-return
def register(
    img: Union[str, NDArray[Any], IO[bytes], List[str], List[NDArray[Any]], List[IO[bytes]]],
    img_name: Optional[str] = None,
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    l2_normalize: bool = False,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    database_type: str = "postgres",
    connection_details: Optional[Union[Dict[str, Any], str]] = None,
    connection: Any = None,
) -> Dict[str, Any]:
    """
    Register identities to database for face recognition
    Args:
        img (str or np.ndarray or IO[bytes] or list): The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If a list is provided, each element should be a string
            or numpy array representing an image, and the function will process images in batch.
        img_name (optional str): image name to store in db, if not provided then we will try to
            extract it from given img.
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov11n',
            'yolov11s', 'yolov11m', 'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m', 'yolov12l',
            'centerface' or 'skip' (default is opencv).
        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).
        align (bool): Flag to enable face alignment (default is True).
        l2_normalize (bool): Flag to enable L2 normalization (unit vector normalization)
        expand_percentage (int): expand detected facial area with a percentage (default is 0).
        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base).
        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).
        database_type (str): Type of database to register identities. Options: 'postgres', 'mongo',
            'weaviate', 'neo4j', 'pgvector', 'pinecone' (default is 'postgres').
        connection_details (dict or str): Connection details for the database.
        connection (Any): Existing database connection object. If provided, this connection
            will be used instead of creating a new one.

        Note:
            Instead of providing `connection` or `connection_details`, database connection
            information can be supplied via environment variables:
            - DEEPFACE_POSTGRES_URI
            - DEEPFACE_MONGO_URI
            - DEEPFACE_WEAVIATE_URI
            - DEEPFACE_NEO4J_URI
            - DEEPFACE_PINECONE_API_KEY
    Returns:
        result (dict): A dictionary containing registration results with following keys.
            - inserted (int): Number of embeddings successfully registered to the database.
    """
    db_client = __connect_database(
        database_type=database_type,
        connection_details=connection_details,
        connection=connection,
    )

    results = __get_embeddings(
        img=img,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        anti_spoofing=anti_spoofing,
        expand_percentage=expand_percentage,
        normalization=normalization,
        l2_normalize=l2_normalize,
        return_face=True,
    )

    embedding_records: List[Dict[str, Any]] = []
    for result in results:
        img_identifier = img_name or (
            img
            if isinstance(img, str) and img.endswith((".jpg", ".jpeg", ".png"))
            else str(uuid.uuid4())
        )

        embedding_record = {
            "id": None,
            "img_name": img_identifier,
            "face": result["face"],
            "model_name": model_name,
            "detector_backend": detector_backend,
            "embedding": result["embedding"],
            "aligned": align,
            "l2_normalized": l2_normalize,
        }
        embedding_records.append(embedding_record)

    inserted = db_client.insert_embeddings(embedding_records, batch_size=100)
    logger.debug(f"Successfully registered {inserted} embeddings to the database.")

    # Close the database connection if it was created internally
    if connection is None:
        db_client.close()

    return {"inserted": inserted}


def search(
    img: Union[str, NDArray[Any], IO[bytes], List[str], List[NDArray[Any]], List[IO[bytes]]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    l2_normalize: bool = False,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    similarity_search: bool = False,
    k: Optional[int] = None,
    database_type: str = "postgres",
    connection_details: Optional[Union[Dict[str, Any], str]] = None,
    connection: Any = None,
    search_method: str = "exact",
) -> List[pd.DataFrame]:
    """
    Search for identities in database for face recognition. This is a stateless facial
        recognition function. Use find function to do it in a stateful way.
    Args:
        img (str or np.ndarray or IO[bytes] or list): The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If a list is provided, each element should be a string
            or numpy array representing an image, and the function will process images in batch.
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov11n',
            'yolov11s', 'yolov11m', 'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m', 'yolov12l',
            'centerface' or 'skip' (default is opencv).
        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'angular' (default is cosine).
        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).
        align (bool): Flag to enable face alignment (default is True).
        l2_normalize (bool): Flag to enable L2 normalization (unit vector normalization)
        expand_percentage (int): expand detected facial area with a percentage (default is 0).
        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base).
        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).
        similarity_search (boolean): If False, performs identity verification and returns images of
            the same person. If True, performs similarity search and returns visually similar faces
            (e.g., celebrity or parental look-alikes). Default is False.
        k (int): Number of top similar faces to retrieve from the database for each detected face.
            If not specified, all faces within the threshold will be returned (default is None).
        search_method (str): Method to use for searching identities. Options: 'exact', 'ann'.
            To use ann search, you must run build_index function first to create the index.
        database_type (str): Type of database to search identities. Options: 'postgres', 'mongo',
            'weaviate', 'neo4j', 'pgvector', 'pinecone' (default is 'postgres').
        connection_details (dict or str): Connection details for the database.
        connection (Any): Existing database connection object. If provided, this connection
            will be used instead of creating a new one.

        Note:
            Instead of providing `connection` or `connection_details`, database connection
            information can be supplied via environment variables:
            - DEEPFACE_POSTGRES_URI
            - DEEPFACE_MONGO_URI
            - DEEPFACE_WEAVIATE_URI
            - DEEPFACE_NEO4J_URI
            - DEEPFACE_PINECONE_API_KEY
    Returns:
        results (List[pd.DataFrame]):
            A list of pandas dataframes or a list of dicts. Each dataframe or dict corresponds
                to the identity information for an individual detected in the source image.

            The DataFrame columns or dict keys include:
            - id: ID of the detected individual.
            - img_name: Name of the image file in the database.
            - model_name: Name of the model used for recognition.
            - aligned: Whether face alignment was performed.
            - l2_normalized: Whether L2 normalization was applied.
            - search_method: Method used for searching identities: exact or ann.
            - target_x, target_y, target_w, target_h: Bounding box coordinates of the
                target face in the database. Notice that source image's face coordinates
                are not included in the result here.
            - threshold: threshold to determine a pair whether same person or different persons
            - confidence: Confidence score indicating the likelihood that the images
                represent the same person. The score is between 0 and 100, where higher values
                indicate greater confidence in the verification result.
            - distance_metric: Distance metric used for similarity measurement.
                Distance metric will be ignored for ann search, and set to cosine if l2_normalize
                is True, euclidean if l2_normalize is False.
            - distance: Similarity score between the faces based on the specified model
                and distance metric
    """
    dfs: List[pd.DataFrame] = []

    # adjust distance metric
    if search_method == "ann":
        # ann does cosine for l2 normalized vectors, euclidean for non-l2 normalized vectors
        new_distance_metric = "cosine" if l2_normalize is True else "euclidean"
        if new_distance_metric != distance_metric:
            logger.warn(
                f"Overwriting distance_metric to '{new_distance_metric}' since "
                f"{'vectors are L2-norm' if l2_normalize else 'vectors are not L2-norm'}."
            )
            distance_metric = new_distance_metric
    elif search_method != "exact":
        if l2_normalize is True and distance_metric == "euclidean":
            logger.warn(
                "Overwriting distance_metric to 'euclidean_l2' since vectors are L2 normalized."
            )
            distance_metric = "euclidean_l2"

    threshold = find_threshold(model_name=model_name, distance_metric=distance_metric)

    db_client = __connect_database(
        database_type=database_type,
        connection_details=connection_details,
        connection=connection,
    )

    results = __get_embeddings(
        img=img,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        anti_spoofing=anti_spoofing,
        expand_percentage=expand_percentage,
        normalization=normalization,
        l2_normalize=l2_normalize,
        return_face=False,
    )

    is_vector_db = database_inventory[database_type]["is_vector_db"]

    if search_method == "ann" and is_vector_db is False:
        try:
            import faiss
        except ImportError as e:
            raise ValueError(
                "faiss is not installed. Please install faiss to use approximate nearest neighbour."
            ) from e

        embeddings_index_bytes = db_client.get_embeddings_index(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=align,
            l2_normalized=l2_normalize,
        )
        embeddings_index_buffer = np.frombuffer(embeddings_index_bytes, dtype=np.uint8)
        embeddings_index = faiss.deserialize_index(embeddings_index_buffer)
        logger.info("Loaded embeddings index from database.")

        for result in results:
            query_vector = np.array(result["embedding"], dtype="float32").reshape(1, -1)
            distances, indices = embeddings_index.search(query_vector, k or 20)
            instances = []
            for i, index in enumerate(indices[0]):
                distance = (
                    math.sqrt(distances[0][i])
                    if distance_metric == "euclidean"
                    else distances[0][i] / 2
                )
                verified = bool(distance <= threshold)
                instance = {
                    "id": index,
                    # "img_name": "N/A",  # need to fetch from DB if required
                    "model_name": model_name,
                    "detector_backend": detector_backend,
                    "aligned": align,
                    "l2_normalized": l2_normalize,
                    "search_method": search_method,
                    "target_x": result.get("facial_area", {}).get("x", None),
                    "target_y": result.get("facial_area", {}).get("y", None),
                    "target_w": result.get("facial_area", {}).get("w", None),
                    "target_h": result.get("facial_area", {}).get("h", None),
                    "threshold": threshold,
                    "distance_metric": distance_metric,
                    "distance": distance,
                    "confidence": find_confidence(
                        distance=distance,
                        model_name=model_name,
                        distance_metric=distance_metric,
                        verified=verified,
                    ),
                }

                if similarity_search is False and verified:
                    instances.append(instance)

            if len(instances) == 0:
                continue

            df = pd.DataFrame(instances)
            df = df.sort_values(by="distance", ascending=True).reset_index(drop=True)
            if k is not None and k > 0:
                df = df.nsmallest(k, "distance")

            # we should query DB to get img_name for each id
            id_mappings = db_client.search_by_id(ids=df["id"].tolist())
            ids_df = pd.DataFrame(id_mappings, columns=["id", "img_name"])
            df = df.merge(ids_df, on="id", how="left")
            del ids_df

            dfs.append(df)
        return dfs

    elif search_method == "ann" and is_vector_db is True:
        for result in results:
            target_vector: List[float] = result["embedding"]
            neighbours = db_client.search_by_vector(
                vector=target_vector,
                model_name=model_name,
                detector_backend=detector_backend,
                aligned=align,
                l2_normalized=l2_normalize,
                limit=k or 20,
            )
            if not neighbours:
                raise ValueError(
                    "No embeddings found in the database for the criteria "
                    f"{model_name=}, {detector_backend=}, {align=}, {l2_normalize=}."
                    "You must call register some embeddings to the database before using search."
                )
            instances = []
            for neighbour in neighbours:
                distance = neighbour["distance"]
                verified = bool(distance <= threshold)
                instance = {
                    "id": neighbour["id"],
                    "img_name": neighbour["img_name"],
                    "model_name": model_name,
                    "detector_backend": detector_backend,
                    "aligned": align,
                    "l2_normalized": l2_normalize,
                    "search_method": search_method,
                    "target_x": result.get("facial_area", {}).get("x", None),
                    "target_y": result.get("facial_area", {}).get("y", None),
                    "target_w": result.get("facial_area", {}).get("w", None),
                    "target_h": result.get("facial_area", {}).get("h", None),
                    "threshold": threshold,
                    "distance_metric": distance_metric,
                    "distance": distance,
                    "confidence": find_confidence(
                        distance=distance,
                        model_name=model_name,
                        distance_metric=distance_metric,
                        verified=verified,
                    ),
                }

                if similarity_search is False and verified:
                    instances.append(instance)

            if len(instances) > 0:
                df = pd.DataFrame(instances)
                df = df.sort_values(by="distance", ascending=True).reset_index(drop=True)
                if k is not None and k > 0:
                    df = df.nsmallest(k, "distance")
                dfs.append(df)

        return dfs

    elif search_method == "exact":
        source_embeddings = db_client.fetch_all_embeddings(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=align,
            l2_normalized=l2_normalize,
        )
        if not source_embeddings:
            raise ValueError(
                "No embeddings found in the database for the criteria "
                f"{model_name=}, {detector_backend=}, {align=}, {l2_normalize=}."
                "You must call register some embeddings to the database before using search."
            )

        for result in results:
            target_embedding = cast(List[float], result["embedding"])

            df = pd.DataFrame(source_embeddings)
            df["target_embedding"] = [target_embedding for _ in range(len(df))]
            df["search_method"] = search_method
            df["target_x"] = result.get("facial_area", {}).get("x", None)
            df["target_y"] = result.get("facial_area", {}).get("y", None)
            df["target_w"] = result.get("facial_area", {}).get("w", None)
            df["target_h"] = result.get("facial_area", {}).get("h", None)
            df["threshold"] = threshold
            df["distance_metric"] = distance_metric

            if distance_metric == "cosine":
                df["distance"] = df.apply(
                    lambda row: find_cosine_distance(row["embedding"], row["target_embedding"]),
                    axis=1,
                )
            elif distance_metric == "euclidean":
                df["distance"] = df.apply(
                    lambda row: find_euclidean_distance(row["embedding"], row["target_embedding"]),
                    axis=1,
                )
            elif distance_metric == "angular":
                df["distance"] = df.apply(
                    lambda row: find_angular_distance(row["embedding"], row["target_embedding"]),
                    axis=1,
                )
            elif distance_metric == "euclidean_l2":
                df["distance"] = df.apply(
                    lambda row: find_euclidean_distance(
                        find_l2_normalize(row["embedding"]),
                        find_l2_normalize(row["target_embedding"]),
                    ),
                    axis=1,
                )
            else:
                raise ValueError(f"Unsupported distance metric: {distance_metric}")

            df["confidence"] = df.apply(
                lambda row: find_confidence(
                    distance=row["distance"],
                    model_name=model_name,
                    distance_metric=distance_metric,
                    verified=bool(row["distance"] <= threshold),
                ),
                axis=1,
            )

            df = df.drop(columns=["embedding", "target_embedding"])

            if similarity_search is False:
                df = df[df["distance"] <= threshold]

            if k is not None and k > 0:
                df = df.nsmallest(k, "distance")

            df = df.sort_values(by="distance", ascending=True).reset_index(drop=True)

            dfs.append(df)

        return dfs

    else:
        raise ValueError(f"Unsupported search method: {search_method}")


def build_index(
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    align: bool = True,
    l2_normalize: bool = False,
    database_type: str = "postgres",
    connection_details: Optional[Union[Dict[str, Any], str]] = None,
    connection: Any = None,
    batch_size: int = 1000,
    max_neighbors_per_node: int = 32,
) -> None:
    """
    Build index for faster search in the database. You should set search_method to 'ann'
        in the search function to use the built index.
    Args:
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov11n',
            'yolov11s', 'yolov11m', 'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m', 'yolov12l',
            'centerface' or 'skip' (default is opencv).
        align (bool): Flag to enable face alignment (default is True).
        l2_normalize (bool): Flag to enable L2 normalization (unit vector normalization)
        max_neighbors_per_node (int): Maximum number of neighbors per node in the index
            (default is 32).
        database_type (str): Type of database to build index. Options: 'postgres', 'mongo',
            'weaviate', 'neo4j', 'pgvector', 'pinecone' (default is 'postgres').
        connection (Any): Existing database connection object. If provided, this connection
            will be used instead of creating a new one.
        connection_details (dict or str): Connection details for the database.

        Note:
            Instead of providing `connection` or `connection_details`, database connection
            information can be supplied via environment variables:
            - DEEPFACE_POSTGRES_URI
            - DEEPFACE_MONGO_URI
            - DEEPFACE_WEAVIATE_URI
            - DEEPFACE_NEO4J_URI
            - DEEPFACE_PINECONE_API_KEY
    """

    if database_inventory.get(database_type) is None:
        raise ValueError(f"Unsupported database type: {database_type}")

    is_vector_db = database_inventory[database_type]["is_vector_db"]

    if is_vector_db is True:
        logger.info(f"{database_type} manages its own indexes. No need to build index manually.")
        return

    try:
        import faiss
    except ImportError as e:
        raise ValueError("faiss is not installed. Please install faiss to use build_index.") from e

    db_client = __connect_database(
        database_type=database_type,
        connection_details=connection_details,
        connection=connection,
    )

    index = __get_index(
        db_client=db_client,
        model_name=model_name,
        detector_backend=detector_backend,
        align=align,
        l2_normalize=l2_normalize,
    )
    if index is not None:
        indexed_indices = faiss.vector_to_array(index.id_map)
        indexed_embeddings = set(indexed_indices)
        logger.info(f"Found {len(indexed_embeddings)} embeddings already indexed in the database.")
    else:
        logger.info("No existing index found in the database. A new index will be created.")
        indexed_embeddings = set()

    tic = time.time()
    source_embeddings = db_client.fetch_all_embeddings(
        model_name=model_name,
        detector_backend=detector_backend,
        aligned=align,
        l2_normalized=l2_normalize,
    )
    toc = time.time()

    if not source_embeddings:
        raise ValueError(
            "No embeddings found in the database for the criteria "
            f"{model_name=}, {detector_backend=}, {align=}, {l2_normalize=}."
            "You must call register some embeddings to the database before using build_index."
        )
    logger.info(
        f"Fetched {len(source_embeddings)} embeddings from database in {toc - tic:.2f} seconds."
    )

    unindexed_source_embeddings = [
        item for item in source_embeddings if item["id"] not in indexed_embeddings
    ]

    if not unindexed_source_embeddings:
        logger.info("All embeddings are already indexed. No new embeddings to index.")
        return

    ids = [item["id"] for item in unindexed_source_embeddings]
    vectors = np.array([item["embedding"] for item in unindexed_source_embeddings], dtype="float32")

    embedding_dim_size = len(source_embeddings[0]["embedding"])

    if index is None:
        base_index = faiss.IndexHNSWFlat(embedding_dim_size, max_neighbors_per_node)
        index = faiss.IndexIDMap(base_index)

    tic = time.time()
    for i in range(0, len(vectors), batch_size):
        batch_ids = np.array(ids[i : i + batch_size], dtype="int64")
        batch_vectors = vectors[i : i + batch_size]
        index.add_with_ids(batch_vectors, batch_ids)

    toc = time.time()
    logger.info(f"Added {len(vectors)} embeddings to index in {toc - tic:.2f} seconds.")

    index_path = os.path.join(
        tempfile.gettempdir(), f"{model_name}_{detector_backend}_{align}_{l2_normalize}.faiss"
    )

    # now create index from scratch, then think how to load an index and add new vectors to it
    tic = time.time()
    faiss.write_index(index, index_path)
    toc = time.time()
    logger.info(f"Saved index to disk in {toc - tic:.2f} seconds")

    with open(index_path, "rb") as f:
        index_data = f.read()

    tic = time.time()
    db_client.upsert_embeddings_index(
        model_name=model_name,
        detector_backend=detector_backend,
        aligned=align,
        l2_normalized=l2_normalize,
        index_data=index_data,
    )
    toc = time.time()
    logger.info(f"Upserted index to database in {toc - tic:.2f} seconds.")

    # clean up temp file
    if os.path.exists(index_path):
        os.remove(index_path)


def __get_embeddings(
    img: Union[str, NDArray[Any], IO[bytes], List[str], List[NDArray[Any]], List[IO[bytes]]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    l2_normalize: bool = False,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    return_face: bool = True,
) -> List[Dict[str, Any]]:
    """
    Get embeddings for given image(s)
    Args:
        img (str or np.ndarray or IO[bytes] or list): The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If a list is provided, each element should be a string
            or numpy array representing an image, and the function will process images in batch.
        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).
        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8n', 'yolov8m', 'yolov8l', 'yolov11n',
            'yolov11s', 'yolov11m', 'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m', 'yolov12l',
            'centerface' or 'skip' (default is opencv).
        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).
        align (bool): Flag to enable face alignment (default is True).
        l2_normalize (bool): Flag to enable L2 normalization (unit vector normalization)
        expand_percentage (int): expand detected facial area with a percentage (default is 0).
        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base).
        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).
        return_face (bool): Whether to return the aligned face along with the embedding
            (default is True).
    Returns:
        results (List[Dict]): A list of dictionaries containing embeddings and optionally
            aligned face images for each detected face in the input image(s).
    """
    results = represent(
        img_path=img,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        anti_spoofing=anti_spoofing,
        expand_percentage=expand_percentage,
        normalization=normalization,
        l2_normalize=l2_normalize,
        return_face=return_face,
    )

    if len(results) == 0:
        raise ValueError("No embeddings were detected in the provided image(s).")

    flat_results: List[Dict[str, Any]] = []
    for result in results:
        if isinstance(result, dict):
            flat_results.append(result)
        elif isinstance(result, list):
            flat_results.extend(result)
    return flat_results


def __connect_database(
    database_type: str = "postgres",
    connection_details: Optional[Union[Dict[str, Any], str]] = None,
    connection: Any = None,
) -> Database:
    """
    Connect to the specified database type
    Args:
        database_type (str): Type of database to connect. Options: 'postgres', 'mongo',
            'weaviate', 'neo4j', 'pgvector', 'pinecone' (default is 'postgres').
        connection_details (dict or str): Connection details for the database.
        connection (Any): Existing database connection object. If provided, this connection
            will be used instead of creating a new one.

        Note:
            Instead of providing `connection` or `connection_details`, database connection
            information can be supplied via environment variables:
            - DEEPFACE_POSTGRES_URI
            - DEEPFACE_MONGO_URI
            - DEEPFACE_WEAVIATE_URI
            - DEEPFACE_NEO4J_URI
            - DEEPFACE_PINECONE_API_KEY
    Returns:
        db_client (Database): An instance of the connected database client.
    """

    if database_inventory.get(database_type) is None:
        raise ValueError(f"Unsupported database type: {database_type}")

    client_class = database_inventory[database_type]["client"]
    return client_class(connection_details=connection_details, connection=connection)


def __get_index(
    db_client: Database,
    model_name: str,
    detector_backend: str,
    align: bool,
    l2_normalize: bool,
) -> Any:
    """
    Retrieve the embeddings index from the database
    Args:
        db_client (Database): An instance of the connected database client.
        model_name (str): Model for face recognition.
        detector_backend (string): face detector backend.
        align (bool): Flag to enable face alignment.
        l2_normalize (bool): Flag to enable L2 normalization (unit vector normalization)
    Returns:
        embeddings_index (Any): The deserialized embeddings index object, or None if not found.
    """
    import faiss

    try:
        embeddings_index_bytes = db_client.get_embeddings_index(
            model_name=model_name,
            detector_backend=detector_backend,
            aligned=align,
            l2_normalized=l2_normalize,
        )
        embeddings_index_buffer = np.frombuffer(embeddings_index_bytes, dtype=np.uint8)
        embeddings_index = faiss.deserialize_index(embeddings_index_buffer)
        return embeddings_index
    except ValueError:
        return None
