# built-in dependencies
from typing import Any, Dict, IO, List, Union, Optional, cast
import uuid
import time
import math

# 3rd party dependencies
import pandas as pd
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.modules.database.postgres import PostgresClient

from deepface.modules.representation import represent
from deepface.modules.verification import (
    find_angular_distance,
    find_cosine_distance,
    find_euclidean_distance,
    find_threshold,
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
    dfs: List[pd.DataFrame] = []

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

    if search_method == "ann":
        if l2_normalize is True:
            distance_metric = "cosine"
        elif l2_normalize is False:
            distance_metric = "euclidean"

        threshold = find_threshold(model_name=model_name, distance_metric=distance_metric)

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
                instance = {
                    "id": index,
                    # "img_name": "N/A",  # need to fetch from DB if required
                    "model_name": model_name,
                    "detector_backend": detector_backend,
                    "aligned": align,
                    "l2_normalized": l2_normalize,
                    "search_method": search_method,
                    "distance_metric": distance_metric,
                    "distance": (
                        math.sqrt(distances[0][i])
                        if distance_metric == "euclidean"
                        else distances[0][i] / 2
                    ),
                }
                if similarity_search is False and instance["distance"] <= threshold:
                    instances.append(instance)

            if len(instances) > 0:
                df = pd.DataFrame(instances)
                df = df.sort_values(by="distance", ascending=True).reset_index(drop=True)
                if k is not None and k > 0:
                    df = df.nsmallest(k, "distance")
                dfs.append(df)
        return dfs

    elif search_method == "exact":

        if l2_normalize is True and distance_metric == "euclidean":
            logger.warn("Overwriting distance_metric to 'cosine' since vectors are L2 normalized.")
            distance_metric = "cosine"

        threshold = find_threshold(model_name=model_name, distance_metric=distance_metric)

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
            else:
                raise ValueError(f"Unsupported distance metric: {distance_metric}")

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
) -> PostgresClient:
    if database_type == "postgres":
        db_client = PostgresClient(connection_details=connection_details, connection=connection)
        return db_client
    raise ValueError(f"Unsupported database type: {database_type}")


def __get_index(
    db_client: PostgresClient,
    model_name: str,
    detector_backend: str,
    align: bool,
    l2_normalize: bool,
) -> Any:
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


def build_index(
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    align: bool = True,
    l2_normalize: bool = False,
    database_type: str = "postgres",
    connection_details: Optional[Union[Dict[str, Any], str]] = None,
    connection: Any = None,
    batch_size: int = 1000,
) -> None:
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
    m = 32

    if index is None:
        base_index = faiss.IndexHNSWFlat(embedding_dim_size, m)
        index = faiss.IndexIDMap(base_index)

    tic = time.time()
    for i in range(0, len(vectors), batch_size):
        batch_ids = np.array(ids[i : i + batch_size], dtype="int64")
        batch_vectors = vectors[i : i + batch_size]
        index.add_with_ids(batch_vectors, batch_ids)

    toc = time.time()
    logger.info(f"Added {len(vectors)} embeddings to index in {toc - tic:.2f} seconds.")

    index_path = f"/tmp/{model_name}_{detector_backend}_{align}_{l2_normalize}.faiss"

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
