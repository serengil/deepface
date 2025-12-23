# built-in dependencies
from typing import Any, Dict, IO, List, Union, Optional, cast
import uuid


# 3rd party dependencies
import pandas as pd
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


# pylint: disable=too-many-positional-arguments
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

    if database_type == "postgres":
        db_client = PostgresClient(connection_details=connection_details, connection=connection)
    else:
        raise ValueError(f"Unsupported database type: {database_type}")

    results = get_embeddings(
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
) -> Union[List[pd.DataFrame], List[List[Dict[str, Any]]]]:
    if database_type == "postgres":
        db_client = PostgresClient(connection_details=connection_details, connection=connection)
    else:
        raise ValueError(f"Unsupported database type: {database_type}")

    results = get_embeddings(
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

    dfs = []
    for result in results:
        target_embedding = cast(List[float], result["embedding"])

        df = pd.DataFrame(source_embeddings)
        df["target_embedding"] = [target_embedding for _ in range(len(df))]

        if distance_metric == "cosine":
            df["distance"] = df.apply(
                lambda row: find_cosine_distance(row["embedding"], row["target_embedding"]), axis=1
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
            threshold = find_threshold(model_name=model_name, distance_metric=distance_metric)
            df = df[df["distance"] <= threshold]

        if k is not None and k > 0:
            df = df.nsmallest(k, "distance")

        df = df.sort_values(by="distance", ascending=True).reset_index(drop=True)

        dfs.append(df)

    return dfs


def get_embeddings(
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
