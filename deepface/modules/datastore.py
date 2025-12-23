# built-in dependencies
from typing import Any, Dict, IO, List, Union, Optional
import uuid


# 3rd party dependencies
from numpy.typing import NDArray

# project dependencies
from deepface.modules.database.postgres import PostgresClient
from deepface.modules.database.types import EmbeddingRecord
from deepface.modules.representation import represent
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
        return_face=True,
    )

    if len(results) == 0:
        raise ValueError("No embeddings were detected in the provided image(s).")

    logger.debug(f"Registering {len(results)} embeddings to {database_type} database.")

    flat_results: List[Dict[str, Any]] = []
    for result in results:
        if isinstance(result, dict):
            # result = cast(Dict[str, Any], result)
            flat_results.append(result)
        elif isinstance(result, list):
            # result = cast(List[Dict[str, Any]], result)
            for inner_result in result:
                flat_results.append(inner_result)

    embedding_records: List[EmbeddingRecord] = []
    for result in flat_results:
        img_identifier = img_name or (
            img
            if isinstance(img, str) and img.endswith((".jpg", ".jpeg", ".png"))
            else str(uuid.uuid4())
        )

        embedding_record: EmbeddingRecord = {
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
