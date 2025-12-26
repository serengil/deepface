# built-in dependencies
import traceback
from typing import Optional, Union, Dict, Any, Tuple, List

# 3rd party dependencies
from numpy.typing import NDArray

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=broad-except, too-many-positional-arguments


def represent(
    img_path: Union[str, NDArray[Any]],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
) -> Tuple[Dict[str, Any], int]:
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs
        return result, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: Union[str, NDArray[Any]],
    img2_path: Union[str, NDArray[Any]],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
) -> Tuple[Dict[str, Any], int]:
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return obj, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


def analyze(
    img_path: Union[str, NDArray[Any]],
    actions: List[str],
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
) -> Tuple[Dict[str, Any], int]:
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = demographies
        return result, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400


def register(
    img: Union[str, NDArray[Any]],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    l2_normalize: bool,
    expand_percentage: int,
    normalization: str,
    anti_spoofing: bool,
    img_name: Optional[str],
    database_type: str,
    connection_details: str,
) -> Tuple[Dict[str, Any], int]:
    try:
        return (
            DeepFace.register(
                img=img,
                img_name=img_name,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
                l2_normalize=l2_normalize,
                expand_percentage=expand_percentage,
                normalization=normalization,
                anti_spoofing=anti_spoofing,
                database_type=database_type,
                connection_details=connection_details,
            ),
            200,
        )
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while registering: {str(err)} - {tb_str}"}, 400


def search(
    img: Union[str, NDArray[Any]],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    l2_normalize: bool,
    expand_percentage: int,
    normalization: str,
    anti_spoofing: bool,
    similarity_search: bool,
    k: Optional[int],
    database_type: str,
    connection_details: str,
    search_method: str,
) -> Tuple[Dict[str, Any], int]:
    try:
        result = {}
        dfs = DeepFace.search(
            img=img,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align,
            l2_normalize=l2_normalize,
            expand_percentage=expand_percentage,
            normalization=normalization,
            anti_spoofing=anti_spoofing,
            similarity_search=similarity_search,
            k=k,
            database_type=database_type,
            connection_details=connection_details,
            search_method=search_method,
        )

        result["results"] = [df.to_dict(orient="records") for df in dfs]
        return result, 200

    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while searching: {str(err)} - {tb_str}"}, 400


def build_index(
    model_name: str,
    detector_backend: str,
    align: bool,
    l2_normalize: bool,
    database_type: str,
    connection_details: str,
) -> Tuple[Dict[str, Any], int]:
    try:
        DeepFace.build_index(
            model_name=model_name,
            detector_backend=detector_backend,
            align=align,
            l2_normalize=l2_normalize,
            database_type=database_type,
            connection_details=connection_details,
        )
        return {"message": "Index built successfully"}, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while building index: {str(err)} - {tb_str}"}, 400
