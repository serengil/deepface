# built-in dependencies
import os
import json
from concurrent import futures
from typing import Any, Dict, Tuple, Union

# 3rd party dependencies
import cv2
import grpc
import numpy as np
from numpy.typing import NDArray
from google.protobuf.struct_pb2 import Struct
from dotenv import load_dotenv

# load environment variables from .env first things first
load_dotenv()

# pylint: disable=wrong-import-position
# project dependencies
from deepface import __version__
from deepface.api.src.app import load_models_on_startup
from deepface.api.src.modules.core import service
from deepface.api.src.dependencies.variables import Variables
from deepface.api.src.dependencies.container import Container
from deepface.commons.logger import Logger

try:
    from deepface.api.proto import deepface_pb2, deepface_pb2_grpc
except ImportError as import_err:
    raise ImportError(
        "gRPC stubs not found. Run `make grpc` in the root of the repo to generate them."
    ) from import_err

logger = Logger()

# default 4MB message size limit is too small for high resolution images
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024

# grpc status codes corresponding to http status codes returned by service layer
STATUS_CODES = {
    400: grpc.StatusCode.INVALID_ARGUMENT,
    401: grpc.StatusCode.UNAUTHENTICATED,
    500: grpc.StatusCode.INTERNAL,
}


def load_image(img: Any) -> Union[str, NDArray[Any]]:
    """
    Extracts an image from Image message of the request.

    Args:
        img (deepface_pb2.Image): image message either having raw bytes or a path.

    Returns:
        img (str or np.ndarray): Given image detail (base64 encoded string, image path or url)
            or the decoded image as a numpy array.
    """
    source = img.WhichOneof("source")
    if source == "content":
        image = cv2.imdecode(np.frombuffer(img.content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    if source == "path":
        if not img.path:
            raise ValueError("empty image path passed")
        return str(img.path)
    raise ValueError("image is not set in the request")


def value_or(request: Any, field: str, default: Any) -> Any:
    """Returns the field of request if it is set explicitly, otherwise the default"""
    return getattr(request, field) if request.HasField(field) else default


def to_struct(obj: Dict[str, Any]) -> Struct:
    """Converts a json serializable dictionary into protobuf struct"""

    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    struct = Struct()
    struct.update(json.loads(json.dumps(obj, default=default)))
    return struct


# pylint: disable=invalid-name, broad-except
class DeepFaceServicer(deepface_pb2_grpc.DeepFaceServiceServicer):  # type: ignore[misc]
    def __init__(self, variables: Variables, container: Container) -> None:
        self.variables = variables
        self.container = container

    def respond(self, context: Any, obj: Dict[str, Any], status_code: int) -> Any:
        logger.debug(obj)
        if status_code != 200:
            context.abort(
                STATUS_CODES.get(status_code, grpc.StatusCode.UNKNOWN),
                str(obj.get("error") or obj.get("exception") or obj.get("message") or obj),
            )
        return deepface_pb2.DeepFaceResponse(result=to_struct(obj))

    def validate(self, context: Any, requires_db: bool = False) -> None:
        metadata = dict(context.invocation_metadata())
        if not self.container.auth_service.validate(
            {"Authorization": metadata.get("authorization")}
        ):
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing authentication token"
            )

        if requires_db and self.variables.conection_details is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Database connection details must be provided in `DEEPFACE_CONNECTION_DETAILS`"
                " environment variables",
            )

    def load_image(self, context: Any, img: Any) -> Union[str, NDArray[Any]]:
        try:
            return load_image(img)
        except Exception as err:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(err))
            raise  # unreachable, abort raises

    def Represent(self, request: Any, context: Any) -> Any:
        self.validate(context)
        img = self.load_image(context, request.img)

        obj, status_code = service.represent(
            img_path=img,
            model_name=value_or(request, "model_name", "VGG-Face"),
            detector_backend=value_or(request, "detector_backend", "opencv"),
            enforce_detection=value_or(request, "enforce_detection", True),
            align=value_or(request, "align", True),
            anti_spoofing=value_or(request, "anti_spoofing", False),
            max_faces=value_or(request, "max_faces", None),
        )
        return self.respond(context, obj, status_code)

    def Verify(self, request: Any, context: Any) -> Any:
        self.validate(context)
        img1 = self.load_image(context, request.img1)
        img2 = self.load_image(context, request.img2)

        obj, status_code = service.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=value_or(request, "model_name", "VGG-Face"),
            detector_backend=value_or(request, "detector_backend", "opencv"),
            distance_metric=value_or(request, "distance_metric", "cosine"),
            align=value_or(request, "align", True),
            enforce_detection=value_or(request, "enforce_detection", True),
            anti_spoofing=value_or(request, "anti_spoofing", False),
        )
        return self.respond(context, obj, status_code)

    def Analyze(self, request: Any, context: Any) -> Any:
        self.validate(context)
        img = self.load_image(context, request.img)

        obj, status_code = service.analyze(
            img_path=img,
            actions=list(request.actions) or ["age", "gender", "emotion", "race"],
            detector_backend=value_or(request, "detector_backend", "opencv"),
            enforce_detection=value_or(request, "enforce_detection", True),
            align=value_or(request, "align", True),
            anti_spoofing=value_or(request, "anti_spoofing", False),
        )
        return self.respond(context, obj, status_code)

    def Register(self, request: Any, context: Any) -> Any:
        self.validate(context, requires_db=True)
        img = self.load_image(context, request.img)

        obj, status_code = service.register(
            img=img,
            img_name=value_or(request, "img_name", None),
            model_name=value_or(request, "model_name", "VGG-Face"),
            detector_backend=value_or(request, "detector_backend", "opencv"),
            enforce_detection=value_or(request, "enforce_detection", True),
            align=value_or(request, "align", True),
            l2_normalize=value_or(request, "l2_normalize", False),
            expand_percentage=value_or(request, "expand_percentage", 0),
            normalization=value_or(request, "normalization", "base"),
            anti_spoofing=value_or(request, "anti_spoofing", False),
            database_type=self.variables.database_type,
            connection_details=self.variables.conection_details,  # type: ignore[arg-type]
        )

        if status_code == 200:
            logger.info("An image has been registered to the database.")
        else:
            logger.error("An error occurred while registering an image to the database.")

        return self.respond(context, obj, status_code)

    def Search(self, request: Any, context: Any) -> Any:
        self.validate(context, requires_db=True)
        img = self.load_image(context, request.img)

        obj, status_code = service.search(
            img=img,
            model_name=value_or(request, "model_name", "VGG-Face"),
            detector_backend=value_or(request, "detector_backend", "opencv"),
            enforce_detection=value_or(request, "enforce_detection", True),
            align=value_or(request, "align", True),
            distance_metric=value_or(request, "distance_metric", "cosine"),
            l2_normalize=value_or(request, "l2_normalize", False),
            database_type=self.variables.database_type,
            connection_details=self.variables.conection_details,  # type: ignore[arg-type]
            search_method=value_or(request, "search_method", "exact"),
            expand_percentage=value_or(request, "expand_percentage", 0),
            normalization=value_or(request, "normalization", "base"),
            anti_spoofing=value_or(request, "anti_spoofing", False),
            similarity_search=value_or(request, "similarity_search", False),
            k=value_or(request, "k", None),
        )
        return self.respond(context, obj, status_code)

    def BuildIndex(self, request: Any, context: Any) -> Any:
        self.validate(context, requires_db=True)

        obj, status_code = service.build_index(
            model_name=value_or(request, "model_name", "VGG-Face"),
            detector_backend=value_or(request, "detector_backend", "opencv"),
            align=value_or(request, "align", True),
            l2_normalize=value_or(request, "l2_normalize", False),
            database_type=self.variables.database_type,
            connection_details=self.variables.conection_details,  # type: ignore[arg-type]
        )
        return self.respond(context, obj, status_code)


def create_server(port: int = 50051, max_workers: int = 1) -> Tuple[grpc.Server, int]:
    """
    Creates a grpc server serving DeepFace API. It is not started yet.

    Args:
        port (int): port to listen. 0 picks a free port.
        max_workers (int): number of threads handling requests concurrently.

    Returns:
        server (grpc.Server): created grpc server
        port (int): port the server is bound to
    """
    variables = Variables()
    container = Container(variables=variables)

    load_models_on_startup(variables)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
        ],
    )
    deepface_pb2_grpc.add_DeepFaceServiceServicer_to_server(  # type: ignore[no-untyped-call]
        DeepFaceServicer(variables=variables, container=container), server
    )
    bound_port = server.add_insecure_port(f"[::]:{port}")
    return server, bound_port


def serve() -> None:
    port = int(os.getenv("DEEPFACE_GRPC_PORT", "50051"))
    max_workers = int(os.getenv("DEEPFACE_GRPC_MAX_WORKERS", "1"))
    server, bound_port = create_server(port=port, max_workers=max_workers)
    server.start()
    logger.info(f"Welcome to DeepFace gRPC API v{__version__}! Listening on port {bound_port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
