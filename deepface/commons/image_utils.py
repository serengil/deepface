# built-in dependencies
import os
import io
from typing import TYPE_CHECKING, Generator, IO, List, Optional, Union, Tuple, cast, Any
import hashlib
import base64
from pathlib import Path

# 3rd party dependencies
import requests  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
import cv2

# project dependencies
from deepface.modules.exceptions import ImgNotFound, DataTypeError

# werkzeug comes with the api extra only, and it is required here for type hints only
if TYPE_CHECKING:
    from werkzeug.datastructures import FileStorage


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
IMAGE_FORMATS = {"jpeg", "png"}

# file signatures (magic bytes) of the supported image formats
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def detect_image_format(header: bytes) -> Optional[str]:
    """
    Detect image format from its content rather than its file extension
    Args:
        header (bytes): leading bytes of the image (at least 8 bytes)
    Returns:
        format (str): "jpeg" or "png", or None if it is not a supported image
    """
    if header.startswith(JPEG_MAGIC):
        return "jpeg"
    if header.startswith(PNG_MAGIC):
        return "png"
    return None


def list_images(path: str) -> List[str]:
    """
    List images in a given path
    Args:
        path (str): path's location
    Returns:
        images (list): list of exact image paths
    """
    return list(yield_images(path))


def yield_images(path: str) -> Generator[str, None, None]:
    """
    Yield images in a given path
    Args:
        path (str): path's location
    Yields:
        image (str): image path
    """
    for r, _, f in os.walk(path):
        for file in f:
            if os.path.splitext(file)[1].lower() not in IMAGE_EXTS:
                continue
            exact_path = os.path.join(r, file)
            if not os.path.isfile(exact_path):
                continue
            try:
                with open(exact_path, "rb") as img_file:
                    image_format = detect_image_format(img_file.read(8))
            except FileNotFoundError:
                # The entry may have disappeared since os.walk listed it.
                continue
            if image_format in IMAGE_FORMATS:
                yield exact_path


def find_image_hash(file_path: str) -> str:
    """
    Find the hash of given image file with its properties
        finding the hash of image content is costly operation
    Args:
        file_path (str): exact image path
    Returns:
        hash (str): digest with sha1 algorithm
    """
    file_stats = os.stat(file_path)

    # some properties
    file_size = file_stats.st_size
    creation_time = file_stats.st_ctime
    modification_time = file_stats.st_mtime

    properties = f"{file_size}-{creation_time}-{modification_time}"

    hasher = hashlib.sha1()
    hasher.update(properties.encode("utf-8"))
    return hasher.hexdigest()


def load_image(
    img: Union[str, NDArray[Any], IO[bytes]],
) -> Tuple[NDArray[Any], str]:
    """
    Load image from path, url, file object, base64 or numpy array.
    Args:
        img: a path, url, file object, base64 or numpy array.
    Returns:
        image (numpy array): the loaded image in BGR format
        image name (str): image name itself
    """

    # The image is already a numpy array
    if isinstance(img, np.ndarray):
        return img, "numpy array"

    # The image is an object that supports `.read`
    if hasattr(img, "read") and callable(img.read):
        if isinstance(img, io.StringIO):
            raise DataTypeError("img requires bytes and cannot be an io.StringIO object.")
        return load_image_from_io_object(cast(IO[bytes], img)), "io object"

    if isinstance(img, Path):
        img = str(img)

    if not isinstance(img, str):
        raise DataTypeError(f"img must be numpy array or str but it is {type(img)}")

    # The image is a base64 string
    if img.startswith("data:image/"):
        return load_image_from_base64(img), "base64 encoded string"

    # The image is a url
    if img.lower().startswith(("http://", "https://")):
        return load_image_from_web(url=img), img

    # The image is a path
    if not os.path.isfile(img):
        raise ImgNotFound(f"Confirm that {img} exists")

    # image must be a file on the system then

    # Read Unicode paths with Python: OpenCV's filename handling is platform-dependent.
    if not img.isascii():
        with open(img, "rb") as img_file:
            return load_image_from_io_object(img_file), img

    img_obj_bgr = cv2.imread(img)
    if img_obj_bgr is None:
        raise ValueError(f"Failed to decode image {img}")
    return img_obj_bgr, img


def load_image_from_io_object(obj: IO[bytes]) -> NDArray[Any]:
    """
    Load image from an object that supports being read
    Args:
        obj: a file like object.
    Returns:
        img (np.ndarray): The decoded image as a numpy array (OpenCV format).
    """
    try:
        obj.seek(0)
    except (AttributeError, TypeError, io.UnsupportedOperation):
        pass
    return _decode_image(obj.read())


def _decode_image(contents: bytes) -> NDArray[Any]:
    """Decode image bytes in BGR format, including a clear error for an empty body."""
    data = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError("Failed to decode image")
    return image


def load_image_from_base64(uri: str) -> NDArray[Any]:
    """
    Load image from base64 string.
    Args:
        uri: a base64 string.
    Returns:
        numpy array: the loaded image.
    """

    encoded_data_parts = uri.split(",")

    if len(encoded_data_parts) < 2:
        raise ValueError("format error in base64 encoded string")

    encoded_data = encoded_data_parts[1]
    decoded_bytes = base64.b64decode(encoded_data)

    # similar to find functionality, we are just considering these extensions
    # content type is safer option than file extension
    file_type = detect_image_format(decoded_bytes[:8])
    if file_type not in IMAGE_FORMATS:
        raise DataTypeError("Input image can be jpg or png, but it is neither")

    return _decode_image(decoded_bytes)


def load_image_from_file_storage(file: "FileStorage") -> NDArray[Any]:
    """
    Loads an image from a FileStorage object and decodes it into an OpenCV image.
    Args:
        file (FileStorage): The FileStorage object containing the image file.
    Returns:
        img (np.ndarray): The decoded image as a numpy array (OpenCV format).
    """
    return _decode_image(file.read())


def load_image_from_web(url: str) -> NDArray[Any]:
    """
    Loading an image from web
    Args:
        url: link for the image
    Returns:
        img (np.ndarray): equivalent to pre-loaded image from opencv (BGR format)
    """
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        # Decode HTTP content encodings (e.g. gzip) before decoding the image.
        response.raw.decode_content = True
        return _decode_image(response.raw.read())
