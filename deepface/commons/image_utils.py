# built-in dependencies
import os
import io
from typing import Generator, IO, List, Union, Tuple, cast, Any
import hashlib
import base64
from pathlib import Path

# 3rd party dependencies
import requests  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
import cv2
from PIL import Image
from werkzeug.datastructures import FileStorage


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PIL_EXTS = {"jpeg", "png"}


def list_images(path: str) -> List[str]:
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
            if os.path.splitext(file)[1].lower() in IMAGE_EXTS:
                exact_path = os.path.join(r, file)
                with Image.open(exact_path) as img:  # lazy
                    if img.format.lower() in PIL_EXTS:
                        images.append(exact_path)
    return images


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
            if os.path.splitext(file)[1].lower() in IMAGE_EXTS:
                exact_path = os.path.join(r, file)
                with Image.open(exact_path) as img:  # lazy
                    if img.format.lower() in PIL_EXTS:
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
            raise ValueError("img requires bytes and cannot be an io.StringIO object.")
        return load_image_from_io_object(cast(IO[bytes], img)), "io object"

    if isinstance(img, Path):
        img = str(img)

    if not isinstance(img, str):
        raise ValueError(f"img must be numpy array or str but it is {type(img)}")

    # The image is a base64 string
    if img.startswith("data:image/"):
        return load_image_from_base64(img), "base64 encoded string"

    # The image is a url
    if img.lower().startswith(("http://", "https://")):
        return load_image_from_web(url=img), img

    # The image is a path
    if not os.path.isfile(img):
        raise ValueError(f"Confirm that {img} exists")

    # image must be a file on the system then

    # image name must have english characters
    if not img.isascii():
        raise ValueError(f"Input image must not have non-english characters - {img}")

    img_obj_bgr = cv2.imread(img)
    # img_obj_rgb = cv2.cvtColor(img_obj_bgr, cv2.COLOR_BGR2RGB)
    return cast(NDArray[Any], img_obj_bgr), img


def load_image_from_io_object(obj: IO[bytes]) -> NDArray[Any]:
    """
    Load image from an object that supports being read
    Args:
        obj: a file like object.
    Returns:
        img (np.ndarray): The decoded image as a numpy array (OpenCV format).
    """
    try:
        _ = obj.seek(0)
    except (AttributeError, TypeError, io.UnsupportedOperation):
        seekable = False
        obj = io.BytesIO(obj.read())
    else:
        seekable = True
    try:
        nparr = np.frombuffer(obj.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    finally:
        if not seekable:
            obj.close()


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
    with Image.open(io.BytesIO(decoded_bytes)) as img:
        file_type = img.format.lower()
        if file_type not in {"jpeg", "png"}:
            raise ValueError(f"Input image can be jpg or png, but it is {file_type}")

    nparr = np.frombuffer(decoded_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cast(NDArray[Any], img_bgr)


def load_image_from_file_storage(file: FileStorage) -> NDArray[Any]:
    """
    Loads an image from a FileStorage object and decodes it into an OpenCV image.
    Args:
        file (FileStorage): The FileStorage object containing the image file.
    Returns:
        img (np.ndarray): The decoded image as a numpy array (OpenCV format).
    """
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


def load_image_from_web(url: str) -> NDArray[Any]:
    """
    Loading an image from web
    Args:
        url: link for the image
    Returns:
        img (np.ndarray): equivalent to pre-loaded image from opencv (BGR format)
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return cast(NDArray[Any], img)
