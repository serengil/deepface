# built-in dependencies
import os
import io
from typing import List, Union, Tuple
import hashlib
import base64
from pathlib import Path

# 3rd party dependencies
import requests
import numpy as np
import cv2
from PIL import Image


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
            exact_path = os.path.join(r, file)

            _, ext = os.path.splitext(exact_path)
            ext_lower = ext.lower()

            if ext_lower not in {".jpg", ".jpeg", ".png"}:
                continue

            with Image.open(exact_path) as img:  # lazy
                if img.format.lower() in ["jpeg", "png"]:
                    images.append(exact_path)
    return images


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


def load_image(img: Union[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """
    Load image from path, url, base64 or numpy array.
    Args:
        img: a path, url, base64 or numpy array.
    Returns:
        image (numpy array): the loaded image in BGR format
        image name (str): image name itself
    """

    # The image is already a numpy array
    if isinstance(img, np.ndarray):
        return img, "numpy array"

    if isinstance(img, Path):
        img = str(img)

    if not isinstance(img, str):
        raise ValueError(f"img must be numpy array or str but it is {type(img)}")

    # The image is a base64 string
    if img.startswith("data:image/"):
        return load_image_from_base64(img), "base64 encoded string"

    # The image is a url
    if img.lower().startswith("http://") or img.lower().startswith("https://"):
        return load_image_from_web(url=img), img

    # The image is a path
    if os.path.isfile(img) is not True:
        raise ValueError(f"Confirm that {img} exists")

    # image must be a file on the system then

    # image name must have english characters
    if img.isascii() is False:
        raise ValueError(f"Input image must not have non-english characters - {img}")

    img_obj_bgr = cv2.imread(img)
    # img_obj_rgb = cv2.cvtColor(img_obj_bgr, cv2.COLOR_BGR2RGB)
    return img_obj_bgr, img


def load_image_from_base64(uri: str) -> np.ndarray:
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
        if file_type not in ["jpeg", "png"]:
            raise ValueError(f"input image can be jpg or png, but it is {file_type}")

    nparr = np.fromstring(decoded_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr


def load_image_from_web(url: str) -> np.ndarray:
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
    return img
