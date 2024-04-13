# built-in dependencies
import os
from typing import List
import hashlib

# 3rd party dependencies
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


def find_hash_of_file(file_path: str) -> str:
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
