# built-in dependencies
import os

# 3rd party dependencies
import gdown

# project dependencies
from deepface.commons.logger import Logger

logger = Logger()


def download_external_file(file_name: str, exact_file_path: str, url: str) -> None:
    """
    Download an external file
    Args:
        file_name (str): file name with extension
        exact_file_path (str): exact location of the file with file name
        url (str): url to be downloaded
    Returns:
        None
    """
    if os.path.exists(exact_file_path) is False:
        logger.info(f"Downloading MiniFASNetV2 weights to {exact_file_path}")
        try:
            gdown.download(url, exact_file_path, quiet=False)
        except Exception as err:
            raise ValueError(
                f"Exception while downloading {file_name} from {url} to {exact_file_path}."
                "You may consider to download it and copy to the target destination."
            ) from err
