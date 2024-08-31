# built-in dependencies
import os
from typing import Optional
import zipfile
import bz2

# 3rd party dependencies
import gdown

# project dependencies
from deepface.commons import folder_utils
from deepface.commons.logger import Logger

logger = Logger()


def download_weights_if_necessary(
    file_name: str, source_url: str, compress_type: Optional[str] = None
) -> str:
    """
    Download the weights of a pre-trained model from external source if not downloaded yet.
    Args:
        file_name (str): target file name with extension
        source_url (url): source url to be downloaded
        compress_type (optional str): compress type e.g. zip or bz2
    Returns
        target_file (str): exact path for the target file
    """
    home = folder_utils.get_deepface_home()

    target_file = os.path.join(home, ".deepface/weights", file_name)

    if os.path.isfile(target_file):
        logger.debug(f"{file_name} is already available at {target_file}")
        return target_file

    try:
        logger.info(f"🔗 {file_name} will be downloaded from {source_url} to {target_file}...")

        if compress_type is None:
            gdown.download(source_url, target_file, quiet=False)
        elif compress_type is not None:
            gdown.download(source_url, f"{target_file}.{compress_type}", quiet=False)

    except Exception as err:
        exception_msg = (
            f"⛓️‍💥 Exception while downloading {file_name} from {source_url}. "
            f"You may consider to download it manually to {target_file}."
        )
        logger.error(exception_msg)
        raise ValueError(exception_msg) from err

    # uncompress downloaded file
    if compress_type == "zip":
        with zipfile.ZipFile(f"{target_file}.zip", "r") as zip_ref:
            zip_ref.extractall(os.path.join(home, ".deepface/weights"))
            logger.info(f"{target_file}.zip unzipped")
    elif compress_type == "bz2":
        bz2file = bz2.BZ2File(f"{target_file}.bz2")
        data = bz2file.read()
        with open(target_file, "wb") as f:
            f.write(data)
        logger.info(f"{target_file}.bz2 unzipped")

    return target_file
