# built-in dependencies
from unittest.mock import patch

# 3rd party dependencies
import pytest

# project dependencies
from deepface.modules import filestore
from deepface.modules.exceptions import PathNotFound
from deepface.commons.logger import Logger

logger = Logger()


@pytest.mark.parametrize(
    "db_path",
    ["dataset", "/home/user/my_db", "C:/my_db", "C://my_db", "C:\\my_db", "c:\\\\my_db"],
)
def test_local_paths_resolve_to_local_store(db_path):
    with patch.object(filestore.LocalFileStore, "validate"):
        store = filestore.build_file_store(db_path)
    assert isinstance(store, filestore.LocalFileStore)
    logger.info(f"✅ {db_path} resolved to local file store")


def test_missing_windows_path_raises_path_not_found():
    with pytest.raises(PathNotFound):
        filestore.build_file_store("Z://not_existing_db")
    logger.info("✅ missing windows path raises PathNotFound")


def test_ftp_path_resolves_to_ftp_store():
    with patch.object(filestore.FtpFileStore, "validate"):
        store = filestore.build_file_store("ftp://user:pass@localhost:2121/my_db")
    assert isinstance(store, filestore.FtpFileStore)
    assert store.join("x.pkl") == "ftp://localhost:2121/my_db/x.pkl"
    logger.info("✅ ftp path resolved to ftp file store")


def test_unsupported_scheme():
    with pytest.raises(ValueError, match="Unsupported db_path scheme"):
        filestore.build_file_store("http://localhost/my_db")
    logger.info("✅ unsupported scheme raises ValueError")
