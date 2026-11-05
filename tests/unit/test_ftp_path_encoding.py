from unittest.mock import Mock
from urllib.parse import urlparse

import pytest

from deepface.modules.filestore import FtpFileStore


@pytest.mark.parametrize("name", ["face#1.jpg", "face?1.png", "face%20.jpg", "face name.jpg"])
def test_joined_ftp_path_reads_the_original_filename(name):
    store = FtpFileStore("ftp://user:secret@example.com:2121/gallery")
    ftp = Mock()
    ftp.retrbinary.side_effect = lambda command, write: write(b"image bytes")
    store.ftp = ftp

    uri = store.join(name)
    assert store.read_bytes(uri) == b"image bytes"

    ftp.retrbinary.assert_called_once()
    assert ftp.retrbinary.call_args.args[0] == f"RETR /gallery/{name}"
    assert not urlparse(uri).query
    assert not urlparse(uri).fragment
    assert "secret" not in uri
    assert " " not in uri


def test_listed_nested_ftp_images_keep_reserved_path_characters():
    store = FtpFileStore("ftp://example.com/gallery%23new")
    ftp = Mock()
    ftp.mlsd.side_effect = [
        [("nested?folder", {"type": "dir"}), ("not-an-image.txt", {"type": "file"})],
        [("face#%20.png", {"type": "file"})],
    ]
    ftp.retrbinary.side_effect = lambda command, write: write(b"image bytes")
    store.ftp = ftp

    images = store.list_images()

    assert images == ["ftp://example.com/gallery%23new/nested%3Ffolder/face%23%2520.png"]
    assert store.read_bytes(images[0]) == b"image bytes"
    assert ftp.retrbinary.call_args.args[0] == "RETR /gallery#new/nested?folder/face#%20.png"
