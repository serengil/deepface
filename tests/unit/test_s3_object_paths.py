import io
import sys
from unittest.mock import Mock

import pytest

from deepface.modules.filestore import S3FileStore


@pytest.fixture
def s3(monkeypatch):
    client = Mock()
    monkeypatch.setitem(sys.modules, "boto3", Mock(client=Mock(return_value=client)))
    return client


@pytest.mark.parametrize("key", ["a?b/c#d%20.jpg", "a#b/c?.png", "/gallery/face.jpg"])
def test_listed_s3_keys_round_trip(s3, key):
    s3.get_paginator.return_value.paginate.return_value = [{"Contents": [{"Key": key}]}]
    s3.get_object.return_value = {"Body": io.BytesIO(b"pixels")}
    store = S3FileStore("s3://faces")

    (path,) = store.list_images()
    assert store.read_bytes(path) == b"pixels"
    s3.get_object.assert_called_once_with(Bucket="faces", Key=key)


def test_prefix_and_joined_path_keep_reserved_key_characters(s3):
    store = S3FileStore("s3://faces/gallery#1?x")
    store.write_bytes(store.join("a%20?#.jpg"), b"data")
    s3.put_object.assert_called_once_with(
        Bucket="faces", Key="gallery#1?x/a%20?#.jpg", Body=b"data"
    )


@pytest.mark.parametrize("fail", [False, True])
def test_response_stream_is_closed_after_read(s3, fail):
    body = Mock()
    error = OSError("interrupted")
    body.read.side_effect = error if fail else None
    body.read.return_value = b"pixels"
    s3.get_object.return_value = {"Body": body}
    store = S3FileStore("s3://faces")
    if fail:
        with pytest.raises(OSError) as result:
            store.read_bytes(store.join("photo.jpg"))
        assert result.value is error
    else:
        assert store.read_bytes(store.join("photo.jpg")) == b"pixels"
    body.close.assert_called_once_with()
