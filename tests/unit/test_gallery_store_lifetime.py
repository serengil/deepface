"""Real FTP sessions must not be retained by failed gallery searches."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import numpy as np
import pytest

from deepface.modules import modeling, recognition
from deepface.modules.exceptions import EmptyDatasource, ImgNotFound, PathNotFound
from deepface.modules.filestore import S3FileStore, build_file_store

CACHE_NAME = "ds_model_sface_detector_skip_aligned_normalization_base_expand_0.npz"


@pytest.fixture
def ftp_gallery(tmp_path):
    pytest.importorskip("pyftpdlib")
    from pyftpdlib.authorizers import DummyAuthorizer
    from pyftpdlib.handlers import FTPHandler
    from pyftpdlib.ioloop import IOLoop
    from pyftpdlib.servers import FTPServer

    disconnected = []

    class Handler(FTPHandler):
        def on_connect(self):
            self.disconnected = Event()
            disconnected.append(self.disconnected)

        def on_disconnect(self):
            self.disconnected.set()

    authorizer = DummyAuthorizer()
    authorizer.add_user("gallery", "test-password", str(tmp_path), perm="elradfmwMT")
    Handler.authorizer = authorizer
    server = FTPServer(("127.0.0.1", 0), Handler, ioloop=IOLoop())
    port = server.socket.getsockname()[1]
    thread = Thread(
        target=server.serve_forever,
        kwargs={"timeout": 0.01, "blocking": True, "handle_exit": False},
        daemon=True,
    )
    thread.start()
    try:
        yield f"ftp://gallery:test-password@127.0.0.1:{port}/", tmp_path, disconnected
    finally:
        server.close_all()
        thread.join(timeout=5)


def cached_gallery(root, identity):
    (root / CACHE_NAME).write_bytes(
        recognition.__serialize_npz(
            [
                {
                    "identity": identity,
                    "hash": "cached",
                    "embedding": [1.0, 0.0],
                    "target_x": 0,
                    "target_y": 0,
                    "target_w": 8,
                    "target_h": 8,
                }
            ]
        )
    )


@pytest.fixture
def inference(monkeypatch):
    class FixedEmbedding:
        input_shape = (8, 8)
        output_shape = 2

        def forward(self, _images):
            return [1.0, 0.0]

    monkeypatch.setattr(modeling, "build_model", lambda **_kwargs: FixedEmbedding())


def find(gallery, **kwargs):
    options = {
        "img_path": np.full((8, 8, 3), 128, dtype=np.uint8),
        "db_path": gallery,
        "model_name": "SFace",
        "detector_backend": "skip",
        "enforce_detection": False,
        "align": True,
        "silent": True,
        "return_type": "dict",
    }
    options.update(kwargs)
    return recognition.find(**options)


@pytest.mark.parametrize(
    "failure, error",
    [("query", ImgNotFound), ("empty_gallery", EmptyDatasource), ("cache", ValueError)],
)
def test_find_releases_ftp_while_failure_traceback_is_retained(ftp_gallery, failure, error):
    uri, root, disconnected = ftp_gallery
    options = {}
    if failure == "query":
        options["img_path"] = str(root / "missing-query.png")
    elif failure == "cache":
        (root / CACHE_NAME).write_bytes(b"not an NPZ archive")
        options["refresh_database"] = False

    with pytest.raises(error) as caught:
        find(uri, **options)

    # Logging often retains an exception and its frames, including the local file store.
    assert caught.value.__traceback__ is not None
    assert len(disconnected) == 1
    assert disconnected[0].wait(2), "failed find kept its FTP control connection open"


def test_failed_store_validation_releases_the_connection(ftp_gallery):
    uri, _root, disconnected = ftp_gallery

    with pytest.raises(PathNotFound) as caught:
        build_file_store(uri + "missing-directory")

    assert caught.value.__traceback__ is not None
    assert len(disconnected) == 1
    assert disconnected[0].wait(2), "failed validation kept its FTP connection open"


@pytest.mark.parametrize("return_type", ["dict", "pandas"])
def test_successful_ftp_search_preserves_results_and_releases_connection(
    ftp_gallery, inference, return_type
):
    uri, root, disconnected = ftp_gallery
    identity = uri + "cached-person.png"
    cached_gallery(root, identity)

    results = find(uri, refresh_database=False, return_type=return_type)

    rows = results[0] if return_type == "dict" else results[0].to_dict("records")
    assert len(rows) == 1
    assert rows[0]["identity"] == identity
    assert rows[0]["distance"] == 0
    assert len(disconnected) == 1
    assert disconnected[0].wait(2)


@pytest.mark.parametrize("return_type", ["dict", "pandas"])
def test_local_gallery_keeps_its_existing_results(tmp_path, inference, return_type):
    cached_gallery(tmp_path, "local-person.png")

    results = find(str(tmp_path), refresh_database=False, return_type=return_type)

    rows = results[0] if return_type == "dict" else results[0].to_dict("records")
    assert rows[0]["identity"] == "local-person.png"
    assert rows[0]["distance"] == 0


def test_s3_sdk_close_releases_actual_http_pool_connections(monkeypatch):
    boto3 = pytest.importorskip("boto3")

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    client = boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{server.server_port}",
        aws_access_key_id="local-test-key",
        aws_secret_access_key="local-test-secret",
        region_name="us-east-1",
    )
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)
    try:
        store = S3FileStore("s3://gallery")
        store.validate()
        pools = client._endpoint.http_session._manager.pools
        assert len(pools) == 1

        store.close()

        assert len(pools) == 0
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
