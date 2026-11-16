"""Exercise weight-source checks through real HTTP, Requests and gdown downloads."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

import pytest

from deepface.commons import weight_utils


@pytest.fixture
def weight_server(tmp_path, monkeypatch):
    routes = {}
    events = []

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond()

        def do_GET(self):
            self.respond()

        def respond(self):
            path = urlsplit(self.path).path
            events.append((self.command, self.path))
            if path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/weights?signed-method=GET")
                self.end_headers()
                return
            head_status, get_status, body = routes[path]
            status = head_status if self.command == "HEAD" else get_status
            payload = body if status == 200 else b"HTTP error, not model weights"
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    weights = tmp_path / ".deepface" / "weights"
    weights.mkdir(parents=True)
    monkeypatch.setattr(weight_utils.folder_utils, "get_deepface_home", lambda: str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", routes, events, weights
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("head_status", [200, 401, 403, 404, 405, 501])
def test_get_authorized_weights_download_when_head_is_rejected(weight_server, head_status):
    base, routes, events, _weights = weight_server
    payload = b"\x00\x01local model weights\xff"
    routes["/weights"] = (head_status, 200, payload)

    target = weight_utils.download_weights_if_necessary(
        "model.bin", base + "/weights?signed-method=GET"
    )

    assert Path(target).read_bytes() == payload
    assert ("GET", "/weights?signed-method=GET") in events


@pytest.mark.parametrize("get_status", [403, 404, 500])
def test_head_unsupported_error_pages_do_not_replace_backup_weights(weight_server, get_status):
    base, routes, events, _weights = weight_server
    routes["/primary"] = (405, get_status, b"not used")
    payload = b"actual backup model weights"
    routes["/backup"] = (200, 200, payload)

    target = weight_utils.download_weights_if_necessary(
        "model.bin", [base + "/primary", base + "/backup"]
    )

    assert Path(target).read_bytes() == payload
    assert ("GET", "/backup") in events


def test_failed_get_still_raises_without_publishing_a_cache_file(weight_server):
    base, routes, _events, weights = weight_server
    routes["/missing"] = (404, 404, b"not used")

    with pytest.raises(ValueError, match="while downloading"):
        weight_utils.download_weights_if_necessary("missing.bin", base + "/missing")

    assert not (weights / "missing.bin").exists()


def test_redirected_get_authorization_is_preserved(weight_server):
    base, routes, events, _weights = weight_server
    payload = b"redirected model weights"
    routes["/weights"] = (403, 200, payload)

    target = weight_utils.download_weights_if_necessary("model.bin", base + "/redirect")

    assert Path(target).read_bytes() == payload
    assert ("GET", "/weights?signed-method=GET") in events


def test_existing_weight_cache_avoids_requests(weight_server):
    base, _routes, events, weights = weight_server
    target = weights / "model.bin"
    target.write_bytes(b"cached model weights")

    result = weight_utils.download_weights_if_necessary("model.bin", base + "/not-contacted")

    assert result == str(target)
    assert target.read_bytes() == b"cached model weights"
    assert events == []
