"""Exercise Pinecone SDK connection lifetimes against a loopback HTTP server."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

pinecone = pytest.importorskip("pinecone")

from deepface.modules import datastore
from deepface.modules.database.pinecone import PineconeClient

CRITERIA = dict(model_name="Facenet", detector_backend="skip", aligned=True, l2_normalized=False)
INDEX_NAME = "embeddings-facenet-skip-aligned-raw"


@pytest.fixture
def endpoint(monkeypatch):
    state = {"failure": None, "missing": False, "empty": False, "requests": [], "active": set()}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            with lock:
                state["active"].add(self.connection)

        def finish(self):
            try:
                super().finish()
            finally:
                with lock:
                    state["active"].discard(self.connection)

        def log_message(self, *args):
            pass

        def reply(self, value, status=200):
            raw = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def handle_request(self):
            parsed = urlparse(self.path)
            path = parsed.path
            payload = {}
            if self.command == "POST":
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.command, path, payload))
            if state["failure"] == path:
                self.reply({"code": 3, "message": "injected data-plane error"}, 400)
                return
            description = {
                "name": INDEX_NAME,
                "dimension": 128,
                "metric": "euclidean",
                "host": state["url"],
                "status": {"ready": True, "state": "Ready"},
                "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
                "vector_type": "dense",
                "deletion_protection": "disabled",
                "schema": {
                    "fields": {
                        "values": {"type": "dense_vector", "dimension": 128, "metric": "euclidean"}
                    }
                },
                "deployment": {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
            }
            if path == "/indexes":
                self.reply({"indexes": [] if state["missing"] else [description]})
            elif path.startswith("/indexes/"):
                if state["missing"]:
                    self.reply({"code": 5, "message": "not found"}, 404)
                else:
                    self.reply(description)
            elif path == "/vectors/upsert":
                self.reply({"upsertedCount": len(payload["vectors"])})
            elif path == "/query":
                self.reply(
                    {
                        "matches": [
                            {"id": "known", "score": 0.25, "metadata": {"img_name": "known.png"}}
                        ],
                        "namespace": "",
                    }
                )
            elif path == "/vectors/list":
                self.reply(
                    {
                        "vectors": [] if state["empty"] else [{"id": "known"}],
                        "namespace": "",
                    }
                )
            elif path == "/vectors/fetch":
                ids = parse_qs(parsed.query).get("ids", [])
                self.reply(
                    {
                        "vectors": {
                            identity: {
                                "id": identity,
                                "values": [0.5] * 128,
                                "metadata": {"img_name": f"{identity}.png"},
                            }
                            for identity in ids
                        },
                        "namespace": "",
                    }
                )
            else:
                self.reply({"code": 5, "message": f"unexpected route: {path}"}, 404)

        do_GET = handle_request
        do_POST = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = pinecone.Pinecone(api_key=f"loopback-test-key-{server.server_port}", host=state["url"])
    original_factory = client.Index
    original_close = getattr(client, "close", None)
    state["supports_close"] = original_close is not None
    state["indexes"] = []
    state["client_close_calls"] = 0

    def index_factory(*_args, **_kwargs):
        # Use the fixture endpoint directly; SDK 8 caches name resolution globally.
        index = original_factory(host=state["url"])
        original_index_close = index.close
        item = {"index": index, "close_calls": 0}
        state["indexes"].append(item)

        def close_index():
            item["close_calls"] += 1
            original_index_close()

        monkeypatch.setattr(index, "close", close_index)
        return index

    def close_client():
        state["client_close_calls"] += 1
        original_close()

    monkeypatch.setattr(client, "Index", index_factory)
    if original_close is not None:
        monkeypatch.setattr(client, "close", close_client)
    state["client"] = client
    try:
        yield state
    finally:
        for item in state["indexes"]:
            item["index"].close()
        if original_close is not None:
            original_close()
        for connection in tuple(state["active"]):
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def operation(adapter, name):
    if name == "insert":
        return adapter.insert_embeddings(
            [
                dict(
                    **CRITERIA,
                    face=np.zeros((2, 2, 3), dtype=np.float32),
                    embedding=[0.5] * 128,
                    img_name="known.png",
                )
            ]
        )
    if name == "search":
        return adapter.search_by_vector(vector=[0.5] * 128, **CRITERIA)
    if name == "fetch":
        return adapter.fetch_embedding("known", **CRITERIA)
    return adapter.fetch_all_embeddings(**CRITERIA)


@pytest.mark.parametrize("name", ["insert", "search", "fetch", "all"])
@pytest.mark.parametrize("fail", [False, True])
def test_each_data_operation_closes_its_sdk_index(endpoint, name, fail):
    adapter = PineconeClient(connection=endpoint["client"])
    # An empty listing isolates connection cleanup from SDK listing-shape compatibility.
    endpoint["empty"] = name == "all"
    if fail:
        endpoint["failure"] = {
            "insert": "/vectors/upsert",
            "search": "/query",
            "fetch": "/vectors/fetch",
            "all": "/vectors/list",
        }[name]
        with pytest.raises(Exception, match="injected data-plane error"):
            operation(adapter, name)
    else:
        result = operation(adapter, name)
        if name == "insert":
            assert result == 1
        elif name == "fetch":
            assert result["img_name"] == "known.png"
            assert result["embedding"] == [0.5] * 128
        elif name == "all":
            assert result == []
        else:
            assert len(result) == 1
            assert result[0]["img_name"] == "known.png"
    assert len(endpoint["indexes"]) == 1
    assert endpoint["indexes"][0]["close_calls"] == 1
    assert endpoint["client_close_calls"] == 0
    # A caller-owned control-plane client remains usable after the operation.
    assert endpoint["client"].has_index(INDEX_NAME)


def test_explicit_adapter_close_releases_control_plane(endpoint):
    adapter = PineconeClient(connection=endpoint["client"])
    assert endpoint["client"].has_index(INDEX_NAME)
    adapter.close()
    assert endpoint["client_close_calls"] == int(endpoint["supports_close"])


@pytest.mark.parametrize("owned", [False, True])
def test_public_search_only_closes_owned_control_plane(endpoint, monkeypatch, owned):
    monkeypatch.setattr(pinecone, "Pinecone", lambda **kwargs: endpoint["client"])
    monkeypatch.setattr(
        datastore,
        "represent",
        lambda **kwargs: [{"embedding": [0.5] * 128, "facial_area": {"x": 1, "y": 2}}],
    )
    result = datastore.search(
        "precomputed",
        model_name="Facenet",
        detector_backend="skip",
        database_type="pinecone",
        search_method="ann",
        similarity_search=True,
        connection=None if owned else endpoint["client"],
        connection_details="loopback-test-key",
        return_type="dict",
    )
    assert result[0][0]["img_name"] == "known.png"
    assert endpoint["indexes"][0]["close_calls"] == 1
    assert endpoint["client_close_calls"] == int(owned and endpoint["supports_close"])


def test_missing_identity_index_never_opens_a_data_client(endpoint):
    endpoint["missing"] = True
    adapter = PineconeClient(connection=endpoint["client"])
    assert adapter.fetch_embedding("unknown", **CRITERIA) is None
    assert endpoint["indexes"] == []


def test_empty_registration_never_opens_a_data_client(endpoint):
    adapter = PineconeClient(connection=endpoint["client"])
    with pytest.raises(ValueError, match="No embeddings"):
        adapter.insert_embeddings([])
    assert endpoint["indexes"] == []


def test_repeated_searches_close_native_sdk_handles(endpoint):
    adapter = PineconeClient(connection=endpoint["client"])
    for _ in range(20):
        assert adapter.search_by_vector(vector=[0.5] * 128, **CRITERIA)
    deadline = time.monotonic() + 2
    while len(endpoint["active"]) > 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    print(
        json.dumps(
            {
                "pinecone": pinecone.__version__,
                "searches": 20,
                "sdk_indexes_closed": sum(item["close_calls"] for item in endpoint["indexes"]),
                "live_http_connections_including_control_plane": len(endpoint["active"]),
            }
        )
    )
    assert [item["close_calls"] for item in endpoint["indexes"]] == [1] * 20
    # SDK 8 closes its worker pool but does not clear its native HTTP connection pool.
    # Socket release is an SDK 10 guarantee; both versions must call native close().
    if int(pinecone.__version__.split(".")[0]) >= 10:
        assert len(endpoint["active"]) <= 1
