"""Check Pinecone listing interoperability through actual SDK HTTP requests."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

pinecone = pytest.importorskip("pinecone")

from deepface.modules import datastore
from deepface.modules.database.pinecone import PineconeClient

CRITERIA = dict(model_name="Facenet", detector_backend="skip", aligned=True, l2_normalized=False)
INDEX_NAME = "embeddings-facenet-skip-aligned-raw"


@pytest.fixture
def gallery(monkeypatch):
    state = {"ids": [], "list_requests": [], "fetch_requests": [], "failure": None}
    indexes = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            path = parsed.path
            status = 200
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
            if state["failure"] == path:
                value = {"code": 3, "message": "injected gallery read error"}
                status = 400
            elif path == "/indexes":
                value = {"indexes": [description]}
            elif path.startswith("/indexes/"):
                value = description
            elif path == "/vectors/list":
                offset = int(query.get("paginationToken", ["0"])[0])
                state["list_requests"].append(offset)
                page_ids = state["ids"][offset : offset + 2]
                next_offset = offset + len(page_ids)
                value = {
                    "vectors": [{"id": identity} for identity in page_ids],
                    "namespace": "",
                }
                if next_offset < len(state["ids"]):
                    value["pagination"] = {"next": str(next_offset)}
            elif path == "/vectors/fetch":
                ids = query.get("ids", [])
                state["fetch_requests"].append(ids)
                value = {
                    "vectors": {
                        identity: {
                            "id": identity,
                            "values": [float(state["ids"].index(identity))] * 128,
                            "metadata": {
                                "img_name": identity + ".png",
                                "face_hash": "face-" + identity,
                                "embedding_hash": "embedding-" + identity,
                            },
                        }
                        for identity in ids
                    },
                    "namespace": "",
                }
            else:
                value = {"code": 5, "message": f"Unexpected route {path}"}
                status = 404
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = pinecone.Pinecone(api_key=f"loopback-gallery-{server.server_port}", host=state["url"])
    factory = client.Index

    def open_index(*_args, **_kwargs):
        # Isolate SDK 8 name-resolution caching from the listing protocol under test.
        index = factory(host=state["url"])
        indexes.append(index)
        return index

    monkeypatch.setattr(client, "Index", open_index)
    state["client"] = client
    try:
        yield state
    finally:
        for index in indexes:
            index.close()
        close = getattr(client, "close", None)
        if close is not None:
            close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("count,batch_size", [(1, 1), (3, 1), (5, 2), (5, 1000)])
def test_fetch_all_follows_pages_and_preserves_record_ids(gallery, count, batch_size):
    gallery["ids"] = ["a", "b-with-dash", "c/slash", "d space", "e"][0:count]
    result = PineconeClient(connection=gallery["client"]).fetch_all_embeddings(
        **CRITERIA, batch_size=batch_size
    )
    assert [row["id"] for row in result] == gallery["ids"]
    for number, row in enumerate(result):
        identity = gallery["ids"][number]
        assert row == {
            "id": identity,
            "embedding": [float(number)] * 128,
            "img_name": identity + ".png",
            "face_hash": "face-" + identity,
            "embedding_hash": "embedding-" + identity,
        }
    assert gallery["list_requests"] == list(range(0, count, 2))
    assert gallery["fetch_requests"] == [
        gallery["ids"][start : start + batch_size] for start in range(0, count, batch_size)
    ]


def test_empty_gallery_does_not_fetch_vectors(gallery):
    assert PineconeClient(connection=gallery["client"]).fetch_all_embeddings(**CRITERIA) == []
    assert gallery["fetch_requests"] == []


@pytest.mark.parametrize("path", ["/vectors/list", "/vectors/fetch"])
def test_sdk_read_errors_are_preserved(gallery, path):
    gallery["ids"] = ["known"]
    gallery["failure"] = path
    with pytest.raises(Exception, match="injected gallery read error"):
        PineconeClient(connection=gallery["client"]).fetch_all_embeddings(**CRITERIA)


def test_public_exact_search_reads_native_sdk_pages(gallery, monkeypatch):
    gallery["ids"] = ["closest", "near", "far"]
    monkeypatch.setattr(
        datastore,
        "represent",
        lambda **kwargs: [{"embedding": [0.0] * 128, "facial_area": {"x": 2, "y": 3}}],
    )
    result = datastore.search(
        "precomputed",
        model_name="Facenet",
        detector_backend="skip",
        database_type="pinecone",
        search_method="exact",
        distance_metric="euclidean",
        similarity_search=True,
        connection=gallery["client"],
        return_type="dict",
    )
    assert [row["id"] for row in result[0]] == gallery["ids"]
    assert [row["distance"] for row in result[0]] == pytest.approx(
        [0, 128**0.5, 2 * 128**0.5], abs=1e-5
    )
