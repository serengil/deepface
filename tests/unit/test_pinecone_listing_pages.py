"""Pinecone gallery listings must be read from both legacy and current SDK page shapes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deepface.modules.database.pinecone import PineconeClient

CRITERIA = dict(model_name="Facenet", detector_backend="skip", aligned=True, l2_normalized=False)


def build_client(pages):
    client = PineconeClient.__new__(PineconeClient)
    client.client = Mock()
    client.client.has_index.return_value = True
    index = client.client.Index.return_value
    index.list.return_value = iter(pages)
    index.fetch.side_effect = lambda ids: {
        "vectors": {
            identity: {
                "id": identity,
                "values": [float(len(identity))] * 3,
                "metadata": {"img_name": f"{identity}.png"},
            }
            for identity in ids
        }
    }
    return client, index


def current_page(*ids):
    """SDK 7+ yields response pages exposing a `vectors` list of items with an `id`."""
    return SimpleNamespace(vectors=[SimpleNamespace(id=identity) for identity in ids])


@pytest.mark.parametrize(
    "pages",
    [
        [current_page("a", "b"), current_page("c")],
        [["a", "b"], ["c"]],
    ],
    ids=["current-sdk-pages", "legacy-id-lists"],
)
def test_fetch_all_embeddings_reads_every_listing_page(pages):
    client, index = build_client(pages)

    rows = client.fetch_all_embeddings(**CRITERIA, batch_size=2)

    assert [row["id"] for row in rows] == ["a", "b", "c"]
    assert [row["img_name"] for row in rows] == ["a.png", "b.png", "c.png"]
    assert [call.kwargs["ids"] for call in index.fetch.call_args_list] == [["a", "b"], ["c"]]


def test_fetch_all_embeddings_with_empty_gallery():
    client, index = build_client([current_page()])

    assert client.fetch_all_embeddings(**CRITERIA) == []
    index.fetch.assert_not_called()
