"""Keep image-path identifiers through real image loading and SQLite registration."""

import base64
import io
import sqlite3
import uuid

import cv2
import numpy as np
import pytest

from deepface.modules import datastore, modeling


class SmallModel:
    input_shape = (8, 8)
    output_shape = 128

    def forward(self, images):
        vectors = np.repeat(images.mean(axis=(1, 2, 3))[:, None], 128, axis=1).tolist()
        return vectors[0] if len(vectors) == 1 else vectors


@pytest.fixture
def gallery(monkeypatch):
    monkeypatch.setattr(modeling, "build_model", lambda **kwargs: SmallModel())
    connection = sqlite3.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def image_file(tmp_path, extension, index=0):
    image = np.full((12, 10, 3), 70 + index * 20, dtype=np.uint8)
    success, encoded = cv2.imencode("." + extension.lower(), image)
    assert success
    path = tmp_path / f"Photo{index}.{extension}"
    path.write_bytes(encoded.tobytes())
    return str(path)


def register(connection, source, **kwargs):
    return datastore.register(
        img=source,
        model_name="Facenet",
        detector_backend="skip",
        database_type="sqlite",
        connection=connection,
        **kwargs,
    )


@pytest.mark.parametrize(
    "extension", ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG", "JpG", "JpEg", "PnG"]
)
def test_registration_and_search_preserve_original_path(gallery, tmp_path, extension):
    source = image_file(tmp_path, extension)
    assert register(gallery, source) == {"inserted": 1}
    assert gallery.execute("SELECT img_name FROM embeddings").fetchone()[0] == source
    results = datastore.search(
        source,
        model_name="Facenet",
        detector_backend="skip",
        database_type="sqlite",
        connection=gallery,
        distance_metric="euclidean",
        similarity_search=True,
        return_type="dict",
    )
    assert results[0][0]["img_name"] == source
    assert results[0][0]["distance"] == 0


def test_mixed_case_batch_retains_each_original_filename(gallery, tmp_path):
    sources = [
        image_file(tmp_path, extension, i) for i, extension in enumerate(["JPG", "png", "JpEg"])
    ]
    assert register(gallery, sources) == {"inserted": 3}
    names = [row[0] for row in gallery.execute("SELECT img_name FROM embeddings ORDER BY id")]
    assert names == sources


@pytest.mark.parametrize("extension", ["jpg", "JPG"])
def test_explicit_name_still_overrides_path(gallery, tmp_path, extension):
    assert register(gallery, image_file(tmp_path, extension), img_name="chosen-name") == {
        "inserted": 1
    }
    assert gallery.execute("SELECT img_name FROM embeddings").fetchone()[0] == "chosen-name"


@pytest.mark.parametrize("kind", ["array", "file-object", "base64"])
def test_inputs_without_path_names_still_receive_uuid(gallery, tmp_path, kind):
    path = image_file(tmp_path, "png")
    with open(path, "rb") as file:
        encoded = file.read()
    if kind == "array":
        source = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    elif kind == "file-object":
        source = io.BytesIO(encoded)
    else:
        source = "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")
    assert register(gallery, source) == {"inserted": 1}
    name = gallery.execute("SELECT img_name FROM embeddings").fetchone()[0]
    assert str(uuid.UUID(name)) == name
