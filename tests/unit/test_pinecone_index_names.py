"""All supported model/detector settings must produce valid Pinecone index names."""

import re
from itertools import product
from unittest.mock import Mock, patch

from deepface.modules.database.pinecone import PineconeClient
from deepface.modules.modeling import AVAILABLE_MODELS

name_for = PineconeClient._PineconeClient__generate_index_name


def test_supported_configurations_have_valid_distinct_names():
    names = set()
    for model, detector, aligned, normalized in product(
        AVAILABLE_MODELS["facial_recognition"],
        AVAILABLE_MODELS["face_detector"],
        (True, False),
        (True, False),
    ):
        name = name_for(model, detector, aligned, normalized)
        assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,43}[a-z0-9])?", name), name
        assert name not in names, (model, detector, aligned, normalized)
        names.add(name)


def test_existing_valid_names_remain_unchanged():
    assert (
        name_for("VGG-Face", "opencv", True, False)
        == "embeddings-vggface-opencv-aligned-raw"
    )
    assert (
        name_for("Facenet", "retinaface", False, True)
        == "embeddings-facenet-retinaface-unaligned-norm"
    )


def test_long_names_keep_criteria_distinct():
    names = {
        name_for("GhostFaceNet", "retinaface", False, value) for value in (False, True)
    }
    assert len(names) == 2
    assert all(len(name) <= 45 for name in names)


def test_collection_creation_uses_the_same_name_as_reads():
    client = PineconeClient.__new__(PineconeClient)
    client.client = Mock()
    client.client.has_index.return_value = False
    client.serverless_spec = Mock()
    with patch(
        "deepface.modules.database.pinecone.build_model",
        return_value=Mock(output_shape=512),
    ):
        client.initialize_database(
            model_name="Buffalo_L", detector_backend="retinaface", aligned=False
        )
    created = client.client.create_index.call_args.kwargs["name"]
    assert created == name_for("Buffalo_L", "retinaface", False, False)
    assert "_" not in created
    assert len(created) <= 45
