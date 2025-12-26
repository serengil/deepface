# built-in dependencies
import os
from unittest.mock import patch

# 3rd party dependencies
import pytest
import psycopg
from deepface import DeepFace

# project dependencies
from deepface.commons.logger import Logger

logger = Logger()

connection_details_dict = {
    "host": "localhost",
    "port": 5433,
    "dbname": "deepface",
    "user": "deepface_user",
    "password": "deepface_pass",
}

connection_details_str = "postgresql://deepface_user:deepface_pass@localhost:5433/deepface"


# pylint: disable=unused-argument
@pytest.fixture
def flush_data():
    conn = psycopg.connect(**connection_details_dict)
    cur = conn.cursor()
    cur.execute("DELETE FROM embeddings;")
    conn.commit()
    cur.close()
    conn.close()
    logger.debug("🗑️ Embeddings data flushed.")


def test_regsiter_with_json(flush_data):
    img_path = "../unit/dataset/img1.jpg"
    result = DeepFace.register(
        img=img_path,
        model_name="Facenet",
        detector_backend="mtcnn",
        connection_details=connection_details_dict,
    )
    assert result["inserted"] == 1
    logger.info("✅ Registered with json test passed.")


def test_register_with_string(flush_data):
    img_path = "../unit/dataset/img1.jpg"
    result = DeepFace.register(
        img=img_path,
        model_name="Facenet",
        detector_backend="mtcnn",
        connection_details=connection_details_str,
    )
    assert result["inserted"] == 1
    logger.info("✅ Registered with string test passed.")


@patch.dict(os.environ, {"DEEPFACE_POSTGRES_URI": connection_details_str})
def test_register_with_envvar(flush_data):
    img_path = "../unit/dataset/img1.jpg"
    result = DeepFace.register(
        img=img_path,
        model_name="Facenet",
        detector_backend="mtcnn",
    )
    assert result["inserted"] == 1
    logger.info("✅ Registered with env var test passed.")


def test_register_with_connection(flush_data):
    conn = psycopg.connect(**connection_details_dict)
    img_path = "../unit/dataset/img1.jpg"
    result = DeepFace.register(
        img=img_path,
        model_name="Facenet",
        detector_backend="mtcnn",
        connection=conn,
    )
    assert result["inserted"] == 1
    conn.close()
    logger.info("✅ Registered with connection test passed.")


def test_register_duplicate(flush_data):
    img1_path = "../unit/dataset/img1.jpg"
    result = DeepFace.register(
        img=img1_path,
        model_name="Facenet",
        detector_backend="mtcnn",
        connection_details=connection_details_dict,
    )
    assert result["inserted"] == 1

    # Facenet & opencv pair should have different extracted face & embedding than Facenet & mtcnn
    result = DeepFace.register(
        img=img1_path,
        model_name="Facenet",
        detector_backend="opencv",
        connection_details=connection_details_dict,
    )
    assert result["inserted"] == 1

    # Duplicate registration with same model & detector should raise error
    with pytest.raises(ValueError, match="Duplicate detected for extracted face and embedding"):
        _ = DeepFace.register(
            img=img1_path,
            model_name="Facenet",
            detector_backend="mtcnn",
            connection_details=connection_details_dict,
        )

    logger.info("✅ Duplicate registration test passed.")
