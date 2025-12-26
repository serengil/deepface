# built-in dependencies
import os
from unittest.mock import patch

# 3rd party dependencies
import pytest
import psycopg
from deepface import DeepFace
from tqdm import tqdm
import pandas as pd

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


# pylint: disable=unused-argument
@pytest.fixture
def flush_data():
    conn = psycopg.connect(**connection_details_dict)
    cur = conn.cursor()
    cur.execute("DELETE FROM embeddings;")
    conn.commit()
    cur.close()
    conn.close()
    logger.info("🗑️ Embeddings data flushed.")


@pytest.fixture
def load_data():
    conn = psycopg.connect(**connection_details_dict)

    # collect items
    database_items = []
    for dirpath, dirnames, filenames in os.walk("../unit/dataset"):
        for filename in filenames:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            filepath = os.path.join(dirpath, filename)
            database_items.append(filepath)

    for img_path in tqdm(database_items):
        _ = DeepFace.register(
            img=img_path,
            model_name="Facenet",
            detector_backend="mtcnn",
            connection=conn,
        )

    conn.close()

    logger.info(f"✅ Data with size {len(database_items)} loaded into Postgres for search tests.")


def test_postgres_search(flush_data, load_data):
    conn = psycopg.connect(**connection_details_dict)

    target_path = "dataset/target.jpg"

    # we loaded data for Facenet and mtcnn, not opencv
    with pytest.raises(ValueError, match="No embeddings found in the database for the criteria"):
        _ = DeepFace.search(
            img=target_path,
            model_name="Facenet",
            detector_backend="opencv",
            connection=conn,
        )

    dfs = DeepFace.search(
        img=target_path,
        model_name="Facenet",
        distance_metric="euclidean",
        detector_backend="mtcnn",
        connection=conn,
    )

    assert isinstance(dfs, list)
    assert len(dfs) == 1

    for df in dfs:
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        logger.info(df)

    conn.close()
    logger.info("✅ Postgres search test passed.")
