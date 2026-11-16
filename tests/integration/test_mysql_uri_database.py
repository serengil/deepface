"""MySQL URI database components must decode once before opening the real database."""

import os
import uuid
from urllib.parse import quote, urlsplit

import numpy as np
import pytest

from deepface.modules.database.mysql import MySqlClient, _parse_uri

CRITERIA = dict(model_name="Facenet", detector_backend="skip", aligned=True, l2_normalized=False)


@pytest.mark.parametrize("suffix", ["plain", "with space", "literal%20", "faces#1?set", "人脸库"])
@pytest.mark.parametrize("use_uri", [False, True], ids=["dict-control", "uri"])
def test_mysql_connects_to_the_exact_database_name(suffix, use_uri):
    pymysql = pytest.importorskip("pymysql")
    uri = os.environ.get("DEEPFACE_MYSQL_URI")
    if not uri:
        pytest.skip("set DEEPFACE_MYSQL_URI to an isolated server with CREATE DATABASE permission")
    name = "deepface_" + uuid.uuid4().hex[:10] + "_" + suffix
    administrator = pymysql.connect(**dict(_parse_uri(uri), autocommit=True))
    client = None
    try:
        with administrator.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{name}` CHARACTER SET utf8mb4")

        details = (
            urlsplit(uri)._replace(path="/" + quote(name, safe="")).geturl()
            if use_uri
            else dict(_parse_uri(uri), database=name)
        )
        client = MySqlClient(connection_details=details)

        with client.conn.cursor() as cursor:
            cursor.execute("SELECT DATABASE()")
            assert cursor.fetchone()[0] == name
        record = dict(
            img_name="synthetic-gallery.png",
            face=np.zeros((2, 2, 3), dtype=np.uint8),
            embedding=[1.0, 0.25],
            **CRITERIA,
        )
        assert client.insert_embeddings([record]) == 1
        rows = client.fetch_all_embeddings(**CRITERIA)
        assert len(rows) == 1
        assert rows[0]["img_name"] == record["img_name"]
        assert rows[0]["embedding"] == record["embedding"]
    finally:
        if client is not None:
            client.close()
        with administrator.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS `{name}`")
        administrator.close()
