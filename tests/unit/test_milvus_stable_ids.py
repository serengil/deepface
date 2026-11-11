"""Persistent Milvus IDs must not depend on Python's per-process hash seed."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from deepface.modules.database.milvus import MilvusClient


def inserted_ids(embeddings):
    client = MilvusClient.__new__(MilvusClient)
    client.client = Mock()
    client.client.has_collection.return_value = True
    client.insert_embeddings(embeddings)
    return [row["id"] for row in client.client.insert.call_args.kwargs["data"]]


def record(value=0.25):
    return {
        "face": np.zeros((2, 2, 3), dtype=np.float32),
        "embedding": [value, 0.5],
        "img_name": "sample.png",
        "model_name": "VGG-Face",
        "detector_backend": "opencv",
        "aligned": True,
        "l2_normalized": False,
    }


def test_id_is_stable_across_processes():
    script = """
import json
from tests.unit.test_milvus_stable_ids import inserted_ids, record
print(json.dumps(inserted_ids([record()])))
"""
    values = []
    for seed in ("1", "2", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        values.append(
            json.loads(
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    env=env,
                    cwd=Path(__file__).resolve().parents[2],
                )
            )
        )
    assert values[0] == values[1] == values[2]


def test_id_stays_in_signed_int64_and_distinguishes_embeddings():
    first, second = inserted_ids([record(), record(0.75)])
    assert 0 <= first < 2**63
    assert 0 <= second < 2**63
    assert first != second
    assert inserted_ids([record(), record()]) == [first, first]
