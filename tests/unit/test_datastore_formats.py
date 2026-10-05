# built-in dependencies
import random
import sys

# 3rd party dependencies
import pytest

# project dependencies
from deepface import DeepFace
from deepface.modules import recognition
from deepface.modules.filestore import build_file_store
from deepface.commons.logger import Logger

logger = Logger()

save_representations = getattr(recognition, "__save_representations")
load_representations = getattr(recognition, "__load_representations")


def __representations(size: int, dimension: int):
    representations = []
    for i in range(size):
        representations.append(
            {
                "identity": f"dataset/ğüşıöç/img{i}.jpg",
                "hash": f"{i:040x}",
                # every 3rd image has no face
                "embedding": (
                    None if i % 3 == 0 else [random.gauss(0, 1) / 7 for _ in range(dimension)]
                ),
                "target_x": i,
                "target_y": i + 1,
                "target_w": 100 + i,
                "target_h": 200 + i,
            }
        )
    return representations


@pytest.mark.parametrize("datastore_format", sorted(recognition.DATASTORE_FORMATS))
@pytest.mark.parametrize("size", [0, 1, 10])
def test_datastore_round_trip_is_bit_exact(tmp_path, datastore_format: str, size: int):
    if datastore_format in recognition.PYARROW_DATASTORE_FORMATS:
        pytest.importorskip("pyarrow")

    store = build_file_store(str(tmp_path))
    datastore_path = store.join(f"ds.{datastore_format}")
    representations = __representations(size=size, dimension=4096)

    save_representations(
        store=store,
        datastore_path=datastore_path,
        datastore_format=datastore_format,
        representations=representations,
    )
    loaded = load_representations(
        store=store, datastore_path=datastore_path, datastore_format=datastore_format
    )

    # list equality on python floats is bit-exact
    assert loaded == representations
    logger.info(f"✅ {datastore_format} datastore round trip with {size} items is bit-exact")


@pytest.mark.parametrize("datastore_format", sorted(recognition.PYARROW_DATASTORE_FORMATS))
def test_pyarrow_formats_require_pyarrow(monkeypatch, datastore_format: str):
    # None in sys.modules makes the import raise ModuleNotFoundError
    for module in ["pyarrow", "pyarrow.feather", "pyarrow.parquet"]:
        monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(ImportError, match="pip install pyarrow"):
        DeepFace.find(
            img_path="dataset/img1.jpg", db_path="dataset", datastore_format=datastore_format
        )
    logger.info(f"✅ {datastore_format} datastore requires pyarrow")
