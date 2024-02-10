import json
from deepface import DeepFace


def test_version():
    with open("../package_info.json", "r", encoding="utf-8") as f:
        package_info = json.load(f)

    assert DeepFace.__version__ == package_info["version"]
