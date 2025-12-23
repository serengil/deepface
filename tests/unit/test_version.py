# built-in dependencies
import json

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


def test_version():
    with open("../package_info.json", "r", encoding="utf-8") as f:
        package_info = json.load(f)

    assert DeepFace.__version__ == package_info["version"]
    logger.info("✅ versions are matching in both package_info.json and deepface/__init__.py")
