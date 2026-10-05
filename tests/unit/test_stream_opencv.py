# 3rd party dependencies
import pytest

# project dependencies
from deepface.modules import streaming
from deepface.commons.logger import Logger

logger = Logger()


class ValidationPassed(Exception):
    pass


def test_streaming_requires_opencv_python(monkeypatch):
    def distribution(name):
        raise streaming.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(streaming.metadata, "distribution", distribution)

    with pytest.raises(ImportError, match="pip install opencv-contrib-python"):
        streaming.analysis(db_path="dataset")
    logger.info("✅ test streaming requires opencv-python done")


@pytest.mark.parametrize("package", ["opencv-python", "opencv-contrib-python"])
def test_streaming_accepts_opencv_gui_packages(monkeypatch, package):
    def distribution(name):
        if name != package:
            raise streaming.metadata.PackageNotFoundError(name)
        return object()

    monkeypatch.setattr(streaming.metadata, "distribution", distribution)
    # validation passes, so analysis moves on to build models; stop it there
    def build_demography_models(**kwargs):
        raise ValidationPassed()

    monkeypatch.setattr(streaming, "build_demography_models", build_demography_models)

    with pytest.raises(ValidationPassed):
        streaming.analysis(db_path="dataset")
    logger.info(f"✅ test streaming accepts {package} done")
