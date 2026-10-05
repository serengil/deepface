# 3rd party dependencies
import pytest

# project dependencies
from deepface.commons import package_utils
from deepface.commons.logger import Logger

logger = Logger()


@pytest.fixture
def pandas_missing(monkeypatch):
    real_find_spec = package_utils.importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "pandas":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(package_utils.importlib.util, "find_spec", find_spec)


def test_default_return_type_is_pandas_if_installed():
    assert package_utils.resolve_dataframe_return_type(None) == "pandas"
    assert package_utils.resolve_dataframe_return_type("pandas") == "pandas"
    assert package_utils.resolve_dataframe_return_type("dict") == "dict"
    logger.info("✅ test default return type if pandas installed done")


def test_default_return_type_is_dict_if_pandas_missing(pandas_missing):
    assert package_utils.resolve_dataframe_return_type(None) == "dict"
    assert package_utils.resolve_dataframe_return_type("dict") == "dict"
    logger.info("✅ test default return type if pandas missing done")


def test_pandas_return_type_raises_if_pandas_missing(pandas_missing):
    with pytest.raises(ImportError, match="pip install pandas.*return_type to 'dict'"):
        package_utils.resolve_dataframe_return_type("pandas")
    logger.info("✅ test pandas return type if pandas missing done")


def test_unsupported_return_type():
    with pytest.raises(ValueError, match="Unsupported return_type"):
        package_utils.resolve_dataframe_return_type("numpy")
    logger.info("✅ test unsupported return type done")
