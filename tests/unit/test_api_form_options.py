import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_cors")

from flask import Flask

from deepface.api.src.modules.core.routes import read_options


@pytest.mark.parametrize("key", ["align", "enforce_detection", "anti_spoofing", "l2_normalize", "similarity_search"])
@pytest.mark.parametrize("value, expected", [("false", False), ("0", False), ("true", True), ("1", True)])
def test_form_booleans(key, value, expected):
    with Flask(__name__).test_request_context(method="POST", data={key: value}):
        assert read_options()[key] is expected


@pytest.mark.parametrize("payload", [[], 1, {"align": "sometimes"}])
def test_invalid_options(payload):
    with Flask(__name__).test_request_context(method="POST", json=payload):
        with pytest.raises(ValueError):
            read_options()
