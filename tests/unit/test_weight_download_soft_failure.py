# built-in dependencies
from pathlib import Path
from unittest.mock import Mock

# 3rd party dependencies
import pytest

# project dependencies
from deepface.commons import weight_utils


def test_backup_is_tried_when_gdown_returns_none(monkeypatch, tmp_path):
    (tmp_path / ".deepface" / "weights").mkdir(parents=True)
    monkeypatch.setattr(weight_utils.folder_utils, "get_deepface_home", lambda: str(tmp_path))
    monkeypatch.setattr(weight_utils, "ensure_source_is_reachable", lambda url: None)
    calls = []

    def download(url, output, quiet):
        calls.append(url)
        if url == "https://example.com/primary":
            Path(output).write_bytes(b"incomplete")
            return None
        assert not Path(output).exists()
        Path(output).write_bytes(b"complete weights")
        return output

    monkeypatch.setattr(weight_utils.gdown, "download", download)
    result = weight_utils.download_weights_if_necessary(
        "weights.bin", ["https://example.com/primary", "https://example.com/backup"]
    )
    assert calls == ["https://example.com/primary", "https://example.com/backup"]
    assert Path(result).read_bytes() == b"complete weights"


def test_all_soft_failures_raise_instead_of_returning_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(weight_utils.folder_utils, "get_deepface_home", lambda: str(tmp_path))
    monkeypatch.setattr(weight_utils, "ensure_source_is_reachable", lambda url: None)
    download = Mock(return_value=None)
    monkeypatch.setattr(weight_utils.gdown, "download", download)

    with pytest.raises(ValueError, match="while downloading"):
        weight_utils.download_weights_if_necessary("weights.bin", ["first", "second"])
    assert download.call_count == 2
