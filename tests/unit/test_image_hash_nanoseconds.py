# built-in dependencies
from types import SimpleNamespace

# 3rd party dependencies
import pytest

# project dependencies
from deepface.commons import image_utils


@pytest.mark.parametrize("field", ["st_mtime_ns", "st_ctime_ns"])
def test_distinct_file_versions_do_not_alias_after_float_rounding(monkeypatch, field):
    # At this epoch, two neighbouring nanoseconds round to the same Python float.
    timestamp = 1_800_000_000_000_000_000
    assert timestamp / 1e9 == (timestamp + 1) / 1e9
    stat = SimpleNamespace(st_size=128, st_mtime=timestamp / 1e9, st_ctime=timestamp / 1e9,
                           st_mtime_ns=timestamp, st_ctime_ns=timestamp)
    monkeypatch.setattr(image_utils, "os", SimpleNamespace(stat=lambda path: stat))
    before = image_utils.find_image_hash("gallery/face.png")
    setattr(stat, field, timestamp + 1)
    after = image_utils.find_image_hash("gallery/face.png")
    assert before != after


def test_unchanged_real_file_keeps_the_same_hash(tmp_path):
    image = tmp_path / "face.png"
    image.write_bytes(b"image content")
    assert image_utils.find_image_hash(str(image)) == image_utils.find_image_hash(str(image))
