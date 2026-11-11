"""A failed gallery save must leave the previous readable cache intact."""

import builtins
import os

import pytest

from deepface.modules.filestore import LocalFileStore


class InterruptedWrite:
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def write(self, content):
        self.file.write(content[:3])
        raise OSError("simulated disk full")


def interrupt_writes(monkeypatch):
    real_open, real_fdopen = builtins.open, os.fdopen

    def open_file(*args, **kwargs):
        file = real_open(*args, **kwargs)
        return (
            InterruptedWrite(file)
            if "w" in (args[1] if len(args) > 1 else "r")
            else file
        )

    monkeypatch.setattr(builtins, "open", open_file)
    monkeypatch.setattr(
        os,
        "fdopen",
        lambda *args, **kwargs: InterruptedWrite(real_fdopen(*args, **kwargs)),
    )


@pytest.mark.parametrize("previous", [None, b"previous complete archive"])
def test_interrupted_save_preserves_previous_cache(tmp_path, monkeypatch, previous):
    target = tmp_path / "gallery.npz"
    if previous is not None:
        target.write_bytes(previous)
    interrupt_writes(monkeypatch)
    with pytest.raises(OSError, match="disk full"):
        LocalFileStore(str(tmp_path)).write_bytes(str(target), b"replacement archive")
    if previous is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == previous
    assert list(tmp_path.iterdir()) == ([] if previous is None else [target])


def test_successful_save_replaces_contents_and_keeps_permissions(tmp_path):
    target = tmp_path / "gallery.npz"
    target.write_bytes(b"old")
    target.chmod(0o640)
    store = LocalFileStore(str(tmp_path))
    store.write_bytes(str(target), b"new archive")
    assert target.read_bytes() == b"new archive"
    if os.name != "nt":
        assert target.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.skipif(
    os.name == "nt", reason="creating Windows symlinks needs additional privileges"
)
def test_linked_cache_keeps_its_target(tmp_path):
    target, link = tmp_path / "actual.npz", tmp_path / "gallery.npz"
    target.write_bytes(b"old")
    link.symlink_to(target.name)
    LocalFileStore(str(tmp_path)).write_bytes(str(link), b"new")
    assert link.is_symlink()
    assert target.read_bytes() == b"new"
