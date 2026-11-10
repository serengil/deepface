import ftplib
from unittest.mock import Mock

import pytest

from deepface.modules.filestore import FtpFileStore


def test_credentials_from_environment_are_literal(monkeypatch):
    monkeypatch.setenv("DEEPFACE_FTP_USER", "photos%40local")
    monkeypatch.setenv("DEEPFACE_FTP_PASSWORD", "secret%2Fvalue")
    ftp = Mock()
    monkeypatch.setattr(ftplib, "FTP", lambda: ftp)

    FtpFileStore("ftp://localhost/gallery").validate()
    ftp.login.assert_called_once_with(user="photos%40local", passwd="secret%2Fvalue")

    FtpFileStore("ftp://photos%40local:secret%2Fvalue@localhost/gallery").validate()
    assert ftp.login.call_args.kwargs == {"user": "photos@local", "passwd": "secret/value"}


@pytest.mark.parametrize("port", ["", ":2121"])
def test_ipv6_identity_can_be_read(port):
    store = FtpFileStore(f"ftp://user:secret@[::1]{port}/gallery")
    ftp = Mock()
    store.ftp = ftp
    uri = store.join("photo.jpg")

    assert uri == f"ftp://[::1]{port}/gallery/photo.jpg"
    store.read_bytes(uri)
    assert ftp.retrbinary.call_args.args[0] == "RETR /gallery/photo.jpg"


def test_wrong_port_is_rejected_without_logging_credentials():
    store = FtpFileStore("ftp://localhost:2121/gallery")
    store.ftp = Mock()
    with pytest.raises(ValueError) as error:
        store.read_bytes("ftp://remote:secret@localhost:21/gallery/photo.jpg")
    assert "secret" not in str(error.value)
    store.ftp.retrbinary.assert_not_called()


def test_failed_login_closes_new_socket_and_allows_retry(monkeypatch):
    rejected, ready = Mock(), Mock()
    failure = ftplib.error_perm("530 Login incorrect")
    rejected.login.side_effect = failure
    monkeypatch.setattr(ftplib, "FTP", Mock(side_effect=[rejected, ready]))
    store = FtpFileStore("ftp://localhost/gallery")

    with pytest.raises(ftplib.error_perm) as error:
        store.read_bytes(store.join("photo.jpg"))
    assert error.value is failure
    rejected.close.assert_called_once_with()

    store.validate()
    ready.cwd.assert_called_once_with("/gallery")


def test_stale_connection_is_closed_before_replacement(monkeypatch):
    store = FtpFileStore("ftp://localhost/gallery")
    stale, ready = Mock(), Mock()
    stale.voidcmd.side_effect = EOFError("lost connection")
    store.ftp = stale
    monkeypatch.setattr(ftplib, "FTP", lambda: ready)
    store.validate()
    stale.close.assert_called_once_with()
    ready.cwd.assert_called_once_with("/gallery")


@pytest.mark.parametrize("reply", ["500 Unknown command", "530 Not logged in"])
def test_size_protocol_errors_propagate(reply):
    store = FtpFileStore("ftp://localhost/gallery")
    store.ftp = Mock()
    store.ftp.size.side_effect = ftplib.error_perm(reply)
    with pytest.raises(ftplib.error_perm, match=reply):
        store.exists(store.join("photo.jpg"))
    store.ftp.size.side_effect = ftplib.error_perm("550 File unavailable")
    assert not store.exists(store.join("photo.jpg"))
